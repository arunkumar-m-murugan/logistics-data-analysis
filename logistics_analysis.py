"""Logistics data simulation, EDA and visualization."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

rng = np.random.default_rng(42)
N = 3000
sns.set_theme(style="whitegrid", context="notebook")
PAL = {"Road": "#2a6f97", "Rail": "#61a5c2", "Air": "#e07a5f", "Sea": "#81b29a"}

# ------------------------------------------------------------------
# 1. DATA SIMULATION
# ------------------------------------------------------------------
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
# seasonality: more shipments in Q4 (peak season)
keep = rng.random(N) < (0.75 + 0.25 * (dates.month >= 10) + 0.08 * np.isin(dates.month, [3, 4]))
dates = dates[keep]
n = len(dates)

modes = rng.choice(["Road", "Rail", "Air", "Sea"], n, p=[0.55, 0.15, 0.12, 0.18])
hubs = rng.choice(["Chennai", "Mumbai", "Delhi", "Kolkata", "Bengaluru"], n, p=[0.25, 0.25, 0.2, 0.1, 0.2])
regions = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.22, 0.25, 0.15, 0.23, 0.15])
carriers = rng.choice(["Carrier A", "Carrier B", "Carrier C", "Carrier D"], n, p=[0.35, 0.30, 0.20, 0.15])

dist_base = {"Road": 650, "Rail": 1300, "Air": 1700, "Sea": 2600}
distance = np.array([rng.gamma(4, dist_base[m] / 4) for m in modes]).clip(80, 6000)
weight = np.array([rng.lognormal({"Road": 6.2, "Rail": 7.2, "Air": 4.8, "Sea": 8.0}[m], 0.7) for m in modes]).clip(5, 40000)
units = np.maximum(1, (weight / rng.uniform(8, 25, n)).astype(int))

speed = {"Road": 55, "Rail": 40, "Air": 600, "Sea": 28}  # effective km/h incl. stops
transit_days = np.array([d / (speed[m] * 24) for d, m in zip(distance, modes)])
handling_days = {"Road": 0.5, "Rail": 1.0, "Air": 0.3, "Sea": 2.0}

dwell = rng.gamma(2.2, 7, n)  # warehouse dwell hours
dwell += np.where(np.isin(dates.month, [10, 11, 12]), rng.gamma(2, 4, n), 0)  # peak congestion
dwell += np.where(hubs == "Mumbai", rng.gamma(2, 3, n), 0)  # hub bottleneck
weather = rng.random(n) < (0.06 + 0.10 * np.isin(dates.month, [6, 7, 8, 9]))  # monsoon

base_days = np.ceil(transit_days + np.array([handling_days[m] for m in modes]) + 1)  # true process time
planned = base_days + 1.5  # promised date includes a buffer
carrier_delay = {"Carrier A": 0.0, "Carrier B": 0.15, "Carrier C": 0.45, "Carrier D": -0.10}
delay_noise = rng.normal(0, 0.6, n)
actual = (base_days
          + dwell / 24 * 0.9
          + weather * rng.gamma(2, 0.8, n)
          + np.array([carrier_delay[c] for c in carriers])
          + delay_noise).clip(0.5)
actual = np.round(actual, 1)
delay = np.round(actual - planned, 1)
on_time = delay <= 0.5

rate = {"Road": 2.1, "Rail": 1.1, "Air": 9.5, "Sea": 0.55}  # cost per tonne-km factor
fixed = {"Road": 40, "Rail": 80, "Air": 150, "Sea": 220}
fuel = rng.normal(1.0, 0.08, n) * (1 + 0.04 * np.sin(2 * np.pi * (dates.dayofyear / 365)))
cost = np.array([fixed[m] + rate[m] * d * (w / 1000) ** 0.55 for m, d, w in zip(modes, distance, weight)])
cost = cost * fuel * rng.lognormal(0, 0.12, n) + dwell * 1.8 + weather * rng.gamma(2, 25, n)
cost = np.round(cost, 2)

rating = (4.8 - 0.55 * np.clip(delay, 0, 6) + rng.normal(0, 0.35, n)).clip(1, 5).round(1)

df = pd.DataFrame({
    "shipment_id": [f"SH{100000 + i}" for i in range(n)],
    "ship_date": dates, "origin_hub": hubs, "dest_region": regions, "carrier": carriers,
    "mode": modes, "distance_km": distance.round(0), "weight_kg": weight.round(0), "units": units,
    "dwell_hours": dwell.round(1), "weather_event": weather.astype(int),
    "planned_days": planned, "actual_days": actual, "delay_days": delay,
    "on_time": on_time.astype(int), "transport_cost": cost, "customer_rating": rating,
})
df["month"] = df["ship_date"].dt.month
df["cost_per_km"] = df["transport_cost"] / df["distance_km"]
df["cost_per_kg"] = df["transport_cost"] / df["weight_kg"]
df.to_csv("/home/claude/logistics_data.csv", index=False)

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
S = {"n": int(n), "missing": int(df.isna().sum().sum())}
num_cols = ["distance_km", "weight_kg", "units", "dwell_hours", "planned_days", "actual_days",
            "delay_days", "transport_cost", "customer_rating"]
desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["skew"] = df[num_cols].skew()
S["desc"] = desc.round(2).reset_index().rename(columns={"index": "variable"}).to_dict("records")

S["on_time_rate"] = round(df.on_time.mean() * 100, 1)
S["avg_delay"] = round(df.delay_days.mean(), 2)
S["avg_cost"] = round(df.transport_cost.mean(), 2)
S["mode_summary"] = (df.groupby("mode").agg(shipments=("shipment_id", "count"),
                     avg_days=("actual_days", "mean"), avg_cost=("transport_cost", "mean"),
                     cost_per_km=("cost_per_km", "mean"), on_time=("on_time", "mean"))
                     .round(2).reset_index().to_dict("records"))
S["carrier_summary"] = (df.groupby("carrier").agg(shipments=("shipment_id", "count"),
                        avg_delay=("delay_days", "mean"), on_time=("on_time", "mean"),
                        rating=("customer_rating", "mean")).round(2).reset_index().to_dict("records"))
S["hub_summary"] = (df.groupby("origin_hub").agg(shipments=("shipment_id", "count"),
                    avg_dwell=("dwell_hours", "mean"), avg_delay=("delay_days", "mean"),
                    on_time=("on_time", "mean")).round(2).reset_index().to_dict("records"))
S["monthly"] = (df.groupby("month").agg(shipments=("shipment_id", "count"), on_time=("on_time", "mean"),
                avg_dwell=("dwell_hours", "mean"), avg_cost=("transport_cost", "mean")).round(3)
                .reset_index().to_dict("records"))

corr_cols = ["distance_km", "weight_kg", "dwell_hours", "planned_days", "actual_days",
             "delay_days", "transport_cost", "customer_rating"]
corr = df[corr_cols].corr(method="spearman")
S["corr_pairs"] = {
    "dwell_delay": round(corr.loc["dwell_hours", "delay_days"], 2),
    "delay_rating": round(corr.loc["delay_days", "customer_rating"], 2),
    "dist_cost": round(corr.loc["distance_km", "transport_cost"], 2),
    "weight_cost": round(corr.loc["weight_kg", "transport_cost"], 2),
    "dwell_cost": round(corr.loc["dwell_hours", "transport_cost"], 2),
}
S["weather"] = {"delay_weather": round(df[df.weather_event == 1].delay_days.mean(), 2),
                "delay_clear": round(df[df.weather_event == 0].delay_days.mean(), 2),
                "share_weather": round(df.weather_event.mean() * 100, 1)}

# hypothesis tests
g = [x.delay_days.values for _, x in df.groupby("carrier")]
S["anova_carrier_p"] = float(stats.f_oneway(*g).pvalue)
S["ttest_weather_p"] = float(stats.ttest_ind(df[df.weather_event == 1].delay_days,
                                             df[df.weather_event == 0].delay_days, equal_var=False).pvalue)
df["dwell_bucket"] = pd.cut(df.dwell_hours, [0, 12, 24, 36, 48, 500], labels=["<12h", "12-24h", "24-36h", "36-48h", ">48h"])
S["dwell_bucket"] = (df.groupby("dwell_bucket", observed=True).agg(shipments=("shipment_id", "count"),
                     avg_delay=("delay_days", "mean"), on_time=("on_time", "mean"))
                     .round(2).reset_index().astype({"dwell_bucket": str}).to_dict("records"))
S["dwell_over_24_share"] = round((df.dwell_hours > 24).mean() * 100, 1)
S["peak_vs_off"] = {
    "peak_dwell": round(df[df.month >= 10].dwell_hours.mean(), 1),
    "off_dwell": round(df[df.month < 10].dwell_hours.mean(), 1),
    "peak_ontime": round(df[df.month >= 10].on_time.mean() * 100, 1),
    "off_ontime": round(df[df.month < 10].on_time.mean() * 100, 1),
}
# cost drivers via OLS (numpy)
X = pd.DataFrame({"const": 1.0, "distance_km": df.distance_km, "weight_t": df.weight_kg / 1000,
                  "dwell_hours": df.dwell_hours, "weather": df.weather_event})
X = pd.concat([X, pd.get_dummies(df["mode"], drop_first=True).astype(float)], axis=1)
beta, *_ = np.linalg.lstsq(X.values, df.transport_cost.values, rcond=None)
pred = X.values @ beta
r2 = 1 - ((df.transport_cost - pred) ** 2).sum() / ((df.transport_cost - df.transport_cost.mean()) ** 2).sum()
S["ols"] = {"r2": round(float(r2), 3), "coef": dict(zip(X.columns, np.round(beta, 3).tolist()))}

# ------------------------------------------------------------------
# 3. VISUALIZATIONS
# ------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"/home/claude/{name}.png", dpi=150)
    plt.close(fig)

# Fig 1: delivery time distribution by mode
fig, ax = plt.subplots(figsize=(8, 4.5))
for m in ["Road", "Rail", "Air", "Sea"]:
    sns.kdeplot(df[df["mode"] == m].actual_days, ax=ax, label=m, color=PAL[m], fill=True, alpha=0.25, clip=(0, 30))
ax.set(xlabel="Actual delivery time (days)", ylabel="Density", title="Distribution of Delivery Time by Transport Mode")
ax.legend(title="Mode")
save(fig, "fig1_delivery_kde")

# Fig 2: cost per km by mode (box, log)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df, x="mode", y="cost_per_km", order=["Road", "Rail", "Air", "Sea"], palette=PAL, hue="mode",
            legend=False, ax=ax, fliersize=2)
ax.set_yscale("log")
ax.set(xlabel="Transport mode", ylabel="Cost per km (log scale)", title="Transportation Cost per Kilometre by Mode")
save(fig, "fig2_cost_box")

# Fig 3: monthly volume and on-time rate
mo = df.groupby("month").agg(vol=("shipment_id", "count"), ot=("on_time", "mean")).reset_index()
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(mo.month, mo.vol, color="#61a5c2", alpha=0.8)
ax1.set(xlabel="Month (2025)", ylabel="Shipments", title="Monthly Shipment Volume vs On-Time Delivery Rate")
ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(mo.month, mo.ot * 100, color="#e07a5f", marker="o", lw=2)
ax2.set_ylabel("On-time rate (%)")
ax2.grid(False)
save(fig, "fig3_monthly_trend")

# Fig 4: correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title("Spearman Correlation Matrix of Key Metrics")
save(fig, "fig4_corr_heatmap")

# Fig 5: distance vs cost scatter
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.scatterplot(data=df, x="distance_km", y="transport_cost", hue="mode", palette=PAL, alpha=0.45, s=18, ax=ax)
ax.set_yscale("log")
ax.set(xlabel="Distance (km)", ylabel="Transport cost (log scale)", title="Distance vs Transportation Cost by Mode")
save(fig, "fig5_dist_cost_scatter")

# Fig 6: carrier performance
cs = df.groupby("carrier").agg(ot=("on_time", "mean"), delay=("delay_days", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
sns.barplot(data=cs, x="carrier", y=cs.ot * 100, color="#2a6f97", ax=axes[0])
axes[0].set(ylabel="On-time rate (%)", xlabel="", title="On-Time Rate by Carrier")
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
sns.barplot(data=cs, x="carrier", y="delay", color="#e07a5f", ax=axes[1])
axes[1].set(ylabel="Mean delay (days)", xlabel="", title="Mean Delay by Carrier")
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
save(fig, "fig6_carrier")

# Fig 7: dwell vs delay bucket boxplot
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=df, x="dwell_bucket", y="delay_days", color="#81b29a", ax=ax, fliersize=2)
ax.axhline(0.5, color="grey", ls="--", lw=1)
ax.set(xlabel="Warehouse dwell time", ylabel="Delay vs plan (days)", title="Warehouse Dwell Time as a Delivery Bottleneck")
save(fig, "fig7_dwell_delay")

# Fig 8: hub x month heatmap of dwell
hm = df.pivot_table(index="origin_hub", columns="month", values="dwell_hours", aggfunc="mean")
fig, ax = plt.subplots(figsize=(9, 3.8))
sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg dwell (hrs)"})
ax.set(xlabel="Month", ylabel="", title="Average Warehouse Dwell Time by Hub and Month")
save(fig, "fig8_hub_heatmap")

# Fig 9: delay vs customer rating
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.regplot(data=df.sample(1200, random_state=1), x="delay_days", y="customer_rating", ax=ax,
            scatter_kws={"alpha": 0.25, "s": 14}, line_kws={"color": "#e07a5f"})
ax.set(xlabel="Delay vs plan (days)", ylabel="Customer rating (1-5)", title="Impact of Delivery Delay on Customer Rating")
save(fig, "fig9_delay_rating")

with open("/home/claude/stats.json", "w") as f:
    json.dump(S, f, indent=1, default=str)
print(json.dumps({k: S[k] for k in ["n", "on_time_rate", "avg_delay", "avg_cost", "corr_pairs", "weather",
      "anova_carrier_p", "ttest_weather_p", "dwell_over_24_share", "peak_vs_off", "ols"]}, indent=1))
print(pd.DataFrame(S["mode_summary"]))
print(pd.DataFrame(S["carrier_summary"]))
print(pd.DataFrame(S["hub_summary"]))
print(pd.DataFrame(S["dwell_bucket"]))
print(pd.DataFrame(S["monthly"]))
print(pd.DataFrame(S["desc"])[["variable", "mean", "50%", "std", "min", "max", "skew"]])
