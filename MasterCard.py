import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load files
tx = pd.read_csv(r"MasterClotheData.csv")
store = pd.read_excel(r"MasterFiles_v2.xlsx", sheet_name="Store Master")
comp = pd.read_excel(r"MasterFiles_v2.xlsx", sheet_name="Competitor Master")

# clean
tx["date"] = pd.to_datetime(tx["date"])
comp["opening_day"] = pd.to_datetime(comp["opening_day"])

tx["revenue"] = tx["selling_price"] * tx["units_sold"]

# filter timeframe
tx = tx[(tx["date"] >= "2025-01-01") & (tx["date"] <= "2025-06-30")]

#normalize
def norm(x):
    return str(x).strip().lower()

store["market_norm"] = store["store_market"].apply(norm)
comp["market_norm"] = comp["comp_market"].apply(norm)

#treated vs control
treated_markets = set(comp["market_norm"].unique())
store["treated"] = store["market_norm"].isin(treated_markets).astype(int)

# merge
df = tx.merge(store[["store_id", "treated"]], on="store_id", how="left")

# weekly
df["week"] = df["date"].dt.to_period("W").dt.start_time

trend = (
    df.groupby(["week", "treated"], as_index=False)
      .agg(avg_revenue=("revenue", "mean"))
)

treated = trend[trend["treated"] == 1]
control = trend[trend["treated"] == 0]

plt.figure(figsize=(11,6))

# lines
plt.plot(treated["week"], treated["avg_revenue"], linewidth=2, label="Competitor Markets")
plt.plot(control["week"], control["avg_revenue"], linewidth=2, label="No Competitor Markets")

# vertical lines for EACH competitor opening
for i, row in comp.iterrows():
    plt.axvline(row["opening_day"], linestyle="--", alpha=0.4)

# label only first few to avoid clutter
for i, row in comp.head(3).iterrows():
    plt.text(row["opening_day"],
             max(trend["avg_revenue"]) * 0.95,
             f"{row['comp_market']}",
             rotation=90,
             fontsize=8)

# labels
plt.title("Sales Trends with Competitor (2025)")
plt.xlabel("Date")
plt.ylabel("Average Weekly Revenue")

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Question 4

# competitor distance + opening date per market
comp_distance = comp.groupby("market_norm").agg(
    comp_distance=("comp_distance","min"),
    opening_day=("opening_day","min")
).reset_index()

# attach to stores
store_q4 = store.merge(comp_distance, on="market_norm", how="left")

# attach to transactions
df_q4 = tx.merge(store_q4[["store_id","comp_distance","opening_day"]], on="store_id", how="left")

# keep only stores exposed to competitors
df_q4 = df_q4[df_q4["opening_day"].notna()]

# before vs after
df_q4["post"] = df_q4["date"] >= df_q4["opening_day"]

store_sales = (
    df_q4.groupby(["store_id","post"], as_index=False)
    .agg(revenue=("revenue","sum"),
         distance=("comp_distance","first"))
)

before = store_sales[store_sales["post"] == False]
after = store_sales[store_sales["post"] == True]

store_effect = before.merge(after, on="store_id", suffixes=("_before","_after"))

# percent change
store_effect["pct_change"] = (
    (store_effect["revenue_after"] - store_effect["revenue_before"])
    / store_effect["revenue_before"]
)

#distances
store_effect["distance_bin"] = pd.cut(
    store_effect["distance_before"],
    bins=[0,7,12,15,20],
    labels=["0-7","7-12","12-17","17-21"]
)

heat_data = (
    store_effect.groupby("distance_bin")
    .agg(avg_sales_change=("pct_change","mean"))
)

heat_matrix = heat_data.values.reshape(-1,1)

#heatmap
plt.figure(figsize=(6,6))

sns.heatmap(
    heat_matrix,
    annot=True,
    cmap="coolwarm",
    yticklabels=heat_data.index,
    xticklabels=["Sales Change"],
    center=0,
    fmt=".2%"
)

plt.title("Sales Impact by Distance to Competitor")
plt.ylabel("Distance to Competitor (Miles)")

plt.tight_layout()
plt.show()