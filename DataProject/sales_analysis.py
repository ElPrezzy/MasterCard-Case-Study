import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import scipy.stats as stats

# file paths
TX_PATH     = r"C:\Users\zeusp\OneDrive\Documents\Data Science\DITW\DataProject\Scenario2_Data (1).csv.xlsx"
MASTER_PATH = r"C:\Users\zeusp\OneDrive\Documents\Data Science\DITW\DataProject\MasterFiles_v2 (1).xlsx"

# analysis window and config
DATE_START       = "2025-01-01"
DATE_END         = "2025-06-30"
CONTROL_SPLIT    = pd.Timestamp("2025-04-01")
COMPETITOR_NAME  = "The Style Spot"
CONFIDENCE_LEVEL = 0.95

# colours for the chart
COLOUR_BEFORE = "#5B8DB8"
COLOUR_AFTER  = "#B85C5C"
COLOUR_STAR   = "#444444"


class SalesAnalysis:
    """
    Analyzes the revenue impact of a competitor opening (TSS) on MasterClothe
    stores using a Difference-in-Differences (DiD) methodology.
    """

    def __init__(self, tx_path, master_path):
        self.tx_path     = tx_path
        self.master_path = master_path
        self.df          = None
        self.cat_pivot   = None

    # Step 1: LOAD & CLEAN
    def load_and_clean(self):
        """Loads all data sources, calculates revenue, and filters to analysis window."""
        try:
            transactions = pd.read_excel(self.tx_path)
            store_master = pd.read_excel(self.master_path, sheet_name="Store Master")
            comp_master  = pd.read_excel(self.master_path, sheet_name="Competitor Master")
            item_master  = pd.read_excel(self.master_path, sheet_name="Item Master")
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            raise

        comp_master["opening_day"] = pd.to_datetime(comp_master["opening_day"])
        transactions["date"]       = pd.to_datetime(transactions["date"])
        transactions["revenue"]    = transactions["selling_price"] * transactions["units_sold"]
        transactions               = transactions[transactions["date"].between(DATE_START, DATE_END)]

        print("Data loaded successfully.")
        return transactions, store_master, comp_master, item_master

    # Step 2: CLASSIFY STORES
    def classify_stores(self, store_master, comp_master):
        """Tags each store as treated (TSS nearby) or control (no TSS nearby)."""
        normalize = lambda x: str(x).strip().lower()
        store_master["market_norm"] = store_master["store_market"].apply(normalize)
        comp_master["market_norm"]  = comp_master["comp_market"].apply(normalize)

        # get earliest TSS opening date per market
        tss_open_dates = (
            comp_master[comp_master["competitor_name"] == COMPETITOR_NAME]
            .groupby("market_norm")["opening_day"]
            .min()
            .reset_index()
            .rename(columns={"opening_day": "tss_open_date"})
        )

        store_master = store_master.merge(tss_open_dates, on="market_norm", how="left")
        store_master["treated"] = store_master["tss_open_date"].notna().astype(int)

        print(f"Treatment stores: {store_master['treated'].sum()}")
        print(f"Control stores: {(store_master['treated'] == 0).sum()}")
        return store_master

    # Step 3: BUILD MERGED DF
    def build_df(self, transactions, store_master, item_master):
        merged = (
            transactions
            .merge(store_master[["store_id", "treated", "tss_open_date"]], on="store_id", how="left")
            .merge(item_master[["item_id", "parent_category"]], on="item_id", how="left")
        )
        merged["period"] = np.where(merged["date"] < merged["tss_open_date"], "Before", "After")
        self.df = merged
        print("Merged dataframe built successfully.")

    # Step 4: CALCULATE DiD
    def calculate_did(self):
        """
        Calculates DiD % change per category.
        DiD = treatment % change minus control % change,
        isolating TSS impact from natural sales trends.
        """
        treatment_stores = self.df[self.df["treated"] == 1]
        control_stores   = self.df[self.df["treated"] == 0]

        treatment_before = self._daily_avg(treatment_stores[treatment_stores["period"] == "Before"])
        treatment_after  = self._daily_avg(treatment_stores[treatment_stores["period"] == "After"])
        control_before   = self._daily_avg(control_stores[control_stores["date"] <  CONTROL_SPLIT])
        control_after    = self._daily_avg(control_stores[control_stores["date"] >= CONTROL_SPLIT])

        # merge treatment and control periods into one pivot table
        treatment = treatment_before.merge(treatment_after, on="parent_category", suffixes=("_before", "_after"))
        control   = control_before.merge(control_after, on="parent_category", suffixes=("_before", "_after"))
        pivot     = treatment.merge(control, on="parent_category", suffixes=("_treat", "_ctrl"))

        pivot["treatment_pct_change"] = (pivot["avg_after_treat"] - pivot["avg_before_treat"]) / pivot["avg_before_treat"] * 100
        pivot["control_pct_change"]   = (pivot["avg_after_ctrl"]  - pivot["avg_before_ctrl"])  / pivot["avg_before_ctrl"]  * 100
        pivot["did_pct"]              = pivot["treatment_pct_change"] - pivot["control_pct_change"]

        self.cat_pivot = pivot.sort_values("did_pct").reset_index(drop=True)
        print("Category DiD impact calculated successfully.")

    # Step 5: PLOT
    def plot(self):
        plot_data      = self.cat_pivot.sort_values("did_pct").reset_index(drop=True)
        categories     = plot_data["parent_category"].tolist()
        before_revenue = plot_data["avg_before_treat"].tolist()
        after_revenue  = plot_data["avg_after_treat"].tolist()
        did_values     = plot_data["did_pct"].tolist()

        ci_before, ci_after = self._calculate_confidence_intervals(categories)

        x, bar_width = np.arange(len(categories)), 0.28

        fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
        ax.set_facecolor("white")

        # plot before and after bars with error bars
        for revenue, ci, offset, colour, edge_colour in [
            (before_revenue, ci_before, -bar_width / 2, COLOUR_BEFORE, "#3a6690"),
            (after_revenue,  ci_after,   bar_width / 2, COLOUR_AFTER,  "#8a3a3a"),
        ]:
            ax.bar(x + offset, revenue, bar_width, color=colour, alpha=0.85,
                   yerr=ci, capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": edge_colour, "capthick": 1.2},
                   zorder=3)

        # DiD % label inside each after bar
        for i, (after_val, did_val) in enumerate(zip(after_revenue, did_values)):
            ax.text(i + bar_width / 2, 18, f"{did_val:.1f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="white", zorder=4)

        # significance star above pairs where CIs do not overlap
        for i, (before_val, after_val, ci_b, ci_a) in enumerate(
            zip(before_revenue, after_revenue, ci_before, ci_after)
        ):
            if (before_val + ci_b) < (after_val - ci_a) or (after_val + ci_a) < (before_val - ci_b):
                top = max(before_val + ci_b, after_val + ci_a) + 30
                ax.text(i, top, "*", ha="center", va="bottom",
                        fontsize=13, color=COLOUR_STAR, fontweight="bold", zorder=5)

        self._format_axes(ax, categories, x)
        plt.tight_layout()
        plt.show()

    # private helpers
    def _daily_avg(self, frame):
        n_stores = frame["store_id"].nunique()
        n_days   = frame["date"].nunique()
        return (
            frame.groupby("parent_category")["revenue"]
            .sum()
            .div(n_stores * n_days)
            .reset_index()
            .rename(columns={"revenue": "avg"})
        )

    def _calculate_confidence_intervals(self, categories):
        alpha = 1 - CONFIDENCE_LEVEL
        ci_before, ci_after = [], []

        for category in categories:
            category_data = self.df[
                (self.df["treated"] == 1) &
                (self.df["parent_category"] == category)
            ]
            for period, ci_list in [("Before", ci_before), ("After", ci_after)]:
                sample = (category_data[category_data["period"] == period]
                          .groupby(["store_id", "date"])["revenue"].sum())
                if len(sample) > 1:
                    margin = stats.sem(sample) * stats.t.ppf(1 - alpha / 2, df=len(sample) - 1)
                else:
                    margin = 0
                ci_list.append(margin)

        return ci_before, ci_after

    def _format_axes(self, ax, categories, x):
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel("Avg Daily Revenue per Store ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v):,}"))
        ax.set_title(
            "MasterClothe Revenue Decline by Category: Before vs After TSS Opening\n"
            "Treatment vs Control Stores (DiD Analysis)",
            fontsize=12, pad=14
        )
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles=[
            mpatches.Patch(color=COLOUR_BEFORE, alpha=0.85, label="Before TSS opening"),
            mpatches.Patch(color=COLOUR_AFTER,  alpha=0.85, label="After TSS opening"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOUR_STAR,
                       markersize=11, label="* Non-overlapping 95% CI (p < 0.05)")
        ], fontsize=10, loc="upper right", framealpha=0.9, edgecolor="#cccccc")
        sns.despine(top=True, right=True)

    # Step 6: RUN FULL PIPELINE
    def run(self):
        transactions, store_master, comp_master, item_master = self.load_and_clean()
        store_master = self.classify_stores(store_master, comp_master)
        self.build_df(transactions, store_master, item_master)
        self.calculate_did()
        self.plot()


#--- Run the analysis
analysis = SalesAnalysis(TX_PATH, MASTER_PATH)
analysis.run()