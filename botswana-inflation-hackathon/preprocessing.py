import pandas as pd
import numpy as np
import os

# Data path setup
DATA_DIR = "./data"

def process_daily_bdi(filepath):
    """Extracts monthly mean, volatility, and returns from Daily BDI data."""
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    
    # 1. Monthly Aggregations
    bdi_m = df.groupby("year_month").agg(
        bdi_mean=("BDI_Close", "mean"),
        bdi_std=("BDI_Close", "std"),
        bdi_max=("BDI_Close", "max"),
        bdi_min=("BDI_Close", "min")
    ).reset_index()
    
    # 2. Volatility Feature (Coefficient of Variation)
    bdi_m["bdi_coef_var"] = bdi_m["bdi_std"] / bdi_m["bdi_mean"]
    
    # 3. Monthly Return / Momentum
    df_sorted = df.sort_values("Date")
    returns = df_sorted.groupby("year_month").apply(
        lambda x: (x["BDI_Close"].iloc[-1] - x["BDI_Close"].iloc[0]) / x["BDI_Close"].iloc[0]
        if len(x) > 1 else 0,
        include_groups=False
    ).reset_index(name="bdi_monthly_return")
    
    return pd.merge(bdi_m, returns, on="year_month")

def build_merged_dataset():
    print("Loading datasets from /data folder...")
    bdi_df = process_daily_bdi(os.path.join(DATA_DIR, "01_baltic_dry_index_daily.csv"))
    
    brent = pd.read_csv(os.path.join(DATA_DIR, "02_brent_crude_monthly.csv"), parse_dates=["Date"])
    pr = pd.read_csv(os.path.join(DATA_DIR, "03_botswana_policy_rate.csv"), parse_dates=["Date"])
    fao = pd.read_csv(os.path.join(DATA_DIR, "04_fao_botswana_prices.csv"), parse_dates=["Date"])
    hcp = pd.read_csv(os.path.join(DATA_DIR, "05_human_capital_project.csv"), parse_dates=["Date"])

    # Prepare Month Keys
    for df in [brent, pr, fao, hcp]:
        df["year_month"] = df["Date"].dt.to_period("M").astype(str)

    brent_m = brent[["year_month", "Brent_USD_per_barrel"]]
    pr_m = pr[["year_month", "policy_rate"]]

    # Pivot FAO (Target Code 23014 is Food Price Inflation % YoY)
    fao["col"] = "FAO_" + fao["Item Code"].astype(str)
    fao_wide = fao.pivot_table(index="year_month", columns="col", values="Value", aggfunc="first").reset_index()

    # Pivot HCP (Cross-country features, e.g., ZAF)
    hcp["col"] = hcp["REF_AREA"] + "_" + hcp["INDICATOR"]
    hcp_wide = hcp.pivot_table(index="year_month", columns="col", values="Value", aggfunc="first").reset_index()

    # Master Outer Join
    merged = bdi_df.copy()
    for df in [brent_m, pr_m, fao_wide, hcp_wide]:
        merged = merged.merge(df, on="year_month", how="outer")

    merged = merged.sort_values("year_month").reset_index(drop=True)

    # 12-Month Lags for features to avoid future leakage
    lag_cols = ["bdi_mean", "bdi_coef_var", "bdi_monthly_return", "Brent_USD_per_barrel", "policy_rate"]
    for col in lag_cols:
        if col in merged.columns:
            merged[f"{col}_lag12"] = merged[col].shift(12)

    return merged

if __name__ == "__main__":
    df = build_merged_dataset()
    print("\n✅ Master Dataset Processed Successfully!")
    print(f"Dataset Shape: {df.shape}")
    print("\nLast 5 rows of merged data:")
    print(df[["year_month", "FAO_23014", "bdi_mean_lag12", "Brent_USD_per_barrel_lag12"]].tail(5))