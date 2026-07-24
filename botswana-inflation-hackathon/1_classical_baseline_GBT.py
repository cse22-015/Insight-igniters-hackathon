import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import build_merged_dataset

def check_stationarity(series, name="Target"):
    """Performs Augmented Dickey-Fuller (ADF) test for stationarity."""
    result = adfuller(series.dropna())
    print(f"\n--- Stationarity Test (ADF) for {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")
    if result[1] <= 0.05:
        print("=> Stationarity Status: Stationary (p <= 0.05)")
    else:
        print("=> Stationarity Status: Non-Stationary (p > 0.05)")

def engineer_features(df, target_col="FAO_23014"):
    df = df.dropna(subset=[target_col]).copy()
    leakage_cols = [c for c in df.columns if "_FAO_CP_" in c or c.startswith("FAO_")]
    exclude = ["year_month", "Date", target_col] + leakage_cols
    
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Lagged regional features
    for col in leakage_cols:
        if col != target_col:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)
            feature_cols.extend([f"{col}_lag1", f"{col}_lag3"])
            
    # Target Lags
    df["target_lag1"] = df[target_col].shift(1)
    df["target_lag2"] = df[target_col].shift(2)
    feature_cols.extend(["target_lag1", "target_lag2"])
    
    X = df[feature_cols].ffill().bfill()
    y = df[target_col]
    
    return X, y, df["year_month"]

def run_classical_baseline():
    df = build_merged_dataset()
    X, y, dates = engineer_features(df)
    
    # Check stationarity on target
    check_stationarity(y, name="FAO_23014 (Food Inflation)")
    
    tscv = TimeSeriesSplit(n_splits=5)
    residuals = []
    
    print("\n--- Running Model 1: XGBoost Classical Baseline ---")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.02, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        fold_res = y_test.values - preds
        residuals.extend(fold_res)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"Fold {fold} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        
    # Residual Diagnostics
    residuals = np.array(residuals)
    print("\n--- Residual Diagnostics ---")
    print(f"Mean Residual (Bias): {np.mean(residuals):.4f}")
    print(f"Residual Std Dev:     {np.std(residuals):.4f}")
    
    # Plot Residual Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.title("Model 1 Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.savefig("model1_residuals.png")
    print("Saved residual diagnostic plot to 'model1_residuals.png'")

if __name__ == "__main__":
    run_classical_baseline()
