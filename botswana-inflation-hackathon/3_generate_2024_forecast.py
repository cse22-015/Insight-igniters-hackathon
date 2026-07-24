import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb

def load_and_preprocess_data():
    print("Loading raw datasets...")
    
    # 1. Baltic Dry Index (Daily)
    bdi_files = glob.glob("01_baltic_dry_index_daily.csv") or glob.glob("data/01_baltic_dry_index_daily.csv")
    bdi = pd.read_csv(bdi_files[0])
    bdi['Date'] = pd.to_datetime(bdi['Date'])
    bdi['year_month'] = bdi['Date'].dt.to_period('M').dt.to_timestamp()
    
    # Extract rich daily features (Volatility, Extremes, Trend)
    bdi_monthly = bdi.groupby('year_month').agg(
        bdi_mean=('BDI_Close', 'mean'),
        bdi_std=('BDI_Close', 'std'),
        bdi_max=('BDI_Close', 'max'),
        bdi_min=('BDI_Close', 'min'),
        bdi_extreme_days=('BDI_Close', lambda x: (x.pct_change().abs() > 0.03).sum())
    ).reset_index()
    
    # 2. Brent Crude (Monthly)
    brent_files = glob.glob("02_brent_crude_monthly.csv") or glob.glob("data/02_brent_crude_monthly.csv")
    brent = pd.read_csv(brent_files[0])
    brent['year_month'] = pd.to_datetime(brent['Date']).dt.to_period('M').dt.to_timestamp()
    
    # 3. Policy Rate (Monthly)
    rate_files = glob.glob("03_botswana_policy_rate.csv") or glob.glob("data/03_botswana_policy_rate.csv")
    rate = pd.read_csv(rate_files[0])
    rate['year_month'] = pd.to_datetime(rate['Date']).dt.to_period('M').dt.to_timestamp()
    
    # 4. Target Dataset: FAO Prices (Monthly)
    fao_files = glob.glob("04_fao_botswana_prices.csv") or glob.glob("data/04_fao_botswana_prices.csv")
    fao = pd.read_csv(fao_files[0])
    fao['year_month'] = pd.to_datetime(fao['Date']).dt.to_period('M').dt.to_timestamp()
    
    # Filter target item 23014 (Food Inflation % YoY)
    target_df = fao[fao['Item Code'] == 23014][['year_month', 'Value']].rename(columns={'Value': 'target_food_inflation'})
    
    # Merge all datasets on year_month
    df = target_df.merge(bdi_monthly, on='year_month', how='left')
    df = df.merge(brent[['year_month', 'Brent_USD_per_barrel']], on='year_month', how='left')
    df = df.merge(rate[['year_month', 'policy_rate']], on='year_month', how='left')
    
    df = df.sort_values('year_month').reset_index(drop=True)
    
    # Engineer Lags (Strategy A: Avoid phantom 2024 features)
    df['target_lag1'] = df['target_food_inflation'].shift(1)
    df['target_lag2'] = df['target_food_inflation'].shift(2)
    df['brent_lag1'] = df['Brent_USD_per_barrel'].shift(1)
    df['bdi_lag1'] = df['bdi_mean'].shift(1)
    df['policy_lag1'] = df['policy_rate'].shift(1)
    
    # Drop initial NaN rows created by shifts
    df = df.dropna().reset_index(drop=True)
    return df

def generate_2024_predictions():
    print("--- Generating Deliverable 1.1a: Best-Model 2024 Predictions ---")
    df = load_and_preprocess_data()
    
    # Define features and target
    feature_cols = ['target_lag1', 'target_lag2', 'brent_lag1', 'bdi_lag1', 'policy_lag1', 
                    'bdi_std', 'bdi_extreme_days']
    target_col = 'target_food_inflation'
    
    # Historical training set (ending Dec 2023)
    train_df = df[df['year_month'] <= '2023-12-01'].copy()
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Fit final XGBoost Model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Autoregressive 12-month prediction loop for 2024
    forecast_dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')
    predictions = []
    
    # Seed values using last row of 2023
    current_features = X_train.iloc[-1:].copy()
    
    for date in forecast_dates:
        pred_val = model.predict(current_features)[0]
        predictions.append(pred_val)
        
        # Shift lag window forward
        current_features = current_features.copy()
        current_features['target_lag2'] = current_features['target_lag1']
        current_features['target_lag1'] = pred_val

    # Create deliverable CSV
    output_df = pd.DataFrame({
        'year_month': forecast_dates.strftime('%Y-%m'),
        'forecast': np.round(predictions, 4)
    })
    
    output_csv = "best_model_predictions.csv"
    output_df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully created '{output_csv}'!")
    print("\nDeliverable 1.1a Output Preview:")
    print(output_df)

if __name__ == "__main__":
    generate_2024_predictions()