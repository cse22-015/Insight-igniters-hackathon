import os
# Prevent OpenBLAS and OMP thread allocation crashes on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import build_merged_dataset

# Force PyTorch to single thread for stability
torch.set_num_threads(1)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. PyTorch LSTM Architecture ---
class InflationLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super(InflationLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 2. Sequence Windowing Helper ---
def create_sequences(X, y, seq_length=6):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

# --- 3. Feature Preparation ---
def prepare_data_for_lstm(df, target_col="FAO_23014", seq_length=6):
    df = df.dropna(subset=[target_col]).copy()
    
    leakage_cols = [c for c in df.columns if "_FAO_CP_" in c or c.startswith("FAO_")]
    exclude = ["year_month", "Date", target_col] + leakage_cols
    feature_cols = [c for c in df.columns if c not in exclude]
    
    for col in leakage_cols:
        if col != target_col:
            df[f"{col}_lag1"] = df[col].shift(1)
            feature_cols.append(f"{col}_lag1")
            
    df["target_lag1"] = df[target_col].shift(1)
    feature_cols.append("target_lag1")
    
    # Explicitly retain year_month alongside feature_cols and target_col
    clean_df = df[["year_month"] + feature_cols + [target_col]].dropna().reset_index(drop=True)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(clean_df[feature_cols])
    y_scaled = scaler_y.fit_transform(clean_df[[target_col]]).flatten()
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length=seq_length)
    dates = clean_df["year_month"].iloc[seq_length:].values
    
    return X_seq, y_seq, dates, scaler_y

# --- 4. Training Loop with Early Stopping ---
def train_and_evaluate_lstm():
    df = build_merged_dataset()
    seq_length = 6
    X_seq, y_seq, dates, scaler_y = prepare_data_for_lstm(df, seq_length=seq_length)
    
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
    val_dates = dates[split_idx:]
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    input_dim = X_seq.shape[2]
    model = InflationLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    patience = 20
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = None
    
    print("\n--- Training Model 2: PyTorch LSTM ---")
    for epoch in range(1, 151):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_X_tensor = torch.tensor(X_val, dtype=torch.float32)
            val_preds_scaled = model(val_X_tensor).squeeze().numpy()
            val_loss = mean_squared_error(y_val, val_preds_scaled)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Val Loss: {val_loss:.6f}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch}!")
            break
            
    model.load_state_dict(best_model_weights)
    model.eval()
    
    with torch.no_grad():
        final_val_preds_scaled = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()
        
    preds_orig = scaler_y.inverse_transform(final_val_preds_scaled.reshape(-1, 1)).flatten()
    actuals_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(actuals_orig, preds_orig))
    mae = mean_absolute_error(actuals_orig, preds_orig)
    r2 = r2_score(actuals_orig, preds_orig)
    
    print("\n--- Model 2 (LSTM) Performance Summary ---")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation R²:   {r2:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(val_dates, actuals_orig, label="Actual Food Inflation", color="black", linewidth=2)
    plt.plot(val_dates, preds_orig, label="LSTM Forecast", color="blue", linestyle="--")
    plt.xticks(rotation=45)
    plt.title("Model 2: PyTorch LSTM Forecast vs Actual")
    plt.xlabel("Year-Month")
    plt.ylabel("Inflation Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model2_lstm_forecast.png")
    print(" Saved forecast chart to 'model2_lstm_forecast.png'")

if __name__ == "__main__":
    train_and_evaluate_lstm()