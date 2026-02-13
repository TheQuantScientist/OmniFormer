import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import os

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
TEST_DAYS = 365
VAL_FRACTION = 0.20
BATCH_SIZE = 256
SEQ_LENGTHS = [60, 90, 180, 256, 365]
D_MODELS = [64, 128, 256]
NHEADS = [4, 8, 16]
NUM_LAYERS_LIST = [2, 3, 4]
EPOCH_OPTIONS = [60, 120, 200]
LR = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FORECAST_DAYS = 365  # Consistent with test set size

# ────────────────────────────────────────────────
#  Dataset Class
# ────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ────────────────────────────────────────────────
#  Transformer Model
# ────────────────────────────────────────────────
class TransformerTS(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # src: (batch, seq_len, features)
        src = self.embedding(src)
        out = self.transformer(src)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

# ────────────────────────────────────────────────
#  Data Loading & Preprocessing
# ────────────────────────────────────────────────
file_paths = {
    'ADAUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/ADAUSDT_1d_full.csv',
    'AVAXUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/AVAXUSDT_1d_full.csv',
    'BNBUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/BNBUSDT_1d_full.csv',
    'BTCUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/BTCUSDT_1d_full.csv',
    'DOGEUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/DOGEUSDT_1d_full.csv',
    'ETHUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/ETHUSDT_1d_full.csv',
    'SOLUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/SOLUSDT_1d_full.csv',
    'LINKUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/LINKUSDT_1d_full.csv',
    'TRXUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/TRXUSDT_1d_full.csv',
    'XRPUSDT': '/home/nckh2/qa/finance/binance_ohlcv_daily/XRPUSDT_1d_full.csv'
}

coins = list(file_paths.keys())

# Load and combine data
data = {}
for coin, path in file_paths.items():
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    data[coin] = df

# Merge with outer join
df_all = None
for coin in coins:
    df = data[coin]
    df = df.rename(columns={
        'open': f'{coin}_open',
        'high': f'{coin}_high',
        'low': f'{coin}_low',
        'close': f'{coin}_close',
        'volume': f'{coin}_volume'
    })
    if df_all is None:
        df_all = df
    else:
        df_all = df_all.join(df, how='outer')

df_all = df_all.ffill().bfill()

# ────────────────────────────────────────────────
#  Time-based split
# ────────────────────────────────────────────────
n_total = len(df_all)
n_test = TEST_DAYS
n_pre = n_total - n_test
n_val = int(n_pre * VAL_FRACTION)
n_train = n_pre - n_val

print(f"Total days: {n_total}")
print(f"Train:      {n_train} days")
print(f"Val:        {n_val} days")
print(f"Test:       {n_test} days\n")

train_data = df_all.iloc[:n_train]
val_data   = df_all.iloc[n_train:n_train + n_val]
test_data  = df_all.iloc[n_train + n_val:]

columns = df_all.columns.tolist()

# ────────────────────────────────────────────────
#  Scaling (fit only on train)
# ────────────────────────────────────────────────
scalers = {}
for col in columns:
    scalers[col] = MinMaxScaler()
    scalers[col].fit(train_data[[col]])

def scale_df(df, scalers):
    scaled = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        scaled[col] = scalers[col].transform(df[[col]]).squeeze()
    return scaled

scaled_train = scale_df(train_data, scalers)
scaled_val   = scale_df(val_data, scalers)
scaled_test  = scale_df(test_data, scalers)

train_values = scaled_train.values.astype(np.float32)
val_values   = scaled_val.values.astype(np.float32)
test_values  = scaled_test.values.astype(np.float32)

train_val_values = np.concatenate((train_values, val_values), axis=0)

# ────────────────────────────────────────────────
#  Hyperparameter Tuning
# ────────────────────────────────────────────────
results = []

for seq_length in SEQ_LENGTHS:
    # Train dataset
    X_train, y_train = [], []
    for i in range(len(train_values) - seq_length):
        X_train.append(train_values[i:i + seq_length])
        y_train.append(train_values[i + seq_length])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Validation dataset (using train_val_values)
    X_val, y_val = [], []
    for j in range(len(val_values)):
        start = len(train_values) + j - seq_length
        if start >= 0:
            seq = train_val_values[start:start + seq_length]
        else:
            pad_len = -start
            pad = np.tile(train_values[0], (pad_len, 1))
            seq = np.concatenate((pad, train_val_values[0:start + seq_length]), axis=0)
        X_val.append(seq)
        y_val.append(val_values[j])

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    for d_model in D_MODELS:
        for nhead in NHEADS:
            if d_model % nhead != 0:
                continue
            for num_layers in NUM_LAYERS_LIST:
                for epochs in EPOCH_OPTIONS:
                    model = TransformerTS(
                        input_dim=len(columns),
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers
                    ).to(DEVICE)

                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    loss_fn = nn.MSELoss()

                    model.train()
                    for epoch in range(epochs):
                        total_loss = 0
                        for batch_x, batch_y in train_loader:
                            batch_x = batch_x.to(DEVICE)
                            batch_y = batch_y.to(DEVICE)
                            out = model(batch_x)
                            loss = loss_fn(out, batch_y)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        if (epoch + 1) % 20 == 0:
                            print(f"Seq={seq_length} | d={d_model} | h={nhead} | l={num_layers} | "
                                  f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        # Start from last sequence of TRAIN
                        current_seq = torch.from_numpy(
                            train_values[-seq_length:]
                        ).unsqueeze(0).to(DEVICE)

                        preds = []

                        for _ in range(len(val_values)):  # full validation horizon
                            next_step = model(current_seq)
                            preds.append(next_step.cpu().numpy().squeeze())
                            current_seq = torch.cat(
                                (current_seq[:, 1:, :], next_step.unsqueeze(1)),
                                dim=1
                            )

                        preds = np.array(preds)
                        targets = val_values[:len(preds)]

                        mse = np.mean((preds - targets) ** 2)
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(preds - targets))

                    results.append({
                        'seq_length': seq_length,
                        'd_model': d_model,
                        'nhead': nhead,
                        'num_layers': num_layers,
                        'epochs': epochs,
                        'val_mse': mse,
                        'val_rmse': rmse,
                        'val_mae': mae
                    })

                    print(f"Seq {seq_length} | d_model {d_model} | nhead {nhead} | layers {num_layers} | "
                          f"Epochs {epochs} | Val MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")

# Save tuning results
pd.DataFrame(results).to_csv('tuning_results.csv', index=False)
print("\nTuning results saved to 'tuning_results.csv'")

# ────────────────────────────────────────────────
#  Select best model & retrain on train+val
# ────────────────────────────────────────────────
df_results = pd.DataFrame(results)
best_row = df_results.loc[df_results['val_rmse'].idxmin()]
print("\nBest config:")
print(best_row)

best_seq = int(best_row['seq_length'])
best_d_model = int(best_row['d_model'])
best_nhead = int(best_row['nhead'])
best_num_layers = int(best_row['num_layers'])
best_epochs = int(best_row['epochs'])

# Retrain on train+val
X_pre, y_pre = [], []
for i in range(len(train_val_values) - best_seq):
    X_pre.append(train_val_values[i:i + best_seq])
    y_pre.append(train_val_values[i + best_seq])
X_pre = np.array(X_pre)
y_pre = np.array(y_pre)

pre_dataset = TimeSeriesDataset(X_pre, y_pre)
pre_loader = DataLoader(pre_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TransformerTS(
    input_dim=len(columns),
    d_model=best_d_model,
    nhead=best_nhead,
    num_layers=best_num_layers
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"\nRetraining best model on train+val ({best_epochs} epochs)...")
for epoch in range(best_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in pre_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{best_epochs} | Loss: {total_loss/len(pre_loader):.6f}")

# ────────────────────────────────────────────────
#  Test Evaluation
# ────────────────────────────────────────────────
full_for_test = np.concatenate((train_val_values, test_values), axis=0)
offset = len(train_val_values)

X_test, y_test = [], []
for j in range(len(test_values)):
    start = offset + j - best_seq
    if start >= 0:
        X_test.append(full_for_test[start:start + best_seq])
    else:
        pad_len = -start
        pad = np.tile(train_val_values[0], (pad_len, 1))
        seq = np.concatenate((pad, full_for_test[0:start + best_seq]), axis=0)
        X_test.append(seq)
    y_test.append(test_values[j])

X_test = np.array(X_test)
y_test = np.array(y_test)

model.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test).to(DEVICE)
    pred_test = model(X_test_t)
    y_test_t = torch.from_numpy(y_test).to(DEVICE)
    test_mse = loss_fn(pred_test, y_test_t).item()
    test_rmse = math.sqrt(test_mse)
    test_mae = torch.mean(torch.abs(pred_test - y_test_t)).item()

test_results = [{
    'seq_length': best_seq,
    'd_model': best_d_model,
    'nhead': best_nhead,
    'num_layers': best_num_layers,
    'epochs': best_epochs,
    'test_mse': test_mse,
    'test_rmse': test_rmse,
    'test_mae': test_mae
}]
pd.DataFrame(test_results).to_csv('test_results.csv', index=False)
print(f"\nTest results saved to 'test_results.csv'")
print(f"Test MSE: {test_mse:.6f} | RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f}")

# ────────────────────────────────────────────────
#  Final model on ALL data + Forecast
# ────────────────────────────────────────────────
scalers_final = {}
for col in columns:
    scalers_final[col] = MinMaxScaler()
    scalers_final[col].fit(df_all[[col]])

scaled_all = pd.DataFrame(index=df_all.index, columns=columns)
for col in columns:
    scaled_all[col] = scalers_final[col].transform(df_all[[col]]).squeeze()

values_all = scaled_all.values.astype(np.float32)

X_all, y_all = [], []
for i in range(len(values_all) - best_seq):
    X_all.append(values_all[i:i + best_seq])
    y_all.append(values_all[i + best_seq])
X_all = np.array(X_all)
y_all = np.array(y_all)

all_dataset = TimeSeriesDataset(X_all, y_all)
all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True)

model_final = TransformerTS(
    input_dim=len(columns),
    d_model=best_d_model,
    nhead=best_nhead,
    num_layers=best_num_layers
).to(DEVICE)

optimizer = torch.optim.Adam(model_final.parameters(), lr=LR)

print(f"\nTraining final model on ALL data ({best_epochs} epochs)...")
for epoch in range(best_epochs):
    model_final.train()
    total_loss = 0
    for batch_x, batch_y in all_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        out = model_final(batch_x)
        loss = loss_fn(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{best_epochs} | Loss: {total_loss/len(all_loader):.6f}")

# Autoregressive forecast
model_final.eval()
forecast = []
current_seq = torch.from_numpy(values_all[-best_seq:]).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    for _ in range(FORECAST_DAYS):
        pred = model_final(current_seq)
        forecast.append(pred.cpu().numpy().squeeze())
        new_input = pred.unsqueeze(1)
        current_seq = torch.cat((current_seq[:, 1:, :], new_input), dim=1)

forecast = np.array(forecast)

# Inverse transform only close prices
forecast_inv = np.zeros((FORECAST_DAYS, len(coins)))
for i, coin in enumerate(coins):
    col = f'{coin}_close'
    idx = columns.index(col)
    forecast_inv[:, i] = scalers_final[col].inverse_transform(
        forecast[:, idx].reshape(-1, 1)
    ).squeeze()

# Create forecast DataFrame
last_date = df_all.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')
df_forecast = pd.DataFrame(forecast_inv, index=future_dates, columns=coins)

df_forecast.to_csv('crypto_forecast_365_days.csv')
print("\nForecast (365 days) saved to 'crypto_forecast_365_days.csv'")