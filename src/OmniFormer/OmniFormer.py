# DEVELOPED BY NGUYEN QUOC ANH
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "./OmniFormer/dataset"

file_paths = {
    'ADAUSDT':  f'{DATA_ROOT}/ADAUSDT_1d_full.csv',
    'AVAXUSDT': f'{DATA_ROOT}/AVAXUSDT_1d_full.csv',
    'BNBUSDT':  f'{DATA_ROOT}/BNBUSDT_1d_full.csv',
    'BTCUSDT':  f'{DATA_ROOT}/BTCUSDT_1d_full.csv',
    'DOGEUSDT': f'{DATA_ROOT}/DOGEUSDT_1d_full.csv',
    'ETHUSDT':  f'{DATA_ROOT}/ETHUSDT_1d_full.csv',
    'SOLUSDT':  f'{DATA_ROOT}/SOLUSDT_1d_full.csv',
    'LINKUSDT': f'{DATA_ROOT}/LINKUSDT_1d_full.csv',
    'TRXUSDT':  f'{DATA_ROOT}/TRXUSDT_1d_full.csv',
    'XRPUSDT':  f'{DATA_ROOT}/XRPUSDT_1d_full.csv',
}

COINS = list(file_paths.keys())

# Training hyperparameters
SEQ_LEN       = 90
BATCH_SIZE    = 256
EPOCHS        = 6000
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 700
MIN_DELTA     = 2e-9
TEST_DAYS     = 365
VAL_FRACTION  = 0.20


def load_and_prepare_multi_coin_data():
    """Load CSVs, rename columns with coin prefix, outer join, ffill/bfill"""
    data = {}
    for coin, path in file_paths.items():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        data[coin] = df

    df_all = None
    for coin in COINS:
        df = data[coin].rename(columns={
            'open':   f'{coin}_open',
            'high':   f'{coin}_high',
            'low':    f'{coin}_low',
            'close':  f'{coin}_close',
            'volume': f'{coin}_volume'
        })
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.join(df, how='outer')

    return df_all.ffill().bfill().sort_index()


class RevIN(nn.Module):
    """Reversible Instance Normalization for time-series"""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        shape = (1, 1, num_features)
        if affine:
            self.weight = nn.Parameter(torch.ones(shape))
            self.bias   = nn.Parameter(torch.zeros(shape))
        else:
            self.register_buffer('weight', torch.ones(shape))
            self.register_buffer('bias',   torch.zeros(shape))

    def norm(self, x):
        mean  = x.mean(dim=1, keepdim=True).detach()
        stdev = x.std(dim=1, keepdim=True).detach() + self.eps
        x_norm = (x - mean) / stdev
        return x_norm * self.weight + self.bias, mean, stdev

    def denorm(self, x, mean, stdev):
        x = (x - self.bias) / (self.weight + self.eps)
        return x * stdev + mean


class MultiAssetTimeSeriesTransformer(nn.Module):
    """
    Patch-based Transformer with RevIN for multi-asset one-step-ahead forecasting.
    """
    def __init__(self, num_features, seq_len=90, patch_len=12, stride=6,
                 d_model=384, nhead=6, num_layers=4, dropout=0.10):
        super().__init__()

        self.revin = RevIN(num_features, affine=True)
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.head = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (B, L, C)
        x_norm, mean, stdev = self.revin.norm(src)
        B, L, C = x_norm.shape

        # Patchify
        x = x_norm.permute(0, 2, 1).reshape(B * C, L)
        x = x.unfold(1, self.patch_len, self.stride)
        x = self.patch_embed(x) + self.pos_embed

        x = self.encoder(x)

        # Reshape back
        x = x.view(B, C, -1, x.size(-1))          # (B, C, P, d)
        x = x.permute(0, 2, 1, 3)                 # (B, P, C, d)
        x = x.reshape(B * x.size(1), C, -1)       # (B*P, C, d)

        x = self.channel_mixer(x)
        x = x.view(B, -1, C, x.size(-1))          # (B, P, C, d)
        x = x.mean(dim=1)                         # (B, C, d)

        pred = self.head(x).squeeze(-1)           # (B, C)

        pred = pred.unsqueeze(1)                  # (B, 1, C)
        pred = self.revin.denorm(pred, mean, stdev).squeeze(1)

        return pred


class OneStepAheadDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.from_numpy(x), torch.from_numpy(y)


# ─── Main ────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()

print("Loading & preparing data...")
df_all = load_and_prepare_multi_coin_data()
columns = df_all.columns.tolist()
n_features = len(columns)

# Split
test_data     = df_all.iloc[-TEST_DAYS:]
pre_test_data = df_all.iloc[:-TEST_DAYS]
val_size      = int(len(pre_test_data) * VAL_FRACTION)

val_data   = pre_test_data.iloc[-val_size:]
train_data = pre_test_data.iloc[:-val_size]

# Scaling (fit on train only)
scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

def scale_df(df):
    return pd.DataFrame(
        np.hstack([scalers[col].transform(df[[col]]) for col in columns]),
        index=df.index, columns=columns
    )

scaled_train = scale_df(train_data).values.astype(np.float32)
scaled_val   = scale_df(val_data).values.astype(np.float32)
scaled_test  = scale_df(test_data).values.astype(np.float32)
pre_test_scaled = np.concatenate((scaled_train, scaled_val))

# Datasets & loaders
train_dataset = OneStepAheadDataset(scaled_train, SEQ_LEN)
val_dataset   = OneStepAheadDataset(scaled_val,   SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Model
model = MultiAssetTimeSeriesTransformer(
    num_features=n_features,
    seq_len=SEQ_LEN,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_cosine   = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8, verbose=True)

# ─── Training ────────────────────────────────────────────────────────────────────

print(f"\nTraining ({EPOCHS} epochs max, patience={PATIENCE})...")

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx = bx.to(device).float()
        by = by.to(device).float()

        bx = bx + torch.randn_like(bx) * 0.0015   # light noise

        pred = model(bx)
        loss = loss_fn(pred, by)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device).float()
            by = by.to(device).float()
            pred = model(bx)
            val_loss += loss_fn(pred, by).item()
            n_batches += 1

    val_loss = val_loss / n_batches if n_batches > 0 else float('inf')

    scheduler_cosine.step()
    scheduler_plateau.step(val_loss)

    torch.cuda.empty_cache()
    gc.collect()

    if (epoch + 1) % 20 == 0:
        print(f"[{epoch+1:4d}] train: {train_loss/len(train_loader):.7f}  "
              f"val: {val_loss:.7f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'OmniFormer.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best
try:
    model.load_state_dict(torch.load('best_multiasset_transformer.pth'))
    print("Loaded best model.")
except Exception as e:
    print(f"Could not load best weights: {e}")

# ─── Evaluation ──────────────────────────────────────────────────────────────────

print("\nComputing one-step-ahead metrics...")

full_scaled = np.concatenate((pre_test_scaled, scaled_test))
test_dataset = OneStepAheadDataset(full_scaled, SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

model.eval()
preds, trues = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device).float()
        pred = model(bx)
        preds.append(pred.cpu().numpy())
        trues.append(by.numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

start_idx = len(pre_test_scaled) - SEQ_LEN
preds_test = preds[start_idx : start_idx + TEST_DAYS]
trues_test = trues[start_idx : start_idx + TEST_DAYS]

err_scaled = preds_test - trues_test

global_mse  = np.mean(err_scaled ** 2)
global_rmse = math.sqrt(global_mse)
global_mae  = np.mean(np.abs(err_scaled))

one_step_global_results = [{
    'model': 'Transformer',
    'test_mse_global':  global_mse,
    'test_rmse_global': global_rmse,
    'test_mae_global':  global_mae,
}]

one_step_per_coin_results = []
for coin in COINS:
    coin_cols = [f"{coin}_{v}" for v in ['open','high','low','close','volume']]
    idxs = [columns.index(c) for c in coin_cols]

    err_coin = preds_test[:, idxs] - trues_test[:, idxs]

    mse  = np.mean(err_coin ** 2)
    rmse = math.sqrt(mse)
    mae  = np.mean(np.abs(err_coin))

    one_step_per_coin_results.append({
        'model': 'Transformer',
        'coin': coin,
        'mse_scaled': mse,
        'rmse_scaled': rmse,
        'mae_scaled': mae,
    })

print(f"Global RMSE (scaled): {global_rmse:.6f} | MAE: {global_mae:.6f}")

# Overfit check
train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

train_preds, train_trues = [], []
with torch.no_grad():
    for bx, by in train_loader_eval:
        bx = bx.to(device).float()
        pred = model(bx)
        train_preds.append(pred.cpu().numpy())
        train_trues.append(by.numpy())

train_preds = np.concatenate(train_preds)
train_trues = np.concatenate(train_trues)

train_mse = np.mean((train_preds - train_trues) ** 2)

overfit_results = [{
    'model': 'Transformer',
    'train_mse_scaled': train_mse,
    'test_mse_scaled':  global_mse,
    'overfit_gap_scaled': global_mse - train_mse
}]

print(f"Train MSE: {train_mse:.6f} | Test MSE: {global_mse:.6f} | Gap: {global_mse - train_mse:.6f}")

# ─── Inverse scale & save forecasts ──────────────────────────────────────────────

pred_prices = np.zeros_like(preds_test)
for i, col in enumerate(columns):
    pred_prices[:, i] = scalers[col].inverse_transform(
        preds_test[:, i].reshape(-1, 1)
    ).ravel()

pred_df = pd.DataFrame(pred_prices, index=test_data.index, columns=columns)

df_export = test_data.add_suffix('_true').copy()
for col in columns:
    df_export[f'{col}_pred'] = pred_df[col]

df_export.to_csv('forecasts_transformer.csv')

pd.DataFrame(one_step_global_results).to_csv('one_step_OmniFormer_global.csv', index=False)
pd.DataFrame(one_step_per_coin_results).to_csv('one_step_OmniFormer_per_coin.csv', index=False)
pd.DataFrame(overfit_results).to_csv('OmniFormer_overfit_check.csv', index=False)