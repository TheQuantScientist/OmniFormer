# DEVELOPED BY NGUYEN QUOC ANH — Ablation-ready version
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

# Training hyperparameters (you can override per experiment)
SEQ_LEN       = 90
BATCH_SIZE    = 256
EPOCHS        = 6000
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 700
MIN_DELTA     = 2e-9
TEST_DAYS     = 365
VAL_FRACTION  = 0.20

# ─── Data Preparation ────────────────────────────────────────────────────────────

def load_and_prepare_multi_coin_data():
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
    def __init__(self,
                 num_features,
                 seq_len=90,
                 patch_len=12,
                 stride=6,
                 d_model=384,
                 nhead=6,
                 num_layers=4,
                 dropout=0.10,
                 use_revin=True,
                 use_pos_embed=True,
                 use_channel_mixer=True,
                 use_patching=True):
        super().__init__()

        self.use_revin = use_revin
        self.use_patching = use_patching
        self.use_pos_embed = use_pos_embed
        self.use_channel_mixer = use_channel_mixer

        self.revin = RevIN(num_features, affine=True) if use_revin else None

        self.seq_len    = seq_len
        self.patch_len  = patch_len
        self.stride     = stride

        if use_patching:
            num_patches = (seq_len - patch_len) // stride + 1
            self.patch_embed = nn.Linear(patch_len, d_model)
            if use_pos_embed:
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
            else:
                self.pos_embed = None
        else:
            # No patching → treat whole sequence as one "token" per channel
            self.patch_embed = nn.Linear(seq_len, d_model)
            if use_pos_embed:
                self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            else:
                self.pos_embed = None
            num_patches = 1

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

        if use_channel_mixer:
            self.channel_mixer = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
        else:
            self.channel_mixer = nn.Identity()

        self.head = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (B, L, C)
        B, L, C = src.shape

        if self.use_revin:
            x, mean, stdev = self.revin.norm(src)
        else:
            x = src
            mean = stdev = None

        if self.use_patching:
            # Patchify → (B*C, num_patches, patch_len) → embed → (B*C, num_patches, d)
            x = x.permute(0, 2, 1).reshape(B * C, L)
            x = x.unfold(1, self.patch_len, self.stride)
            x = self.patch_embed(x)                     # (B*C, P, d)
        else:
            # No patching: flatten time → (B*C, L) → (B*C, 1, d)
            x = x.permute(0, 2, 1).reshape(B * C, L)
            x = self.patch_embed(x.unsqueeze(1))        # (B*C, 1, d)

        if self.use_pos_embed:
            x = x + self.pos_embed

        x = self.encoder(x)                             # (B*C, P or 1, d)

        # Reshape back to (B, P or 1, C, d)
        P = x.size(1)
        x = x.view(B, C, P, -1).permute(0, 2, 1, 3)     # (B, P, C, d)
        x = x.reshape(B * P, C, -1)                     # (B*P, C, d)

        x = self.channel_mixer(x)
        x = x.view(B, P, C, -1).mean(dim=1)             # (B, C, d)

        pred = self.head(x).squeeze(-1)                 # (B, C)

        if self.use_revin:
            pred = pred.unsqueeze(1)                    # (B, 1, C)
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


# ─── Training & Evaluation Helpers ──────────────────────────────────────────────

def create_dataloaders(scaled_train, scaled_val, seq_len, batch_size):
    train_ds = OneStepAheadDataset(scaled_train, seq_len)
    val_ds   = OneStepAheadDataset(scaled_val,   seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs, device, save_path="model_ablation.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cosine   = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8, verbose=True)

    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device).float()
            by = by.to(device).float()
            bx = bx + torch.randn_like(bx) * 0.0015

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

        val_loss /= n_batches if n_batches > 0 else float('inf')

        scheduler_cosine.step()
        scheduler_plateau.step(val_loss)

        torch.cuda.empty_cache()
        gc.collect()

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1:4d}] train: {train_loss/len(train_loader):.7f}  val: {val_loss:.7f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path))
    print(f"→ Loaded best weights from {save_path}")
    return model


def evaluate_one_step(model, full_scaled, seq_len, batch_size, device, scalers, columns, test_data, test_days):
    dataset = OneStepAheadDataset(full_scaled, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device).float()
            pred = model(bx)
            preds.append(pred.cpu().numpy())
            trues.append(by.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    start_idx = len(full_scaled) - seq_len - test_days
    if start_idx < 0:
        start_idx = 0
    preds_test = preds[start_idx : start_idx + test_days]
    trues_test = trues[start_idx : start_idx + test_days]

    err = preds_test - trues_test
    global_mse  = np.mean(err ** 2)
    global_rmse = math.sqrt(global_mse)
    global_mae  = np.mean(np.abs(err))

    results_global = {
        'test_mse_global':  global_mse,
        'test_rmse_global': global_rmse,
        'test_mae_global':  global_mae,
    }

    results_per_coin = []
    for coin in COINS:
        coin_cols = [f"{coin}_{v}" for v in ['open','high','low','close','volume']]
        idxs = [columns.index(c) for c in coin_cols]
        err_coin = preds_test[:, idxs] - trues_test[:, idxs]
        mse  = np.mean(err_coin ** 2)
        rmse = math.sqrt(mse)
        mae  = np.mean(np.abs(err_coin))
        results_per_coin.append({
            'coin': coin,
            'mse_scaled': mse,
            'rmse_scaled': rmse,
            'mae_scaled': mae,
        })

    return results_global, results_per_coin, preds_test, trues_test


# ─── Main — Ablation runner ─────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading & preparing data...")
df_all = load_and_prepare_multi_coin_data()
columns = df_all.columns.tolist()
n_features = len(columns)

# Split & scale (done once — shared across ablations)
test_data     = df_all.iloc[-TEST_DAYS:]
pre_test_data = df_all.iloc[:-TEST_DAYS]
val_size      = int(len(pre_test_data) * VAL_FRACTION)

val_data   = pre_test_data.iloc[-val_size:]
train_data = pre_test_data.iloc[:-val_size]

scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

def scale_df(df):
    return pd.DataFrame(
        np.hstack([scalers[col].transform(df[[col]]) for col in columns]),
        index=df.index, columns=columns
    )

scaled_train = scale_df(train_data).values.astype(np.float32)
scaled_val   = scale_df(val_data).values.astype(np.float32)
scaled_test  = scale_df(test_data).values.astype(np.float32)
full_scaled  = np.concatenate((scaled_train, scaled_val, scaled_test))

# ── Choose ablation experiments ─────────────────────────────────────────────────

experiments = [
    # Baseline (your original setting)
    {"name": "Full_model", "use_revin":True, "use_patching":True, "use_pos_embed":True, "use_channel_mixer":True},

    # Ablations
    {"name": "No_RevIN",       "use_revin":False, "use_patching":True,  "use_pos_embed":True,  "use_channel_mixer":True},
    {"name": "No_Patching",    "use_revin":True,  "use_patching":False, "use_pos_embed":True,  "use_channel_mixer":True},
    {"name": "No_PosEmbed",    "use_revin":True,  "use_patching":True,  "use_pos_embed":False, "use_channel_mixer":True},
    {"name": "No_ChannelMixer","use_revin":True,  "use_patching":True,  "use_pos_embed":True,  "use_channel_mixer":False},
    # You can add more combinations...
]

all_global_results = []
all_per_coin_results = []

for exp in experiments:
    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT: {exp['name']}")
    print("="*70)

    model = MultiAssetTimeSeriesTransformer(
        num_features=n_features,
        seq_len=SEQ_LEN,
        use_revin=exp["use_revin"],
        use_patching=exp["use_patching"],
        use_pos_embed=exp["use_pos_embed"],
        use_channel_mixer=exp["use_channel_mixer"],
    ).to(device)

    train_loader, val_loader = create_dataloaders(scaled_train, scaled_val, SEQ_LEN, BATCH_SIZE)

    save_path = f"model_{exp['name']}.pth"
    model = train_model(model, train_loader, val_loader, EPOCHS, device, save_path=save_path)

    global_res, per_coin_res, _, _ = evaluate_one_step(
        model, full_scaled, SEQ_LEN, BATCH_SIZE, device,
        scalers, columns, test_data, TEST_DAYS
    )

    global_row = {'model': exp['name'], **global_res}
    all_global_results.append(global_row)

    for r in per_coin_res:
        all_per_coin_results.append({'model': exp['name'], **r})

    print(f"→ Global RMSE (scaled): {global_res['test_rmse_global']:.6f}")

# ── Save results ────────────────────────────────────────────────────────────────

pd.DataFrame(all_global_results).to_csv("ablation_global_results.csv", index=False)
pd.DataFrame(all_per_coin_results).to_csv("ablation_per_coin_results.csv", index=False)

print("\nAblation study finished. Results saved to:")
print("• ablation_global_results.csv")
print("• ablation_per_coin_results.csv")