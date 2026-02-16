import os
import sys

sys.path.append('.')

import argparse
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Add Time-Series-Library to path
from utils.timefeatures import time_features

# Import PatchTST model
from models.PatchTST import Model as PatchTST

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "./OmniFormer/dataset"

COIN_PATHS = {
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

COINS = list(COIN_PATHS.keys())

# Training settings ─ aligned with your custom script
SEQ_LEN      = 90
LABEL_LEN    = 45
PRED_LEN     = 1
EPOCHS       = 6000
BATCH_SIZE   = 256
LR           = 8e-5
WEIGHT_DECAY = 5e-6
PATIENCE     = 700
MIN_DELTA    = 2e-9
TEST_DAYS    = 365
VAL_FRACTION = 0.20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.MSELoss()


def load_and_combine_coin_data():
    """Load all coin CSVs, prefix columns, outer join, ffill/bfill"""
    data = {}
    for coin, path in COIN_PATHS.items():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        data[coin] = df

    df_all = None
    for coin in COINS:
        renamed = data[coin].rename(columns={
            'open':   f'{coin}_open',
            'high':   f'{coin}_high',
            'low':    f'{coin}_low',
            'close':  f'{coin}_close',
            'volume': f'{coin}_volume'
        })
        df_all = renamed if df_all is None else df_all.join(renamed, how='outer')

    return df_all.ffill().bfill().sort_index()


class TimeSeriesForecastDataset(Dataset):
    """Dataset compatible with PatchTST / Autoformer family"""
    def __init__(self, data: np.ndarray, time_features: np.ndarray,
                 seq_len: int, label_len: int, pred_len: int = 1):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len

    def __len__(self):
        return len(self.data) - self.total_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.time_features[s_begin:s_end]
        seq_y_mark = self.time_features[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float()
        )


def get_patchtst_config(n_features: int):
    common = {
        'seq_len': SEQ_LEN,
        'label_len': LABEL_LEN,
        'pred_len': PRED_LEN,
        'enc_in': n_features,
        'dec_in': n_features,
        'c_out': n_features,
        'd_model': 384,
        'n_heads': 6,
        'e_layers': 4,
        'd_layers': 1,
        'd_ff': 256,
        'dropout': 0.1,
        'activation': 'gelu',
        'embed': 'timeF',
        'freq': 'd',
        'task_name': 'long_term_forecast',
        'factor': 3,
        'patch_len': 12,
        'stride': 6,
    }
    return common


def scale_and_split_data(df_all: pd.DataFrame):
    """Split data and create per-column scalers (fit only on train)"""
    test_data    = df_all.iloc[-TEST_DAYS:]
    pre_test_df  = df_all.iloc[:-TEST_DAYS]
    val_size     = int(len(pre_test_df) * VAL_FRACTION)
    val_data     = pre_test_df.iloc[-val_size:]
    train_data   = pre_test_df.iloc[:-val_size]

    columns = df_all.columns.tolist()

    scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

    def apply_scaling(df):
        arr = np.hstack([scalers[col].transform(df[[col]]) for col in columns])
        return pd.DataFrame(arr, index=df.index, columns=columns)

    scaled_train = apply_scaling(train_data).values.astype(np.float32)
    scaled_val   = apply_scaling(val_data).values.astype(np.float32)
    scaled_test  = apply_scaling(test_data).values.astype(np.float32)

    return scaled_train, scaled_val, scaled_test, test_data, columns, scalers


def main():
    print("Loading and preparing multi-coin data...")
    df_all = load_and_combine_coin_data()

    # Time features
    dates = df_all.index
    time_feat = time_features(dates, freq='d').T.astype(np.float32)

    scaled_train, scaled_val, scaled_test, test_data, columns, scalers = \
        scale_and_split_data(df_all)

    n_features = len(columns)

    pre_test_values = np.concatenate((scaled_train, scaled_val))
    pre_test_time   = time_feat[:len(pre_test_values)]
    test_time       = time_feat[-TEST_DAYS:]

    configs = get_patchtst_config(n_features)
    model_name = 'PatchTST'
    print(f"\n=== Training {model_name} ===")

    args = argparse.Namespace(**configs)
    model = PatchTST(args).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler_cosine   = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                          patience=100, min_lr=1e-8, verbose=True)

    train_ds = TimeSeriesForecastDataset(scaled_train, pre_test_time[:len(scaled_train)], SEQ_LEN, LABEL_LEN, PRED_LEN)
    val_ds   = TimeSeriesForecastDataset(scaled_val,   pre_test_time[len(scaled_train):],   SEQ_LEN, LABEL_LEN, PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]

            # Light Gaussian noise augmentation (same as your custom script)
            x = x + torch.randn_like(x) * 0.0015

            dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                 torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

            out = model(x, x_mark, dec_inp, y_mark)
            pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

            loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item() * x.size(0)  # better averaging

        train_avg = train_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                     torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])

                val_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

        val_loss_avg = val_loss / n_samples if n_samples > 0 else float('inf')

        scheduler_cosine.step()
        scheduler_plateau.step(val_loss_avg)

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch+1:5d}]  train: {train_avg:.7f}   val: {val_loss_avg:.7f}   lr: {current_lr:.2e}")

        if val_loss_avg < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss_avg
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()
        gc.collect()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val loss: {best_val_loss:.8f})")

    # ─── One-step-ahead evaluation ───────────────────────────────────────────────

    full_values = np.concatenate((pre_test_values, scaled_test))
    full_time   = np.concatenate((pre_test_time, test_time))

    test_ds = TimeSeriesForecastDataset(full_values, full_time, SEQ_LEN, LABEL_LEN, PRED_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
            dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                 torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

            out = model(x, x_mark, dec_inp, y_mark)
            pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

            preds_list.append(pred.cpu().numpy())
            trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

    preds = np.concatenate(preds_list, axis=0).reshape(-1, n_features)
    trues = np.concatenate(trues_list, axis=0).reshape(-1, n_features)

    start_idx = len(pre_test_values) - SEQ_LEN
    preds_test = preds[start_idx : start_idx + TEST_DAYS]
    trues_test = trues[start_idx : start_idx + TEST_DAYS]

    err = preds_test - trues_test
    global_mse  = np.mean(err ** 2)
    global_rmse = math.sqrt(global_mse)
    global_mae  = np.mean(np.abs(err))

    print(f"{model_name} global RMSE (scaled): {global_rmse:.6f} | MAE: {global_mae:.6f}")

    # Per-coin results
    per_coin_results = []
    for coin in COINS:
        coin_cols = [f"{coin}_{v}" for v in ['open','high','low','close','volume']]
        idxs = [columns.index(c) for c in coin_cols]

        err_coin = preds_test[:, idxs] - trues_test[:, idxs]
        mse  = np.mean(err_coin ** 2)
        rmse = math.sqrt(mse)
        mae  = np.mean(np.abs(err_coin))

        per_coin_results.append({
            'model': model_name,
            'coin': coin,
            'mse_scaled': mse,
            'rmse_scaled': rmse,
            'mae_scaled': mae,
        })

    # Overfit check on train
    train_loader_eval = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    train_preds_list, train_trues_list = [], []
    with torch.no_grad():
        for batch in train_loader_eval:
            x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
            dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                 torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

            out = model(x, x_mark, dec_inp, y_mark)
            pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

            train_preds_list.append(pred.cpu().numpy())
            train_trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

    train_preds = np.concatenate(train_preds_list, axis=0).reshape(-1, n_features)
    train_trues = np.concatenate(train_trues_list, axis=0).reshape(-1, n_features)

    train_mse = np.mean((train_preds - train_trues) ** 2)

    print(f"Train MSE: {train_mse:.6f} | Test MSE: {global_mse:.6f} | Gap: {global_mse - train_mse:.6f}")

    # Inverse scale predictions for saving
    pred_prices = np.zeros_like(preds_test)
    for i, col in enumerate(columns):
        pred_prices[:, i] = scalers[col].inverse_transform(
            preds_test[:, i].reshape(-1, 1)
        ).ravel()

    pred_df = pd.DataFrame(pred_prices, index=test_data.index, columns=columns)

    df_export = test_data.add_suffix('_true').copy()
    for col in columns:
        df_export[f'{col}_one_step'] = pred_df[col]

    df_export.to_csv(f'forecasts_{model_name}_full_ohlcv.csv')

    # Save results
    pd.DataFrame([{
        'model': model_name,
        'test_mse_global': global_mse,
        'test_rmse_global': global_rmse,
        'test_mae_global': global_mae,
    }]).to_csv('one_step_patchtst_global.csv', index=False)

    pd.DataFrame(per_coin_results).to_csv('one_step_patchtst_per_coin.csv', index=False)

    pd.DataFrame([{
        'model': model_name,
        'train_mse_scaled': train_mse,
        'test_mse_scaled': global_mse,
        'overfit_gap_scaled': global_mse - train_mse
    }]).to_csv('overfit_patchtst.csv', index=False)

    print("\nBenchmark finished.")
    print("Saved files:")
    print("  • one_step_patchtst_global.csv")
    print("  • one_step_patchtst_per_coin.csv")
    print("  • overfit_patchtst.csv")
    print("  • forecasts_PatchTST_full_ohlcv.csv")


if __name__ == "__main__":
    main()