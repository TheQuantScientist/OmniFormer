import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, wasserstein_distance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────

file_paths = {
    'ADAUSDT':  'Your Path',
    'AVAXUSDT': 'Your Path',
    'BNBUSDT':  'Your Path',
    'BTCUSDT':  'Your Path',
    'DOGEUSDT': 'Your Path',
    'ETHUSDT':  'Your Path',
    'SOLUSDT':  'Your Path',
    'LINKUSDT': 'Your Path',
    'TRXUSDT':  'Your Path',
    'XRPUSDT':  'Your Path'
}

seq_len = 90
pred_len = 1
epochs = 200
batch_size = 214
lr = 1e-4
test_size = 365
horizons = [1, 7, 30, 90, 180, 365]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()

# ────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────

dfs = {}
for k, p in file_paths.items():
    df = pd.read_csv(p)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    dfs[k] = df.set_index('timestamp')

df_all = None
for c, df in dfs.items():
    df = df.rename(columns={
        'open': f'{c}_open',
        'high': f'{c}_high',
        'low': f'{c}_low',
        'close': f'{c}_close',
        'volume': f'{c}_volume'
    })
    df_all = df if df_all is None else df_all.join(df, how='outer')

df_all = df_all.ffill().bfill().sort_index()
columns = df_all.columns.tolist()
n_features = len(columns)

coins = list(file_paths.keys())
close_columns = [f'{c}_close' for c in coins]
close_idxs = [columns.index(c) for c in close_columns]

# ────────────────────────────────────────────────
# Split + scale
# ────────────────────────────────────────────────

test_data = df_all.iloc[-test_size:]
pre_test_data = df_all.iloc[:-test_size]

val_size = int(len(pre_test_data) * 0.15)
train_data = pre_test_data.iloc[:-val_size]
val_data = pre_test_data.iloc[-val_size:]

scalers = {c: MinMaxScaler().fit(train_data[[c]]) for c in columns}

def scale(df):
    return np.hstack([scalers[c].transform(df[[c]]) for c in columns]).astype(np.float32)

train_scaled = scale(train_data)
val_scaled = scale(val_data)
test_scaled = scale(test_data)
pre_test_scaled = np.concatenate([train_scaled, val_scaled], axis=0)

# ────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - seq_len - pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+seq_len]
        y = self.data[idx+seq_len:idx+seq_len+pred_len]
        return torch.tensor(x), torch.tensor(y)

# ────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────

class VanillaRNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, input_size)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1]).unsqueeze(1)

class LSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, input_size)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1]).unsqueeze(1)

class GRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, input_size)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1]).unsqueeze(1)

class CLAM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 200, 3, batch_first=True)
        self.attn = nn.Linear(200, 1)
        self.fc = nn.Linear(200, input_size)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = (w * x).sum(dim=1)
        return self.fc(x).unsqueeze(1)

models = [
    ('RNN', VanillaRNN),
    ('LSTM', LSTM),
    ('GRU', GRU),
    ('CLAM', CLAM)
]

# ────────────────────────────────────────────────
# Results
# ────────────────────────────────────────────────

one_step, multi, dist, vol, rng, corr, direc, overfit = ([] for _ in range(8))

# ────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────

for name, cls in models:
    print(f'\n=== Training {name} ===')
    model = cls(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TimeSeriesDataset(pre_test_scaled), batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    # One-step
    full = np.concatenate([pre_test_scaled, test_scaled])
    loader = DataLoader(TimeSeriesDataset(full), batch_size, shuffle=False)

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())

    preds = np.concatenate(preds)[:, 0]
    trues = np.concatenate(trues)[:, 0]

    start = len(pre_test_scaled) - seq_len
    p_test = preds[start:start+test_size]
    t_test = trues[start:start+test_size]

    mse = np.mean((p_test - t_test) ** 2)
    one_step.append({'model': name, 'test_mse': mse, 'test_rmse': math.sqrt(mse), 'test_mae': np.mean(np.abs(p_test - t_test))})

    # Overfit
    train_preds = preds[:len(train_scaled)]
    train_trues = trues[:len(train_scaled)]
    train_mse = np.mean((train_preds - train_trues) ** 2)
    overfit.append({'model': name, 'train_mse': train_mse, 'test_mse': mse, 'overfit_gap': mse - train_mse})

    # Autoregressive
    current = torch.tensor(pre_test_scaled[-seq_len:]).unsqueeze(0).to(device)
    forecast = []

    with torch.no_grad():
        for _ in range(test_size):
            nxt = model(current)[:, 0]
            forecast.append(nxt.cpu().numpy())
            current = torch.cat([current[:, 1:], nxt.unsqueeze(1)], dim=1)

    forecast = np.vstack(forecast)

    # Multi-horizon
    mh = {'model': name, 'horizons': horizons, 'multi_mse': [], 'multi_rmse': [], 'multi_mae': []}
    for h in horizons:
        hh = min(h, test_size)
        e = forecast[:hh] - test_scaled[:hh]
        mh['multi_mse'].append(np.mean(e**2))
        mh['multi_rmse'].append(np.sqrt(np.mean(e**2)))
        mh['multi_mae'].append(np.mean(np.abs(e)))
    multi.append(mh)

    # Inverse closes
    fc = np.zeros((test_size, len(coins)))
    tc = np.zeros_like(fc)
    hist = pre_test_data[close_columns].iloc[-365:].values

    for i, c in enumerate(coins):
        idx = close_idxs[i]
        fc[:, i] = scalers[f'{c}_close'].inverse_transform(forecast[:, idx:idx+1]).ravel()
        tc[:, i] = scalers[f'{c}_close'].inverse_transform(test_scaled[:, idx:idx+1]).ravel()

    dist.append({'model': name,
                 'avg_ks_stat': np.mean([ks_2samp(fc[:, i], hist[:, i])[0] for i in range(len(coins))]),
                 'avg_wasserstein': np.mean([wasserstein_distance(fc[:, i], hist[:, i]) for i in range(len(coins))])})

    v = []
    for i in range(len(coins)):
        f = np.diff(np.log(fc[:, i] + 1e-8))
        t = np.diff(np.log(tc[:, i] + 1e-8))
        v.append(np.mean(np.abs(pd.Series(f).rolling(30, 1).std() - pd.Series(t).rolling(30, 1).std())))
    vol.append({'model': name, 'avg_vol_mae': np.mean(v)})

    rng.append({'model': name, 'avg_violations': np.mean([(fc[:, i] < 0).sum() for i in range(len(coins))])})

    corr.append({'model': name, 'corr_frobenius': np.linalg.norm(np.corrcoef(fc.T) - np.corrcoef(tc.T), 'fro')})

    direc.append({'model': name,
                   'avg_dir_acc': np.mean([np.mean(np.sign(np.diff(fc[:, i])) == np.sign(np.diff(tc[:, i])))
                                            for i in range(len(coins))])})

# ────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────

pd.DataFrame(one_step).to_csv('one_step_results.csv', index=False)
pd.DataFrame(multi).to_csv('multi_horizon_results.csv', index=False)
pd.DataFrame(dist).to_csv('dist_results.csv', index=False)
pd.DataFrame(vol).to_csv('vol_results.csv', index=False)
pd.DataFrame(rng).to_csv('range_results.csv', index=False)
pd.DataFrame(corr).to_csv('corr_results.csv', index=False)
pd.DataFrame(direc).to_csv('dir_acc_results.csv', index=False)
pd.DataFrame(overfit).to_csv('overfit_results.csv', index=False)

print('\nBenchmark complete.')