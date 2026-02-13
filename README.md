```markdown
# OmniFormer

State-of-the-Art **Patch-based Transformer** designed for **long-term multi-asset cryptocurrency price forecasting** (daily OHLCV).

Focuses on **one-step-ahead** prediction of 10 major cryptocurrencies simultaneously, using shared representation learning across assets.

```
thequantscientist-omniformer/
├── README.md
└── src/
    ├── Benchmark/
    │   ├── patchtst.py               # PatchTST benchmark (from Time-Series-Library)
    │   └── transformer_benchmark.py  # Autoformer / FEDformer / Informer / iTransformer
    └── OmniFormer/
        ├── ablation_study.py         # Component ablation experiments
        ├── backtesting.py            # Simple trading strategy simulation
        ├── computation.py            # Training + inference with energy & GPU tracking
        ├── full_pipeline.py          # (likely end-to-end script – binary in provided tree)
        └── OmniFormer.py             # Core model training + evaluation
```

## Key Features

- Multi-asset modeling (10 cryptocurrencies jointly)
- Patch-based Transformer architecture inspired by PatchTST
- **Reversible Instance Normalization** (RevIN)
- Channel-wise mixing after temporal attention
- Positional embeddings on patches
- Strong emphasis on **ablation studies** and **energy/carbon footprint** awareness
- Realistic **backtesting** with transaction costs, exposure limits, ranking strategies

## Supported Cryptocurrencies

| Symbol   | Pair      |
|----------|-----------|
| BTC      | BTCUSDT   |
| ETH      | ETHUSDT   |
| BNB      | BNBUSDT   |
| SOL      | SOLUSDT   |
| ADA      | ADAUSDT   |
| AVAX     | AVAXUSDT  |
| DOGE     | DOGEUSDT  |
| LINK     | LINKUSDT  |
| TRX      | TRXUSDT   |
| XRP      | XRPUSDT   |

## Model Highlights

- Input sequence length: **90 days**
- Prediction: **next day close** (one-step-ahead)
- Architecture: Patch Transformer + RevIN + Channel Mixer
- Embedding dim: **384**
- Heads / Layers: **6 heads**, **4 encoder layers**
- Patch length / stride: **12 / 6**
- Optimizer: **AdamW** + cosine annealing + plateau reduction
- Regularization: dropout 0.1, gradient clipping, light Gaussian noise

## Benchmark Comparison

The repository includes reference implementations (adapted from [Time-Series-Library](https://github.com/thuml/Time-Series-Library)) of:

- PatchTST
- Autoformer
- FEDformer
- Informer
- iTransformer (optional)

for fair comparison on the same multi-asset daily crypto dataset.

## Results & Outputs

Typical generated files:

- `one_step_OmniFormer_global.csv`  
- `one_step_OmniFormer_per_coin.csv`  
- `OmniFormer_overfit_check.csv`  
- `forecasts_transformer.csv` (true vs predicted OHLCV)  
- `ablation_*.csv` (component importance)  
- Energy & emission logs (via codecarbon)

## Installation

```bash
# Recommended: Python 3.10–3.12
pip install torch pandas numpy scikit-learn codecarbon pynvml psutil

# Optional – for original benchmark models
git clone https://github.com/thuml/Time-Series-Library.git
# then adjust sys.path or install it properly
```

## Usage

```bash
# 1. Core model training + evaluation
cd src/OmniFormer
python OmniFormer.py

# 2. Run ablation study
python ablation_study.py

# 3. Compare against classic transformers
cd ../Benchmark
python patchtst.py
python transformer_benchmark.py

# 4. Simple backtest on predictions
python backtesting.py
# (edit input file path inside script if needed)

# 5. Track energy consumption & GPU usage
python computation.py
```

## Project Status (as of Feb 2026)

- Active development
- Focus on interpretability, efficiency, and real-world trading simulation
- Looking for collaborators interested in:
  - multi-step forecasting
  - volatility / VaR modeling
  - cross-market (crypto + equities) extensions
  - quantization / pruning for edge deployment

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{omniformer2025,
  author = {Nguyen Quoc Anh},
  title  = {OmniFormer: Patch Transformer for Multi-Asset Cryptocurrency Forecasting},
  year   = {2025},
  url    = {https://github.com/thequantscientist-omniformer}
}
```

## License

MIT

---