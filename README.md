# 🐎 Multimodal Hybrid Deep Learning for Cryptocurrency Volatility Forecasting: Quantifying Gains Beyond Persistence Baselines

<div align="center">

[![Status](https://img.shields.io/badge/Status-Under%20Review%20%40%20IEEE%20TAI-00629B.svg?logo=ieee&logoColor=white)](https://cis.ieee.org/publications/t-ai)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-FinBERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/ProsusAI/finbert)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Official PyTorch/TensorFlow Implementation & Benchmarking Framework**

---

### ✍️ Authors & Affiliations

**Mohammed Alaa Ala'anzy**<sup>1,✉</sup> &nbsp;•&nbsp; **Yeskendir Kaliyev**<sup>1</sup>

<sup>1</sup>*SDU University, Kaskelen, Almaty Region, Kazakhstan*  
<sup>✉</sup>*Corresponding Author:* [`m.alanzy@ieee.org`](mailto:m.alanzy@ieee.org) &nbsp;|&nbsp; [`230107021@sdu.edu.kz`](mailto:230107021@sdu.edu.kz)

> 📑 *This research article is currently **under review** in the **IEEE Transactions on Artificial Intelligence** (IEEE TAI).*

---

[Overview](#-overview) •
[Architecture](#-system-architecture) •
[Data Pillars](#-the-three-data-pillars) •
[Model Zoo](#-model-zoo--horserace-competitors) •
[Evaluation Suite](#-evaluation--benchmarking-suite) •
[Quick Start](#-quick-start) •
[Configuration](#-configuration--experiment-presets) •
[Citation](#-citation--research-attribution)

</div>

---

## 📖 Overview

This repository hosts the official experimental framework and benchmark codebase for the paper **"Multimodal Hybrid Deep Learning for Cryptocurrency Volatility Forecasting: Quantifying Gains Beyond Persistence Baselines"** (submitted to *IEEE Transactions on Artificial Intelligence*).

The study rigorously assesses whether modern **multimodal deep learning architectures** (fusing market price dynamics, NLP news sentiment, and on-chain fundamental network features) achieve statistically significant forecasting gains over robust econometric and persistence baselines for **daily realized cryptocurrency volatility forecasting** ($\sigma_{t+h}$).

### 🎯 Core Research Hypothesis
> *"Asset-specific market dynamics, NLP-derived sentiment, and on-chain network fundamentals provide complementary non-linear signals that significantly outperform traditional econometric models, especially during market crises and speculative regimes."*

### 🌟 Key Highlights
* 🔬 **Tri-Pillar Multimodal Fusion:** Seamlessly joins market price action, Fear & Greed / FinBERT sentiment, and on-chain network fundamentals.
* 🧠 **Flagship Deep Learning Model:** 1D-CNN (Local Feature Extraction) $\rightarrow$ Bidirectional LSTM (Temporal Dynamics) $\rightarrow$ Multi-Head Self-Attention (Cross-Timestep Interpretability).
* ⚔️ **Rigorous Econometric Horserace:** ARIMA(5,1,0), GARCH(1,1), GJR-GARCH(1,1,1), HAR-RV, SVR (RBF), Naive Persistence, and DL Ablations.
* 🛡️ **Leakage-Free Multi-Horizon Design:** Point-in-time windowing with strictly isolated train-only feature & target scalers across forecast horizons ($h \in \{1, 3, 7\}$ days).
* 📊 **Exhaustive Evaluation Suite:** Diebold-Mariano tests with HAC variance, regime-aware crisis slicing (e.g., LUNA & FTX crashes), MC Dropout uncertainty intervals, and volatility timing backtests.

---

## 🏛 System Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 THREE DATA PILLARS                     │
                  └────────────────────────────────────────────────────────┘
                       │                        │                       │
           ┌───────────┴──────────┐ ┌───────────┴──────────┐ ┌──────────┴───────────┐
           │   Pillar 1: Market   │ │  Pillar 2: Sentiment │ │   Pillar 3: On-Chain  │
           │  • Yahoo Finance     │ │  • Alternative.me    │ │  • CoinMetrics API   │
           │  • CoinMarketCap API │ │  • CryptoPanic News  │ │  • Glassnode API     │
           │  • OHLCV + Log-Ret   │ │  • ProsusAI/FinBERT  │ │  • Active Addr & Tx  │
           └───────────┬──────────┘ └───────────┬──────────┘ └──────────┬───────────┘
                       │                        │                       │
                       └────────────────────────┼───────────────────────┘
                                                ▼
                  ┌────────────────────────────────────────────────────────┐
                  │             PHASE 1: ETL & FUSION PIPELINE             │
                  │  • Time-aligned (Date, Asset) Left Joins               │
                  │  • Forward/Backward Fill for Missing Intervals         │
                  │  • Strict Data Preflight Quality Checks (PAPER_MODE)   │
                  │  • Train-Only MinMaxScaler (No Look-Ahead Leakage)     │
                  │  • Lookback Window Generator (L=14 days, Target t+h)   │
                  └─────────────────────────────┬──────────────────────────┘
                                                ▼
            ┌───────────────────────────────────┴───────────────────────────────────┐
            │                                                                       │
            ▼                                                                       ▼
┌───────────────────────────────┐                       ┌───────────────────────────────────────┐
│ PHASE 2: ECONOMETRIC BASELINES│                       │   PHASE 3: DEEP LEARNING ARCHITECTURE │
│ • ARIMA(5, 1, 0)              │                       │  ┌─────────────────────────────────┐  │
│ • GARCH(1, 1)                 │                       │  │ Input Tensor: (Batch, L=14, F)  │  │
│ • GJR-GARCH(1, 1, 1)          │                       │  └────────────────┬────────────────┘  │
│ • HAR-RV (OLS Components)     │                       │                   ▼                   │
│ • Naive Persistence           │                       │  ┌─────────────────────────────────┐  │
│ • SVR (RBF Kernel)            │                       │  │ 1D-CNN (Conv1D + BatchNorm)     │  │
│ • Hybrid (ARIMA + DL)         │                       │  └────────────────┬────────────────┘  │
│ • Residual Hybrid (ARIMA+Res) │                       │                   ▼                   │
└───────────────┬───────────────┘                       │  ┌─────────────────────────────────┐  │
                │                                       │  │ Bidirectional LSTM (32 Units)   │  │
                │                                       │  └────────────────┬────────────────┘  │
                │                                       │                   ▼                   │
                │                                       │  ┌─────────────────────────────────┐  │
                │                                       │  │ Multi-Head Attention (4 Heads)  │  │
                │                                       │  └────────────────┬────────────────┘  │
                │                                       │                   ▼                   │
                │                                       │  ┌─────────────────────────────────┐  │
                │                                       │  │ GlobalAvgPool1D + Dense Output  │  │
                │                                       │  └────────────────┬────────────────┘  │
                │                                       └───────────────────┼───────────────────┘
                │                                                           │
                └─────────────────────────────┬─────────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            PHASE 4: BENCHMARKING & EVALUATION          │
                  │  • Performance Metrics: RMSE, MAE, DoC, Test R²        │
                  │  • Diebold-Mariano Equal Accuracy Tests (Newey-West)   │
                  │  • Crisis Shock Window Performance (LUNA & FTX)        │
                  │  • Market Regime Slices (Calm, Bull, Bear, Crisis)     │
                  │  • Multi-Head Attention Temporal & Feature Heatmaps    │
                  │  • MC Dropout Uncertainty Quantification (90% CI)      │
                  │  • Volatility Timing Strategy Backtest (Sharpe/Sortino)│
                  └────────────────────────────────────────────────────────┘
```

---

## 🌐 The Three Data Pillars

The pipeline aggregates heterogeneous signals across three foundational pillars into daily time-series panels:

| Pillar | Data Source | Features Extracted | Description |
| :--- | :--- | :--- | :--- |
| **📈 Pillar 1: Market** | `yfinance` & CoinMarketCap | `Open`, `High`, `Low`, `Close`, `Volume`, `Log_Return`, `CMC_Close`, `CMC_Volume` | Captures core price action, intraday ranges, liquidity dynamics, and cross-exchange validated close prices. |
| **🧠 Pillar 2: Sentiment** | Alternative.me & CryptoPanic | `Market_FearGreed`, `FinBERT_Polarity` | Tracks macro crowd sentiment and contextual sentiment polarity scores from financial news headlines via FinBERT. |
| **⛓️ Pillar 3: On-Chain** | Coin Metrics & Glassnode | `Active_Addresses`, `Tx_Count`, `Transfer_Volume_USD`, `Hash_Rate`, `Network_Cap_USD` | Fundamental network usage, miner security/computational commitment, and settlement volume on the ledger. |

> 🎯 **Target Variable ($\sigma$):** 14-day rolling realized standard deviation of daily log-returns:
> $$\sigma_t = \sqrt{\frac{1}{N-1} \sum_{i=0}^{N-1} \left( r_{t-i} - \bar{r}_t \right)^2}$$

---

## 🥊 Model Zoo & Horserace Competitors

The benchmarking harness compares 10 distinct models under identical train/test splits:

### 1. Econometric & Statistical Baselines
* **`ARIMA(5,1,0)`**: Autoregressive Integrated Moving Average on realized volatility series with rolling $h$-step point forecasts.
* **`GARCH(1,1)`**: Generalized Autoregressive Conditional Heteroskedasticity fitted on daily returns, forecasting conditional variance $\sigma_{t+h}^2$.
* **`GJR-GARCH(1,1,1)`**: Asymmetric GARCH accounting for leverage effects and negative return volatility shocks.
* **`HAR-RV`**: Heterogeneous Autoregressive model of Realized Volatility with Daily ($RV_d$), Weekly ($RV_w$), and Monthly ($RV_m$) components.

### 2. Machine Learning & Ensembles
* **`Naive Persistence`**: One/Multi-step persistence benchmark predicting volatility at $t+h$ using the latest observed volatility at forecast origin $t$.
* **`SVR (RBF)`**: Support Vector Regression with Radial Basis Function kernel trained on flattened lookback windows.
* **`Hybrid (ARIMA + DL)`**: Linear ensemble combining econometric time-series forecast with deep neural network representation.
* **`Residual Hybrid (ARIMA + DL-resid)`**: Two-stage model where the deep neural network learns to predict ARIMA forecast residuals.

### 3. Deep Learning Architectures
* **`CNN-only`**: 1D Convolutional Neural Network with Batch Normalization and Global Average Pooling.
* **`LSTM-only`**: Unidirectional LSTM layer for sequential modeling.
* **`GRU-only`**: Gated Recurrent Unit baseline.
* **`CNN-BiLSTM-Attention` (Flagship)**:
  1. **Conv1D**: Captures localized multi-feature temporal patterns.
  2. **Bidirectional LSTM**: Encodes past-to-future and future-to-past temporal context.
  3. **Multi-Head Self-Attention**: Discovers long-range inter-timestep and cross-feature dependencies while preserving interpretability.

---

## 📊 Evaluation & Benchmarking Suite

```
results/
├── 📄 comparison_table_BTC_USD_h1.csv      # Primary metrics: RMSE, MAE, DoC, R²
├── 📄 dm_tests_BTC_USD_h1.csv               # Pairwise Diebold-Mariano tests & p-values
├── 📄 shock_window_metrics_BTC_USD_h1.csv   # Performance in stress windows (LUNA, FTX)
├── 📄 regime_metrics_BTC_USD_h1.csv         # Performance across Calm, Bull, Bear, Crisis
├── 📄 backtest_metrics_BTC_USD_h1.csv       # Volatility timing Sharpe, Sortino, Drawdown
├── 📄 interval_metrics_BTC_USD_h1.csv       # MC Dropout prediction interval coverage & width
├── 📄 sensitivity_metrics_BTC_USD_h1.csv    # Robustness against missing data & noise
├── 🖼️ predictions_overlay_BTC_USD_h1.png   # Time-series actual vs predicted overlay
├── 🖼️ attention_heatmap_BTC_USD_h1.png     # Attention weights heatmap across timesteps
├── 🖼️ feature_importance_BTC_USD_h1.png    # Attention-weighted feature importance
└── 🖼️ training_curves_BTC_USD_h1.png       # Training vs Validation loss & MAE
```

### Metrics Computed

| Metric | Formulation / Purpose |
| :--- | :--- |
| **RMSE** | $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^N (y_t - \hat{y}_t)^2}$ — Penalizes large forecast misses. |
| **MAE** | $\text{MAE} = \frac{1}{N}\sum_{t=1}^N \|y_t - \hat{y}_t\|$ — Measures median absolute magnitude of error. |
| **DoC** | Direction of Change (Directional Accuracy): percentage of correct sign predictions on $\Delta y_t$. |
| **Test $R^2$** | Coefficient of determination on the strictly held-out test partition. |
| **Diebold-Mariano** | Tests null hypothesis $H_0: \mathbb{E}[d_t] = 0$ of equal predictive accuracy with Newey-West HAC variance. |
| **MC Dropout** | Computes epistemic uncertainty bounds via $T=50$ Monte Carlo forward passes with active dropout. |

---

## 📁 Repository Structure

```
crypto-horserace/
├── ⚙️ config.py                     # Central configuration (hyperparameters, tickers, paths)
├── 🚀 main.py                       # Main pipeline orchestrator (Phases 1-4)
├── 📋 requirements.txt              # Standard CPU/GPU Python dependencies
├── 🐧 requirements-gpu-wsl.txt      # WSL2 NVIDIA GPU configuration
├── 🔐 .env.example                  # Template for API credentials
├── 📜 README.md                     # Project documentation
│
├── 📂 scripts/                      # Utility and setup scripts
│   ├── 🔄 recompute_sensitivity_doc.py
│   └── 🐧 wsl_tensorflow_gpu_setup.sh
│
├── 📂 src/                          # Core source code modules
│   ├── 📦 __init__.py
│   ├── 🔀 asset_utils.py            # Ticker alias resolver & metadata
│   ├── 📈 baselines.py              # ARIMA, GARCH, GJR-GARCH, HAR-RV baselines
│   ├── 🛠️ collection_utils.py       # Rate-limiting, caching & date normalizers
│   ├── 📥 data_collection.py        # Three-pillar orchestrator
│   ├── 🛡️ data_quality.py           # Preflight integrity & missingness audits
│   ├── 📊 evaluate.py               # Metrics, DM tests, regime tables & plots
│   ├── 🔄 fuse_data.py              # Data alignment, windowing & scaling
│   ├── 🏇 horserace_baselines.py     # Naive persistence & SVR baselines
│   ├── 💹 ingest_market.py          # Yahoo Finance market ingestion
│   ├── 🪙 ingest_market_cmc.py      # CoinMarketCap API client
│   ├── ⛓️ ingest_onchain.py         # Coin Metrics & Glassnode client
│   ├── 📰 ingest_sentiment.py       # Fear & Greed + CryptoPanic client
│   ├── 🧠 model_cnn_lstm.py         # CNN-BiLSTM-Attention Keras architecture
│   ├── 🧬 model_dl_baselines.py     # CNN-only, LSTM-only, GRU-only baselines
│   └── 🤖 sentiment_finbert.py      # HuggingFace FinBERT sentiment classifier
│
├── 📂 data/                         # Persistent datasets (created at runtime)
│   ├── 📂 raw/                      # Unprocessed API responses & CSVs
│   ├── 📂 features/                 # Cleaned per-pillar feature tables
│   └── 📂 processed/                # Fully fused multimodal panels
│
└── 📂 results/                      # Output figures, tables & benchmark reports
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/crypto-horserace.git
cd crypto-horserace

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS / WSL:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

Copy `.env.example` to `.env` and provide your API keys for live feeds:

```bash
cp .env.example .env
```

```ini
# .env
COINMARKETCAP_API_KEY=your_cmc_key_here
COINMETRICS_API_KEY=your_coinmetrics_key_here
CRYPTOPANIC_API_KEY=your_cryptopanic_key_here
GLASSNODE_API_KEY=your_glassnode_key_here
ENABLE_FINBERT_ON_INGEST=0
```

> 💡 *Note: If no API keys are provided, the system seamlessly utilizes public endpoints (Yahoo Finance, Alternative.me Fear & Greed, and Coin Metrics Community).*

### 3. Run the Full Horserace Benchmark

```bash
python main.py
```

---

## 🐧 Hardware Acceleration (WSL2 GPU Setup)

Native Windows TensorFlow 2.11+ does not support direct GPU execution. For full NVIDIA CUDA GPU acceleration on Windows, execute inside **WSL2 Ubuntu**:

```bash
# Inside WSL2 terminal:
chmod +x scripts/wsl_tensorflow_gpu_setup.sh
./scripts/wsl_tensorflow_gpu_setup.sh

# Activate GPU virtual environment
source .venv-wsl/bin/activate

# Run pipeline on GPU
python main.py
```

---

## 🎛 Configuration & Experiment Presets

All pipeline settings can be modified in [`config.py`](file:///c:/crypto-horserace-main/config.py) or overridden dynamically via environment variables:

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `CORE_TICKERS` | `["BTC-USD", "ETH-USD", "DOGE-USD", "XRP-USD"]` | Asset universe benchmarked. |
| `LOOKBACK_WINDOW` | `14` | History length in days fed into models ($L$). |
| `EXPERIMENT_HORIZONS`| `[1, 3, 7]` | Forecasting horizons ($h$ days ahead). |
| `PAPER_MODE` | `True` | Fails preflight if synthetic data is detected. |
| `TRAIN_RATIO` | `0.80` | Chronological train/test split boundary. |
| `CNN_FILTERS` | `32` | Number of 1D convolutional filters. |
| `LSTM_UNITS` | `32` | Hidden units per direction in BiLSTM. |
| `NUM_ATTENTION_HEADS`| `4` | Number of parallel self-attention heads. |
| `DROPOUT_RATE` | `0.40` | Dropout probability for regularization. |

### Experiment Matrix Presets

Set the `CRYPTO_HORSERACE_DATA_EXPERIMENT` environment variable to run preset benchmark suites:

```bash
# Extended timeframe evaluation (2020 - 2025)
export CRYPTO_HORSERACE_DATA_EXPERIMENT="extended_calendar"

# Intraday 1-hour BTC focus experiment
export CRYPTO_HORSERACE_DATA_EXPERIMENT="hourly_btc_focus"

# Extended 6-asset universe (including SOL & SHIB)
export CRYPTO_HORSERACE_DATA_EXPERIMENT="six_assets"

python main.py
```

---

## 📜 Citation & Research Attribution

If you find this research, dataset pipeline, or benchmark methodology useful in your academic work, please cite:

```bibtex
@article{alaanzy2026multimodal,
  title   = {Multimodal Hybrid Deep Learning for Cryptocurrency Volatility Forecasting: Quantifying Gains Beyond Persistence Baselines},
  author  = {Ala'anzy, Mohammed Alaa and Kaliyev, Yeskendir},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2026},
  note    = {Under Review. SDU University}
}
```

---

<div align="center">
  <sub>Research conducted at <strong>SDU University</strong> (Department of Computer Science / Engineering).</sub>
</div>
