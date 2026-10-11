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
[Results](#-results) •
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
| **📈 Pillar 1: Market** | `yfinance` & CoinMarketCap (optional) | `Open`, `High`, `Low`, `Close`, `Volume`, `Log_Return`, plus `CMC_Close`/`CMC_Volume` if a CoinMarketCap key is configured | Captures core price action, intraday ranges, and liquidity dynamics. |
| **🧠 Pillar 2: Sentiment** | Alternative.me, CryptoPanic (optional), FinBERT (optional) | `Market_FearGreed`, `FinBERT_Polarity`, `News_Sentiment`, `Social_Sentiment`, `Attention_Index` | Tracks macro crowd sentiment and, when a CryptoPanic key / text corpus is supplied, contextual polarity from financial news via FinBERT. Columns that reduce to an all-zero placeholder (no key configured) are auto-dropped from the model input — see [Performance & Reproducibility Controls](#-performance--reproducibility-controls). |
| **⛓️ Pillar 3: On-Chain** | Coin Metrics & Glassnode (optional) | `Active_Addresses`, `Tx_Count`, `Transfer_Volume_USD`, `Hash_Rate`, `Network_Cap_USD` | Fundamental network usage, miner security/computational commitment, and settlement volume on the ledger. |
| **🔁 Engineered: Lagged Volatility** | Derived from Pillar 1's own target series | `Volatility`, `RV_Week` (5-day trailing mean), `RV_Month` (22-day trailing mean) | HAR-style autoregressive features computed with the exact same trailing-window alignment as the HAR-RV baseline (`src/baselines.py::run_har_rv`), so they carry no look-ahead. Gives the DL model direct access to the same lagged-realized-volatility signal ARIMA/GARCH/HAR-RV already use as their primary input — see [Results](#-results). |

> 🎯 **Target Variable ($\sigma$):** 14-day rolling realized standard deviation of daily log-returns:
> $$\sigma_t = \sqrt{\frac{1}{N-1} \sum_{i=0}^{N-1} \left( r_{t-i} - \bar{r}_t \right)^2}$$

---

## 🥊 Model Zoo & Horserace Competitors

The benchmarking harness compares 12 distinct models under identical train/test splits:

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

Every neural forecast is the **mean of several seeds** (`DL_SEEDS`); the per-seed spread is written to `seed_metrics.csv` / `seed_summary.csv`. Interpretation, calibration and robustness analyses use the first seed's network.

```
results/
├── comparison_table_<A>_USD_h<h>.csv        # RMSE, MAE, DoC (+ binomial p vs 0.5), R², per model
├── dm_tests_<A>_USD_h<h>.csv                # Diebold-Mariano: lag h-1 + Harvey-Leybourne-Newbold, raw and Holm p-values
├── ratio_ci_<A>_USD_h<h>.csv                # RMSE ratio vs Naive / HAR-RV / ARIMA / KRP with moving-block bootstrap 95% CI
├── regime_metrics_<A>_USD_h<h>.csv          # calm / bull / bear / high_vol (terciles of realized vol in the test sample)
├── predictions_<A>_USD_h<h>.csv             # every model's aligned forecasts (figures and tests can be rebuilt offline)
├── attention_lag_profile_*.csv/.png         # temporal attention averaged over ALL test windows, heads and queries
├── attention_summary_*.csv                  # min/max weight, normalised entropy (1 = uniform), weight on the last 3 days
├── permutation_importance_*.csv/.png        # RMSE increase when a feature / pillar is shuffled (standalone network and AHI)
├── interval_metrics_*.csv                   # MC-dropout AND split-conformal coverage/width, standalone network and AHI
├── sensitivity_metrics_*.csv                # missing/noise tests on all channels and on sentiment+on-chain only, both models
├── ablation_results_*.csv, ablation_runs_*  # feature-group ablation, mean/std over ABLATION_SEEDS (BTC-USD)
├── rolling_*  (comparison, dm, regime, shock_window, ratio_ci, predictions, folds)
│                                            # rolling-origin evaluation from ROLLING_START that spans LUNA and FTX
├── cross_asset_generalization.csv           # per-asset-scaled transfer, every horizon
├── seed_metrics.csv, seed_summary.csv       # per-seed RMSE/MAE for every neural model
├── feature_audit.csv                        # which input channels were active, and why any were dropped
├── sample_manifest.csv, run_manifest.json   # sample sizes, test dates, purge counts, library versions, profile
└── paper_inputs/                            # produced by scripts/build_paper_inputs.py (all manuscript tables)
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

## 📈 Results

Results are not hard-coded in this README. After a run, `python scripts/build_paper_inputs.py` writes the aggregate tables to `results/paper_inputs/` (see `KEY_NUMBERS.md`), and the manuscript tables are generated from those files.

**Benchmarks that matter for the "gain beyond persistence" question.** Besides Naive Persistence, ARIMA and HAR-RV, the suite includes `KRP` (known-return persistence): the target is the sample std of the last 14 daily returns, so at origin *t* the 14-*h* returns already observed inside the target window are known. KRP uses them (parameter-free) and only forecasts the *h* unseen days. GARCH/GJR-GARCH are evaluated on the same estimand (`GARCH_ESTIMAND="window"`). Hybrid forecasts exist with an ARIMA stage (**AHI**) and a KRP stage (**AHI-K**).

---

## ▶️ Running it

```powershell
python tests/test_fixes.py                       # 10 checks, no training
$env:CRYPTO_HORSERACE_PROFILE="standard"; python main.py
```

* **Progress is shown.** Every network fit prints its Keras progress bar, and a banner like `TASK 7/26: rolling BTC-USD h=3 (elapsed ..., rough time left ... min)` precedes each task. TensorFlow's `use_unbounded_threadpool` log noise is silenced (`TF_CPP_MIN_LOG_LEVEL=3` is set in `config.py`).
* **Every experiment is a separate process (task).** TensorFlow/Keras leaks about 45 MB per trained model inside one Python process (measured), which made a single long run grow beyond 40 GB. Tasks are `combo` (hold-out evaluation of all models for one asset/horizon), `rolling`, `ablation` and `cross`.
* **Resume.** Finished tasks write `results/_done/<task>.json`; running the same command again skips them (unless the configuration changed). A crash costs one task, not the whole run. `CRYPTO_HORSERACE_RESUME=0` forces a fresh run.
* **One run per results folder.** An OS lock (`results/.run.lock`) stops a second copy from starting; use `CRYPTO_HORSERACE_RESULTS_DIR` for a separate folder.
* **Faster:** `python main.py --workers 3` runs three tasks at once (progress bars then go to `results/logs/<task>.log`).

| Profile | Seeds (standalone / AHI-K / AHI) | Rolling | Ablation | Transfer | Rough cost (CPU) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `smoke` | 2 / 2 / 1, 3 epochs | 2 epochs | 1 seed | h=1 | ~15 min, 2 assets x 1 horizon |
| `lite` | 1 / 1 / 1 | 240-day folds, 25 epochs | h=1, 1 seed | h=1, 1 seed | ~2-3 h |
| `standard` (default) | 2 / 2 / 1 | 180-day folds, 30 epochs | h=1, 2 seeds | h=1, 1 seed | ~3-5 h |
| `full` | 3 / 3 / 3 | 120-day folds, 40 epochs, 2 seeds, + ARIMA hybrid | h=1,3,7, 3 seeds | all horizons, 3 seeds | 10 h+ |

**Network inputs.** `CRYPTO_HORSERACE_FEATURE_SET=stationary` (default) feeds scale-free transforms (log return, |return|, ln(High/Low), log-volume change, ln(on-chain / 30-day mean), Fear & Greed, trailing realised volatility); `raw` feeds price/volume levels as in earlier runs. The ablation always includes `market_raw_levels` and `full_raw_levels` so the two are compared directly.

**Hybrids are gated.** AHI and AHI-K forecast `floor(stage + lambda * network)`, with `lambda` in {0, .25, .5, .75, 1} chosen on the last 15% of the training windows (per fold in the rolling evaluation). `results/gate_lambda.csv` reports lambda and the ungated test RMSE, so the effect of the gate is visible. Forecasts are floored at half the smallest training volatility.

---

## 🔬 Evaluation protocol (v2) and what changed

| Issue found in the earlier run | Fix |
| :--- | :--- |
| Naive Persistence targets were passed through the target scaler's *inverse* transform (RMSE 0.0189, R² -254), and the same corrupted series was the "actual" in the overlay figure and in DM pairs where Naive came first | Raw targets are used; `assert_common_actual` stops the run if any two models are scored on different realized series; the DM table refuses such pairs |
| Test sample is calendar 2024: LUNA and FTX lie in the training split, so the shock-window tables were empty and the "FTX" attention heatmap was test window 0 | New **rolling-origin evaluation** from `ROLLING_START` (2022-01-01) to the end of the sample: expanding window, networks re-fitted every `ROLLING_STEP_DAYS`, scalers re-fitted per fold, labels purged. Shock windows are populated; hold-out results are kept |
| Labels of the last h-1 training windows are realized after the first test origin | Purged (`PURGE_TRAIN_OVERLAP`) in the hold-out split, HAR-RV, and every rolling fold |
| GARCH/GJR forecast a one-day conditional std but are scored against a 14-day rolling std | `GARCH_ESTIMAND="window"` forecasts the rolling target itself; `"daily"` reproduces the old behaviour |
| Persistence baselines ignored the 14-*h* known returns inside the target window | New `KRP` baseline and AHI-K hybrid |
| Cross-asset transfer pooled raw price/volume channels under one scaler (BTC test closes mapped to 8-22 on a [0,1] scale) and ran at h=1 only | Every asset scaled with its own training statistics; all horizons |
| One seed per neural model (run-to-run RMSE spread ~20% at BTC h=1) | Seed ensembles; per-seed metrics saved; ablation with several seeds, mean/std, plus new `rv_only`, `market_rv`, `full_minus_sentiment`, `full_minus_onchain` configurations |
| Attention heatmap tiled one temporal weight across feature columns; feature importance used one test window and scaled with input magnitude | Lag profile over all windows and heads with entropy summary; permutation importance by feature and pillar on the whole test set |
| Robustness and MC-dropout analyses covered only the standalone network | Both models, plus split-conformal intervals; perturbations on all channels and on sentiment+on-chain only |
| DM: truncation lag h, normal-style reference, no multiplicity control | Lag h-1, Harvey-Leybourne-Newbold correction, Student-t(n-1), Holm-adjusted p-values |
| Offline market cache re-derived Log_Return/Volatility from an already-truncated cache, losing 14 more rows on every `USE_MARKET_API=0` run (and re-saving the shrunken file) | Cached derived columns are kept; the row count is stable across repeated offline runs |
| Windowing note: the 15% validation tail is time-ordered, and test data are never used during fitting (checked) | unchanged |

`python tests/test_fixes.py` checks the naive-target fix, label purging, KRP, DM properties and the cache idempotence without training any model.

---

## ⚡ Performance & Reproducibility Controls

Recent additions for runtime and reproducibility, all opt-in via environment variables (defaults preserve original behavior):

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `CRYPTO_HORSERACE_BASELINE_JOBS` | `cpu_count - 1` | Worker processes for the per-day ARIMA/GARCH/GJR-GARCH rolling refits (`src/baselines.py`). Each day's refit is independent, so this only changes wall-clock time, never the numbers — verified to reproduce identical RMSE at `n_jobs=1` vs parallel. |
| `CRYPTO_HORSERACE_FAST_DEV` | off | Skips the ablation suite, rolling-origin evaluation and cross-asset transfer for fast iteration while debugging. Leave unset for a full paper-grade run. |
| `CRYPTO_HORSERACE_PROFILE` | `standard` | `smoke` (minutes; verifies every code path, numbers are not results), `lite` (one seed; closest to the original cost), `standard` (3-seed ensembles, 3-seed ablations), `full` (5 seeds). Defined in `config.py`. |
| `CRYPTO_HORSERACE_TICKERS`, `CRYPTO_HORSERACE_HORIZONS` | all | Restrict the universe, e.g. `BTC-USD,ETH-USD` and `1,3` (used by the smoke test). |
| `CRYPTO_HORSERACE_RESULTS_DIR` | `results/` | Write outputs elsewhere, so a smoke run never overwrites real results. |
| `CRYPTO_HORSERACE_FORECAST_CACHE`, `CRYPTO_HORSERACE_CACHE_DIR` | on, `cache/` | Rolling-ARIMA forecasts are cached by a hash of the series, horizon and order; the baseline and the linear stage of AHI share them. |
| `CRYPTO_HORSERACE_GARCH_ESTIMAND` | `window` | `daily` restores the legacy one-day conditional-std forecast. |
| `CRYPTO_HORSERACE_USE_MARKET_API` | `1` | Set to `0` to skip the yfinance network call and reuse the local market data cache (safe for a fixed historical `START_DATE`/`END_DATE`). |
| `CRYPTO_HORSERACE_USE_API` | `1` | Same idea for Pillars 2/3 (sentiment, on-chain) — set to `0` to reuse each pillar's previously-saved feature CSV instead of re-hitting CryptoPanic/Google Trends/Fear&Greed/CoinMetrics. |

The pipeline also auto-drops any feature column that is constant (e.g. `FinBERT_Polarity` when `ENABLE_FINBERT_ON_INGEST` is off and no text corpus is supplied) the same way it already dropped all-NaN columns — a constant column is pushed through `MinMaxScaler` for no information gain, so it's logged and excluded rather than silently wasting an input channel.

---



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
│   ├── 📊 build_paper_inputs.py     # results/ -> paper_inputs/ (every manuscript table)
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
│   ├── 🔬 interpret.py              # attention profile, permutation importance, conformal, perturbations
│   ├── 🧪 protocol.py               # seed ensembles, rolling-origin evaluation, cross-asset transfer
│   ├── 🧠 model_cnn_lstm.py         # CNN-BiLSTM-Attention Keras architecture
│   ├── 🧬 model_dl_baselines.py     # CNN-only, LSTM-only, GRU-only baselines
│   └── 🤖 sentiment_finbert.py      # HuggingFace FinBERT sentiment classifier
│
├── 📂 data/                         # Persistent datasets (created at runtime)
│   ├── 📂 raw/                      # Unprocessed API responses & CSVs
│   ├── 📂 features/                 # Cleaned per-pillar feature tables
│   └── 📂 processed/                # Fully fused multimodal panels
│
├── 📂 docs/
│   └── 📂 figures/                  # Backup copies of the figures embedded below (README images actually load from results/, since that's regenerated on every run — see note under Results)
│
└── 📂 results/                      # Output figures, tables & benchmark reports
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Al3nzy/Crypto-Volatility-Horserace.git
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

> Runtime/reproducibility environment variables (`CRYPTO_HORSERACE_BASELINE_JOBS`, `CRYPTO_HORSERACE_FAST_DEV`, `CRYPTO_HORSERACE_USE_MARKET_API`, `CRYPTO_HORSERACE_USE_API`) are documented in [Performance & Reproducibility Controls](#-performance--reproducibility-controls) above.

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
