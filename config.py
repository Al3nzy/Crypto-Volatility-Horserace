"""
Global configuration for the Crypto Volatility Horserace project.
"""
import os

# Silence TensorFlow's C++ log noise (the "NodeDef mentions attribute
# use_unbounded_threadpool" ERROR lines are a harmless version-mismatch notice
# that is printed once per dataset iterator). Must be set before TensorFlow is
# imported; config.py is imported first everywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "3")
import multiprocessing as _mp

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.environ.get("CRYPTO_HORSERACE_RESULTS_DIR") or os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_MARKET_DIR = os.path.join(DATA_RAW_DIR, "market")
RAW_SENTIMENT_DIR = os.path.join(DATA_RAW_DIR, "sentiment")
RAW_ONCHAIN_DIR = os.path.join(DATA_RAW_DIR, "onchain")

FEATURE_MARKET_DIR = os.path.join(DATA_FEATURES_DIR, "market")
FEATURE_SENTIMENT_DIR = os.path.join(DATA_FEATURES_DIR, "sentiment")
FEATURE_ONCHAIN_DIR = os.path.join(DATA_FEATURES_DIR, "onchain")

MARKET_DATA_FILE = os.path.join(DATA_RAW_DIR, "market_data.csv")
SENTIMENT_DATA_FILE = os.path.join(DATA_RAW_DIR, "sentiment_data.csv")
ONCHAIN_DATA_FILE = os.path.join(DATA_RAW_DIR, "onchain_data.csv")

RAW_MARKET_FILE = os.path.join(RAW_MARKET_DIR, "market_data.csv")
FEATURE_MARKET_FILE = os.path.join(FEATURE_MARKET_DIR, "market_data.csv")

RAW_SENTIMENT_LEGACY_FILE = os.path.join(DATA_RAW_DIR, "sentiment_raw.csv")
RAW_SENTIMENT_ARCHIVE_DIR = os.path.join(DATA_RAW_DIR, "sentiment_raw")
RAW_SENTIMENT_CRYPTOPANIC_FILE = os.path.join(RAW_SENTIMENT_DIR, "cryptopanic_posts.csv")
FEATURE_SENTIMENT_FILE = os.path.join(FEATURE_SENTIMENT_DIR, "sentiment_data.csv")

RAW_ONCHAIN_LEGACY_FILE = os.path.join(DATA_RAW_DIR, "onchain_raw.csv")
RAW_ONCHAIN_LEGACY_DIR = os.path.join(DATA_RAW_DIR, "onchain_raw")
RAW_ONCHAIN_COINMETRICS_FILE = os.path.join(RAW_ONCHAIN_DIR, "coinmetrics_community.csv")
FEATURE_ONCHAIN_FILE = os.path.join(FEATURE_ONCHAIN_DIR, "onchain_data.csv")

for d in [
    DATA_RAW_DIR,
    DATA_FEATURES_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    RAW_MARKET_DIR,
    RAW_SENTIMENT_DIR,
    RAW_ONCHAIN_DIR,
    FEATURE_MARKET_DIR,
    FEATURE_SENTIMENT_DIR,
    FEATURE_ONCHAIN_DIR,
]:
    os.makedirs(d, exist_ok=True)

RAW_SENTIMENT_CORPUS_DIR = os.path.join(RAW_SENTIMENT_DIR, "corpus")
FEATURE_SENTIMENT_FINBERT_FILE = os.path.join(FEATURE_SENTIMENT_DIR, "finbert_daily.csv")
MARKET_CMC_VALIDATION_FILE = os.path.join(RESULTS_DIR, "market_cmc_validation.csv")
os.makedirs(RAW_SENTIMENT_CORPUS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Time range (defaults; may be overridden by data experiment preset)
# ──────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# ──────────────────────────────────────────────────────────────
# Asset universe / market frequency controls
# ──────────────────────────────────────────────────────────────
CORE_TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "DOGE-USD",
    "XRP-USD",
]
OPTIONAL_TICKERS = [
    "SOL-USD",
    "SHIB-USD",
]
TICKERS = list(CORE_TICKERS)
_env_t = os.environ.get("CRYPTO_HORSERACE_TICKERS", "").strip()
if _env_t:
    TICKERS = [t.strip() for t in _env_t.split(",") if t.strip()]
PRIMARY_TICKER = TICKERS[0] if (_env_t and "BTC-USD" not in TICKERS) else "BTC-USD"
MARKET_INTERVAL = "1d"          # e.g. 1d, 1h, 4h
INTRADAY_PERIOD = "730d"        # used only for intraday intervals

ASSET_METADATA = {
    "BTC-USD": {
        "symbol": "BTC",
        "name": "bitcoin",
        "cryptopanic_currency": "BTC",
        "coinmetrics_asset": "btc",
        "glassnode_asset": "BTC",
        "coinmarketcap_id": 1,
        "trend_terms": ["Bitcoin", "BTC"],
        "aliases": ["BTC", "bitcoin", "btc-usd"],
    },
    "ETH-USD": {
        "symbol": "ETH",
        "name": "ethereum",
        "cryptopanic_currency": "ETH",
        "coinmetrics_asset": "eth",
        "glassnode_asset": "ETH",
        "coinmarketcap_id": 1027,
        "trend_terms": ["Ethereum", "ETH"],
        "aliases": ["ETH", "ethereum", "eth-usd"],
    },
    "DOGE-USD": {
        "symbol": "DOGE",
        "name": "dogecoin",
        "cryptopanic_currency": "DOGE",
        "coinmetrics_asset": "doge",
        "glassnode_asset": "DOGE",
        "coinmarketcap_id": 74,
        "trend_terms": ["Dogecoin", "DOGE"],
        "aliases": ["DOGE", "dogecoin", "doge-usd"],
    },
    "XRP-USD": {
        "symbol": "XRP",
        "name": "ripple",
        "cryptopanic_currency": "XRP",
        "coinmetrics_asset": "xrp",
        "glassnode_asset": "XRP",
        "coinmarketcap_id": 52,
        "trend_terms": ["XRP", "Ripple"],
        "aliases": ["XRP", "ripple", "xrp-usd"],
    },
    "SOL-USD": {
        "symbol": "SOL",
        "name": "solana",
        "cryptopanic_currency": "SOL",
        "coinmetrics_asset": "sol",
        "glassnode_asset": "SOL",
        "coinmarketcap_id": 5426,
        "trend_terms": ["Solana", "SOL"],
        "aliases": ["SOL", "solana", "sol-usd"],
    },
    "SHIB-USD": {
        "symbol": "SHIB",
        "name": "shiba inu",
        "cryptopanic_currency": "SHIB",
        "coinmetrics_asset": "shib",
        "glassnode_asset": None,
        "coinmarketcap_id": 5994,
        "trend_terms": ["Shiba Inu", "SHIB coin"],
        "aliases": ["SHIB", "shiba inu", "shib-usd"],
    },
}

# Data scale matrix: set env CRYPTO_HORSERACE_DATA_EXPERIMENT to a preset name.
_DATA_EXPERIMENT_PRESETS: dict[str, dict] = {
    "default": {},
    "extended_calendar": {"END_DATE": "2025-12-31"},
    "hourly_btc_focus": {
        "MARKET_INTERVAL": "1h",
        "INTRADAY_PERIOD": "730d",
        "tickers": ["BTC-USD"],
    },
    "six_assets": {"include_optional_tickers": True},
}
_ACTIVE_DATA_EXPERIMENT = (os.environ.get("CRYPTO_HORSERACE_DATA_EXPERIMENT") or "default").strip()
_EXPERIMENT_LAYER = _DATA_EXPERIMENT_PRESETS.get(
    _ACTIVE_DATA_EXPERIMENT, _DATA_EXPERIMENT_PRESETS["default"]
)
if not isinstance(_EXPERIMENT_LAYER, dict):
    _EXPERIMENT_LAYER = {}
if "END_DATE" in _EXPERIMENT_LAYER:
    END_DATE = str(_EXPERIMENT_LAYER["END_DATE"])
if "START_DATE" in _EXPERIMENT_LAYER:
    START_DATE = str(_EXPERIMENT_LAYER["START_DATE"])
if "MARKET_INTERVAL" in _EXPERIMENT_LAYER:
    MARKET_INTERVAL = str(_EXPERIMENT_LAYER["MARKET_INTERVAL"])
if "INTRADAY_PERIOD" in _EXPERIMENT_LAYER:
    INTRADAY_PERIOD = str(_EXPERIMENT_LAYER["INTRADAY_PERIOD"])
if _EXPERIMENT_LAYER.get("include_optional_tickers"):
    TICKERS = list(CORE_TICKERS) + list(OPTIONAL_TICKERS)
elif _EXPERIMENT_LAYER.get("tickers"):
    TICKERS = list(_EXPERIMENT_LAYER["tickers"])
ACTIVE_DATA_EXPERIMENT = _ACTIVE_DATA_EXPERIMENT

COINMARKETCAP_API_KEY = os.environ.get("COINMARKETCAP_API_KEY", "").strip()
ENABLE_COINMARKETCAP = bool(COINMARKETCAP_API_KEY)
COINMETRICS_API_KEY = os.environ.get("COINMETRICS_API_KEY", "").strip()

MARKET_YF_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume", "Log_Return",
]
MARKET_CMC_FEATURE_COLS = ["CMC_Close", "CMC_Volume"]
MARKET_FEATURE_COLS = MARKET_YF_FEATURE_COLS + (
    MARKET_CMC_FEATURE_COLS if ENABLE_COINMARKETCAP else []
)

ASSET_SENTIMENT_FEATURE_COLS = [
    "Market_FearGreed",
    "FinBERT_Polarity",
]
ASSET_SENTIMENT_PREFLIGHT_REQUIRED_COLS = ["Market_FearGreed"]
ASSET_SENTIMENT_PREFLIGHT_OPTIONAL_COLS = ["FinBERT_Polarity"]

ASSET_ONCHAIN_FEATURE_COLS = [
    "Active_Addresses",
    "Tx_Count",
    "Transfer_Volume_USD",
    "Hash_Rate",
    "Network_Cap_USD",
]
ASSET_ONCHAIN_CORE_FEATURE_COLS = [
    "Active_Addresses",
    "Tx_Count",
]
ASSET_ONCHAIN_OPTIONAL_FEATURE_COLS = [
    "Transfer_Volume_USD",
    "Hash_Rate",
    "Network_Cap_USD",
]
# ── Feature set ───────────────────────────────────────────────
# "raw"        : price/volume LEVELS (Open, High, Low, Close, Volume) and raw
#                on-chain counts, as in the earlier runs. These series are
#                non-stationary: in a hold-out year the levels leave the range
#                seen in training (BTC 2024 closes exceed the 2020-2023 maximum),
#                a network can use them as a clock, and permutation importance
#                showed the standalone network relying on Close/High/Volume.
# "stationary" : scale-free transforms of the same information (default).
FEATURE_SET = (os.environ.get("CRYPTO_HORSERACE_FEATURE_SET") or "stationary").strip().lower()
if FEATURE_SET not in ("raw", "stationary"):
    raise ValueError("CRYPTO_HORSERACE_FEATURE_SET must be raw or stationary")

MARKET_STATIONARY_COLS = ["Log_Return", "Abs_Return", "Range_HL", "LogVolume_Change"]
ONCHAIN_STATIONARY_COLS = ["Active_Addresses_LogRatio", "Tx_Count_LogRatio"]   # ln(x_t / 30-day mean)
MARKET_RAW_COLS = list(MARKET_FEATURE_COLS)
ONCHAIN_RAW_COLS = list(ASSET_ONCHAIN_CORE_FEATURE_COLS)

if FEATURE_SET == "stationary":
    _MARKET_G, _ONCHAIN_G = MARKET_STATIONARY_COLS, ONCHAIN_STATIONARY_COLS
else:
    _MARKET_G, _ONCHAIN_G = MARKET_RAW_COLS, ONCHAIN_RAW_COLS

FULL_FEATURES_NO_RV = (
    list(_MARKET_G)
    + ASSET_SENTIMENT_FEATURE_COLS
    + list(_ONCHAIN_G)
)

# HAR-style trailing realized-volatility features. ARIMA/GARCH/HAR-RV all
# forecast future volatility primarily from *past observed volatility*
# (that's what makes HAR-RV a same-day OLS on RV_d/RV_w/RV_m, and why ARIMA
# fits the raw Volatility series directly) -- crypto realized volatility is
# strongly persistent (volatility clustering), so lagged RV is typically the
# single most informative predictor of near-term RV. FULL_FEATURES_NO_RV
# above never gave the DL model that same signal; it had to infer volatility
# indirectly from OHLCV/returns/macro/on-chain features only. Concretely,
# in a real run this showed up as the DL model overshooting hard in the
# 2024 "calm" regime (regime-split RMSE ~0.029 vs ARIMA/HAR-RV's ~0.0023) --
# it had no way to anchor to the fact that realized vol had genuinely
# dropped, the way ARIMA/HAR-RV do automatically via their own lagged input.
# RV_Week/RV_Month are computed in src/fuse_data.py with the exact same
# trailing window alignment as src/baselines.py's run_har_rv(), so adding
# them is leakage-free by construction: RV_Week[t0] only uses Volatility up
# to and including t0, matching create_windows()'s own window boundary.
RV_FEATURE_COLS = ["Volatility", "RV_Week", "RV_Month"]
FULL_FEATURES = FULL_FEATURES_NO_RV + RV_FEATURE_COLS
# Pillar groups for the ablation and the importance analyses (follow FEATURE_SET)
FEATURE_GROUPS = {
    "market": list(_MARKET_G),
    "sentiment": list(ASSET_SENTIMENT_FEATURE_COLS),
    "onchain": list(_ONCHAIN_G),
    "lagged_vol": list(RV_FEATURE_COLS),
}
EXTENDED_FEATURES = (
    MARKET_FEATURE_COLS
    + ASSET_SENTIMENT_FEATURE_COLS
    + ASSET_ONCHAIN_FEATURE_COLS
)

ENABLE_FINBERT_ON_INGEST = os.environ.get("ENABLE_FINBERT_ON_INGEST", "").lower() in (
    "1",
    "true",
    "yes",
)
FINBERT_MODEL_NAME = os.environ.get("FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_BATCH_SIZE = int(os.environ.get("FINBERT_BATCH_SIZE", "16"))

USE_POOLED_DL_TRAINING = os.environ.get("CRYPTO_HORSERACE_POOLED_DL", "").lower() in (
    "1",
    "true",
    "yes",
)

# ──────────────────────────────────────────────────────────────
# Volatility parameters
# ──────────────────────────────────────────────────────────────
VOLATILITY_WINDOW = 14          # rolling window (days) for realized vol
LOG_RETURNS = True              # use log-returns for vol calculation


# ──────────────────────────────────────────────────────────────
# Run profile
#   smoke    : minutes; verifies that every code path runs and every output
#              file is written. Numbers from this profile are NOT results.
#   lite     : one seed everywhere; closest in cost to the original single-seed run.
#   standard : paper-grade run; 3-seed ensembles for the standalone network, AHI and AHI-K,
#              single-seed CNN/LSTM/GRU baselines, 3-seed ablations.
#   full     : 5-seed ensembles, 3-seed baselines, 5-seed ablations, 3-seed rolling ensembles.
# Select with CRYPTO_HORSERACE_PROFILE=smoke|lite|standard|full. A smoke run can
# also restrict the universe with CRYPTO_HORSERACE_TICKERS="BTC-USD,ETH-USD"
# and CRYPTO_HORSERACE_HORIZONS="1,3", and should use its own results
# directory (CRYPTO_HORSERACE_RESULTS_DIR) so real results are not overwritten.
# ──────────────────────────────────────────────────────────────
PROFILE = (os.environ.get("CRYPTO_HORSERACE_PROFILE") or "standard").strip().lower()
# Budget guide (CPU, ~2 min per 100-epoch fit; early stopping usually ends sooner):
#   smoke ~ 15 min for 2 assets x 1 horizon | lite ~ 2-3 h | standard ~ 3-5 h | full ~ 10 h+
_PROFILES = {
    "smoke": dict(EPOCHS=3, DL_SEEDS=[42, 7], AHI_SEEDS=[42], BASELINE_DL_SEEDS=[42], ABLATION_SEEDS=[42],
                  ABLATION_HORIZONS=[1, 3], ROLLING_EPOCHS=2, ROLLING_STEP_DAYS=240, ROLLING_SEEDS=[42],
                  ROLLING_INCLUDE_ARIMA_HYBRID=False, CROSS_ASSET_HORIZONS=[1, 3], CROSS_ASSET_SEEDS=[42],
                  MC_DROPOUT_PASSES=6, BOOTSTRAP_RESAMPLES=60, PERMUTATION_REPEATS=2),
    "lite": dict(EPOCHS=100, DL_SEEDS=[42], AHI_SEEDS=[42], BASELINE_DL_SEEDS=[42], ABLATION_SEEDS=[42],
                 ABLATION_HORIZONS=[1], ROLLING_EPOCHS=25, ROLLING_STEP_DAYS=240, ROLLING_SEEDS=[42],
                 ROLLING_INCLUDE_ARIMA_HYBRID=False, CROSS_ASSET_HORIZONS=[1], CROSS_ASSET_SEEDS=[42],
                 MC_DROPOUT_PASSES=30, BOOTSTRAP_RESAMPLES=500, PERMUTATION_REPEATS=5),
    "standard": dict(EPOCHS=100, DL_SEEDS=[42, 7], AHI_SEEDS=[42], BASELINE_DL_SEEDS=[42],
                     ABLATION_SEEDS=[42, 7], ABLATION_HORIZONS=[1], ROLLING_EPOCHS=30, ROLLING_STEP_DAYS=180,
                     ROLLING_SEEDS=[42], ROLLING_INCLUDE_ARIMA_HYBRID=False, CROSS_ASSET_HORIZONS=[1],
                     CROSS_ASSET_SEEDS=[42], MC_DROPOUT_PASSES=30, BOOTSTRAP_RESAMPLES=1000,
                     PERMUTATION_REPEATS=5),
    "full": dict(EPOCHS=100, DL_SEEDS=[42, 7, 21], AHI_SEEDS=[42, 7, 21], BASELINE_DL_SEEDS=[42, 7, 21],
                 ABLATION_SEEDS=[42, 7, 21], ABLATION_HORIZONS=[1, 3, 7], ROLLING_EPOCHS=40,
                 ROLLING_STEP_DAYS=120, ROLLING_SEEDS=[42, 7], ROLLING_INCLUDE_ARIMA_HYBRID=True,
                 CROSS_ASSET_HORIZONS=[1, 3, 7], CROSS_ASSET_SEEDS=[42, 7, 21], MC_DROPOUT_PASSES=50,
                 BOOTSTRAP_RESAMPLES=2000, PERMUTATION_REPEATS=10),
}
if PROFILE not in _PROFILES:
    raise ValueError(f"Unknown CRYPTO_HORSERACE_PROFILE={PROFILE!r}; use smoke|lite|standard|full")
_P = _PROFILES[PROFILE]

# ──────────────────────────────────────────────────────────────
# Windowing / sequencing
# ──────────────────────────────────────────────────────────────
LOOKBACK_WINDOW = 14            # days fed into the model
FORECAST_HORIZON = 1            # predict 1-day-ahead volatility
EXPERIMENT_HORIZONS = [1, 3, 7] # thesis experiments
_env_h = os.environ.get("CRYPTO_HORSERACE_HORIZONS", "").strip()
if _env_h:
    EXPERIMENT_HORIZONS = [int(x) for x in _env_h.split(",") if x.strip()]

# ──────────────────────────────────────────────────────────────
# Train / Test split
# ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8

# ──────────────────────────────────────────────────────────────
# Deep-learning hyper-parameters
# ──────────────────────────────────────────────────────────────
CNN_FILTERS = 32               # reduced to improve generalization
CNN_KERNEL_SIZE = 3
LSTM_UNITS = 32                # reduced to reduce overfitting
NUM_ATTENTION_HEADS = 4
ATTENTION_KEY_DIM = 16         # reduced with smaller LSTM
DROPOUT_RATE = 0.4             # increased regularization
L2_REG = 1e-4                  # L2 regularization on Conv/Dense
LEARNING_RATE = 5e-4           # lower for more stable training
BATCH_SIZE = 16                # smaller batch for noisier gradients
EPOCHS = _P["EPOCHS"]
PATIENCE = 15
EARLY_STOP_MIN_DELTA = float(os.environ.get("CRYPTO_HORSERACE_ES_MIN_DELTA", "1e-4"))   # val-loss improvement that counts
VALIDATION_SPLIT = 0.15        # last 15% of train for early stopping (no test leak)

# ──────────────────────────────────────────────────────────────
# Random seed
# ──────────────────────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────────────────────
# Experiment controls
# ──────────────────────────────────────────────────────────────
RUN_ABLATIONS = True
RUN_ABLATIONS_FOR_ALL_TICKERS = False  # if False, ablations run only for PRIMARY_TICKER (saves compute)
PAPER_MODE = True  # stop the pipeline if any paper-critical pillar uses synthetic data
ENABLE_GOOGLE_TRENDS = False
ENABLE_ONCHAIN_API = True
# Keep raw local exports as the default paper workflow.
# Optional paid vendors can be wired in later, but they are not assumed.
ONCHAIN_PROVIDER = "coinmetrics_community"
MIN_PREFLIGHT_COVERAGE = 0.90
MAX_PREFLIGHT_MISSING_RATE = 0.10
DATA_SOURCE_REPORT_FILE = os.path.join(RESULTS_DIR, "data_source_report.csv")
DATA_QUALITY_REPORT_FILE = os.path.join(RESULTS_DIR, "data_quality_report.csv")
DATA_PREFLIGHT_JSON_FILE = os.path.join(RESULTS_DIR, "data_preflight_report.json")

# Named windows for stress-regime evaluation
SHOCK_WINDOWS = [
    ("2022-05-05", "2022-05-20", "LUNA"),
    ("2022-11-05", "2022-11-20", "FTX"),
]

# Persistent run log
RUN_HISTORY_FILE = os.path.join(RESULTS_DIR, "run_history.csv")

# Advanced experiment controls
WALKFORWARD_STEP = 30
WALKFORWARD_MIN_TRAIN = 240
WALKFORWARD_EPOCHS = 20
TRANSACTION_COST_BPS = 10
MC_DROPOUT_PASSES = _P["MC_DROPOUT_PASSES"]
MC_INTERVAL_ALPHA = 0.10
SENSITIVITY_MISSING_RATES = [0.10, 0.20]
SENSITIVITY_NOISE_STD_RATES = [0.10, 0.20]
REPRO_SEEDS = [7, 21, 42, 84, 126]   # legacy name; the seed ensemble is DL_SEEDS below

# ──────────────────────────────────────────────────────────────
# Evaluation-protocol controls (all documented in README.md)
# ──────────────────────────────────────────────────────────────
# Every neural model is trained once per seed; its reported forecast is the
# average over seeds (a seed ensemble) and the per-seed spread is saved to
# seed_metrics_*.csv. Attention/permutation/MC-dropout/perturbation analyses
# use the first seed's network.
DL_SEEDS = list(_P["DL_SEEDS"])                     # standalone network and AHI-K
AHI_SEEDS = list(_P["AHI_SEEDS"])                   # AHI (ARIMA stage)
BASELINE_DL_SEEDS = list(_P["BASELINE_DL_SEEDS"])   # CNN-only / LSTM-only / GRU-only
ABLATION_HORIZONS = list(_P["ABLATION_HORIZONS"])
CROSS_ASSET_SEEDS = list(_P["CROSS_ASSET_SEEDS"])
ROLLING_INCLUDE_ARIMA_HYBRID = bool(_P["ROLLING_INCLUDE_ARIMA_HYBRID"])
TRAIN_VERBOSE = int(os.environ.get("CRYPTO_HORSERACE_TRAIN_VERBOSE", "1"))   # 1 = Keras progress bar, 0 = silent
# Tasks (one per asset-horizon combo, ablation, rolling, transfer) run in separate worker processes:
# TensorFlow/Keras leaks ~45 MB per trained model inside one process, which made a long single-process
# run grow past 40 GB. WORKERS=1 runs tasks one after another with the progress bars on the console.
WORKERS = int(os.environ.get("CRYPTO_HORSERACE_WORKERS", "1"))
RESUME = os.environ.get("CRYPTO_HORSERACE_RESUME", "1").lower() not in ("0", "false", "no")
ABLATION_SEEDS = list(_P["ABLATION_SEEDS"])

# Label purging: with horizon h the target of training window k lies h-1 days
# after the origin of the next window, so the last h-1 training windows carry
# labels realized after the first test origin. They are dropped when True.
PURGE_TRAIN_OVERLAP = True

# Disk cache for the rolling ARIMA forecasts (deterministic given the series,
# horizon and order; the key is a hash of those, so a changed series never
# reuses a stale file). Set CRYPTO_HORSERACE_FORECAST_CACHE=0 to disable.
CACHE_DIR = os.environ.get("CRYPTO_HORSERACE_CACHE_DIR") or os.path.join(BASE_DIR, "cache")
USE_FORECAST_CACHE = os.environ.get("CRYPTO_HORSERACE_FORECAST_CACHE", "1").lower() not in ("0", "false", "no")

# GARCH estimand. "window": forecast the 14-day rolling volatility that the
# target actually is (known returns inside the window + forecast variances
# for the remaining days). "daily": legacy one-day conditional std.
GARCH_ESTIMAND = (os.environ.get("CRYPTO_HORSERACE_GARCH_ESTIMAND") or "window").strip().lower()

# Diebold-Mariano: HAC truncation lag h-1 (textbook for h-step forecasts) with
# the Harvey-Leybourne-Newbold small-sample correction; Holm-adjusted p-values
# are reported next to the raw ones.
DM_HLN_CORRECTION = True

# Cross-asset transfer: every asset is scaled with its own training-period
# statistics before pooling, at every horizon.
CROSS_ASSET_HORIZONS = list(_P["CROSS_ASSET_HORIZONS"])

# Rolling-origin evaluation that spans the Terra-LUNA and FTX episodes. The
# networks are re-fitted every ROLLING_STEP_DAYS days on all earlier windows
# (scalers re-fitted per fold, labels purged). Baselines are re-estimated at
# every fold (HAR-RV) or every origin (ARIMA, GARCH).
RUN_ROLLING_ORIGIN = True
ROLLING_START = "2022-01-01"
ROLLING_STEP_DAYS = _P["ROLLING_STEP_DAYS"]
ROLLING_EPOCHS = _P["ROLLING_EPOCHS"]
ROLLING_SEEDS = list(_P["ROLLING_SEEDS"])

# Uncertainty quantification of the point forecasts
BOOTSTRAP_RESAMPLES = _P["BOOTSTRAP_RESAMPLES"]   # moving-block bootstrap
BOOTSTRAP_BLOCK_MIN = 10
PERMUTATION_REPEATS = _P["PERMUTATION_REPEATS"]

# Q1 horserace: optional baselines (disable to shorten runs)
RUN_SVR_BASELINE = os.environ.get("RUN_SVR_BASELINE", "1").lower() not in ("0", "false", "no")
RUN_DL_ABLATION_BASELINES = os.environ.get("RUN_DL_ABLATION_BASELINES", "1").lower() not in (
    "0",
    "false",
    "no",
)
RUN_HAR_GJR_BASELINES = os.environ.get("RUN_HAR_GJR_BASELINES", "1").lower() not in (
    "0",
    "false",
    "no",
)
RUN_NAIVE_PERSISTENCE_BASELINE = os.environ.get("RUN_NAIVE_PERSISTENCE_BASELINE", "1").lower() not in (
    "0",
    "false",
    "no",
)

# SVR hyperparameters (multimodal flattened windows)
SVR_C = float(os.environ.get("SVR_C", "10.0"))
_svr_g = os.environ.get("SVR_GAMMA", "scale").strip()
try:
    SVR_GAMMA: float | str = float(_svr_g)
except ValueError:
    SVR_GAMMA = _svr_g
SVR_EPSILON = float(os.environ.get("SVR_EPSILON", "0.01"))

# ──────────────────────────────────────────────────────────────
# Performance controls
# ──────────────────────────────────────────────────────────────
# Worker processes for the per-day ARIMA/GARCH/GJR-GARCH rolling refits
# (src/baselines.py). Each day's refit is independent of every other day's,
# so running them across cores changes only wall-clock time, never the
# numbers. n_jobs=1 reproduces the original strictly-sequential behaviour.
BASELINE_N_JOBS = int(
    os.environ.get("CRYPTO_HORSERACE_BASELINE_JOBS", str(max(1, (_mp.cpu_count() or 2) - 1)))
)

# When True, skip the most compute-heavy *optional* robustness analyses
# (walk-forward re-estimation, the 7-way ablation feature-set suite, the
# multi-seed reproducibility matrix, and cross-asset generalization) so a
# debugging run finishes in minutes instead of hours. These analyses still
# run in full whenever this is False (the default), so the paper-grade
# results are unaffected unless this is explicitly turned on.
FAST_DEV_MODE = os.environ.get("CRYPTO_HORSERACE_FAST_DEV", "").lower() in ("1", "true", "yes")

# When False, Pillar 1 (market OHLCV) skips the yfinance network call and
# reads directly from the local cache in data/raw or data/features, mirroring
# the use_api switch already used for the sentiment/on-chain pillars. Useful
# for fast repeat runs once the historical window has already been fetched
# once (START_DATE/END_DATE are fixed historical dates, so the data will not
# change between runs).
USE_MARKET_API = os.environ.get("CRYPTO_HORSERACE_USE_MARKET_API", "1").lower() not in (
    "0",
    "false",
    "no",
)

# Same idea as USE_MARKET_API but for Pillars 2/3 (sentiment, on-chain).
# main.py wires this into collect_feature_datasets(use_api=...); when False,
# load_sentiment_data()/load_onchain_data() reuse each pillar's own
# previously-saved feature CSV (data/features/.../*.csv) instead of hitting
# CryptoPanic/Google Trends/Fear&Greed/CoinMetrics again.
USE_API = os.environ.get("CRYPTO_HORSERACE_USE_API", "1").lower() not in ("0", "false", "no")
