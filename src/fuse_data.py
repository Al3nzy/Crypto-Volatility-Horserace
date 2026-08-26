"""
Data Fusion & Preprocessing Module (Section 3.1).

Joins the three pillars on Date and Asset, handles missing values, scales
features, and generates time-series lookback windows for modelling.
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_PROCESSED_DIR,
    FEATURE_MARKET_FILE,
    FEATURE_ONCHAIN_FILE,
    FEATURE_SENTIMENT_FILE,
    FORECAST_HORIZON,
    FULL_FEATURES,
    LOOKBACK_WINDOW,
    PRIMARY_TICKER,
    TICKERS,
    TRAIN_RATIO,
)
from src.asset_utils import ticker_suffix
from src.collection_utils import normalize_dates


def _load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = normalize_dates(df["Date"])
    if "Asset" not in df.columns:
        df["Asset"] = PRIMARY_TICKER
    return df


def _prepare_asset_panel(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    out["Asset"] = out["Asset"].astype(str)
    out = out[out["Asset"] == ticker].copy()
    return out


def _effective_feature_cols(merged: pd.DataFrame, requested: list[str]) -> list[str]:
    """
    Drop requested columns that are missing, entirely NaN (e.g. a failed
    vendor fetch), or constant (e.g. FinBERT_Polarity sitting at its 0.0
    placeholder because ENABLE_FINBERT_ON_INGEST is off and no text corpus
    was supplied, or Market_FearGreed falling back to a fixed value when the
    Fear&Greed API is unreachable). A constant column carries no information
    for the model but still occupies an input channel and gets pushed
    through MinMaxScaler, so it's dropped the same way an all-NaN column
    already was.
    """
    out = []
    for c in requested:
        if c not in merged.columns:
            continue
        col = merged[c]
        if col.notna().sum() == 0:
            print(f"[Fusion] Omitting all-NaN feature column: {c}")
            continue
        if col.dropna().nunique() <= 1:
            print(f"[Fusion] Omitting constant feature column (no information): {c}")
            continue
        out.append(c)
    return out


# In-memory cache of the fused per-ticker panel. fuse_pillars() joins the
# three pillar CSVs and does not depend on feature_cols, forecast_horizon, or
# residual_target (those are applied downstream in run_fusion_pipeline), so
# for a given ticker its output is identical every time it's called within
# the same process. Phase 1 collection writes the pillar CSVs once at the
# start of main() before any fuse_pillars() call, so caching for the
# lifetime of the process is safe. This avoids re-reading, re-merging, and
# re-writing the same panel dozens of times across horizons, ablation
# feature sets, and cross-asset generalization.
_FUSE_CACHE: dict[str, pd.DataFrame] = {}
_WRITTEN_MERGED: set[str] = set()


def fuse_pillars(primary_ticker: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Join market, asset-specific sentiment, and asset-specific on-chain features.
    The merge key is (Date, Asset).
    """
    ticker = primary_ticker if primary_ticker is not None else PRIMARY_TICKER

    if use_cache and ticker in _FUSE_CACHE:
        return _FUSE_CACHE[ticker].copy()

    market = _prepare_asset_panel(_load_panel(FEATURE_MARKET_FILE), ticker)
    sentiment = _prepare_asset_panel(_load_panel(FEATURE_SENTIMENT_FILE), ticker)
    onchain = _prepare_asset_panel(_load_panel(FEATURE_ONCHAIN_FILE), ticker)

    if market.empty:
        raise ValueError(f"No market data found for ticker {ticker}.")

    merged = market.merge(sentiment, on=["Date", "Asset"], how="left")
    merged = merged.merge(onchain, on=["Date", "Asset"], how="left")
    merged.sort_values(["Asset", "Date"], inplace=True)

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    merged[numeric_cols] = merged.groupby("Asset")[numeric_cols].ffill().bfill()

    # HAR-style trailing realized-volatility features (see RV_FEATURE_COLS
    # in config.py for the full rationale). Computed per-asset with a
    # trailing window ending at the same row t0 the model conditions on,
    # exactly mirroring src/baselines.py's run_har_rv(): RV_Week[t0] and
    # RV_Month[t0] only use Volatility up to and including t0, so this
    # carries no look-ahead relative to create_windows()'s window boundary.
    if "Volatility" in merged.columns:
        merged["RV_Week"] = merged.groupby("Asset")["Volatility"].transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
        merged["RV_Month"] = merged.groupby("Asset")["Volatility"].transform(
            lambda s: s.rolling(22, min_periods=1).mean()
        )

    required_cols = [c for c in FULL_FEATURES + ["Volatility"] if c in merged.columns]
    merged.dropna(subset=required_cols, inplace=True)
    merged.set_index("Date", inplace=True)

    print(f"[Fusion] {ticker} merged shape: {merged.shape}")
    if use_cache:
        _FUSE_CACHE[ticker] = merged
        return merged.copy()
    return merged


FEATURE_COLS = FULL_FEATURES
TARGET_COL = "Volatility"


def scale_window_splits(X_train, X_test, y_train, y_test):
    """
    Fit scalers on TRAIN only (no look-ahead leakage), then transform both
    train and test windows.
    """
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    n_features = X_train.shape[-1]

    # Fit feature scaler on flattened train windows only
    X_train_flat = X_train.reshape(-1, n_features)
    feature_scaler.fit(X_train_flat)

    # Transform train/test and restore original 3D shape
    X_train_scaled = feature_scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape)

    # Fit target scaler on train targets only
    target_scaler.fit(y_train)
    y_train_scaled = target_scaler.transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler


# ──────────────────────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────────────────────
def create_windows(X: np.ndarray, y: np.ndarray, forecast_horizon: int = FORECAST_HORIZON):
    """
    Build (lookback_window, num_features) windows and corresponding targets.
    Target = volatility at time t + forecast_horizon.
    """
    Xs, ys = [], []
    for i in range(len(X) - LOOKBACK_WINDOW - forecast_horizon + 1):
        Xs.append(X[i: i + LOOKBACK_WINDOW])
        ys.append(y[i + LOOKBACK_WINDOW + forecast_horizon - 1])
    return np.array(Xs), np.array(ys)


# ──────────────────────────────────────────────────────────────
# Train / test split (time-ordered, no shuffle)
# ──────────────────────────────────────────────────────────────
def split_data(X_win, y_win):
    split_idx = int(len(X_win) * TRAIN_RATIO)
    X_train, X_test = X_win[:split_idx], X_win[split_idx:]
    y_train, y_test = y_win[:split_idx], y_win[split_idx:]
    print(f"[Split] Train {X_train.shape[0]} | Test {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────
def run_fusion_pipeline(
    feature_cols=None,
    target_col=TARGET_COL,
    residual_target=None,
    primary_ticker: str | None = None,
    forecast_horizon: int = FORECAST_HORIZON,
):
    """
    End-to-end: load → fuse → scale → window → split.
    If residual_target is provided (1D array, length = len(merged)), use it as y instead of target_col.
    primary_ticker: which asset to use when market has multiple. Default from config.
    """
    merged = fuse_pillars(primary_ticker=primary_ticker)

    # Save merged CSV for reference (only once per ticker per process; the
    # fused panel doesn't change across horizons/feature-set calls, so
    # rewriting it dozens of times per run was pure redundant I/O).
    suffix = ticker_suffix(primary_ticker or PRIMARY_TICKER)
    merged_path = os.path.join(DATA_PROCESSED_DIR, f"merged_dataset_{suffix}.csv")
    if suffix not in _WRITTEN_MERGED:
        merged.to_csv(merged_path)
        print(f"[Fusion] Saved merged dataset -> {merged_path}")
        _WRITTEN_MERGED.add(suffix)

    # Build windows from raw values first, split by time, then scale using
    # train-only fit to avoid look-ahead bias.
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    feature_cols = _effective_feature_cols(merged, list(feature_cols))
    if not feature_cols:
        raise ValueError("No usable feature columns after dropping all-NaN pillars.")

    X_raw = merged[feature_cols].values
    if residual_target is not None:
        y_raw = np.asarray(residual_target, dtype=float).reshape(-1, 1)
    else:
        y_raw = merged[[target_col]].values

    X_win, y_win = create_windows(X_raw, y_raw, forecast_horizon=forecast_horizon)
    X_train, X_test, y_train, y_test = split_data(X_win, y_win)

    X_train, X_test, y_train, y_test, feat_scaler, tgt_scaler = scale_window_splits(
        X_train, X_test, y_train, y_test
    )

    # Also keep the raw merged df date index for plotting later
    usable_dates = merged.index[LOOKBACK_WINDOW + forecast_horizon - 1:]
    split_idx = int(len(X_win) * TRAIN_RATIO)
    test_dates = usable_dates[split_idx:]

    return {
        "merged_df": merged,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feat_scaler": feat_scaler,
        "tgt_scaler": tgt_scaler,
        "test_dates": test_dates,
        "feature_cols": feature_cols,
    }


def run_fusion_pipeline_pooled(
    feature_cols=None,
    target_col=TARGET_COL,
    residual_target=None,
    forecast_horizon: int = FORECAST_HORIZON,
):
    """
    Stack train/test windows from every ticker in TICKERS (each asset keeps its
    own time-ordered split), then fit scalers on the pooled training batch.
    Increases DL sample count without changing per-asset date logic.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_train_parts, X_test_parts = [], []
    y_train_parts, y_test_parts = [], []
    test_dates_parts = []
    merged_ref = None
    effective_cols = None

    for ticker in TICKERS:
        merged = fuse_pillars(primary_ticker=ticker)
        merged_ref = merged if merged_ref is None else merged_ref
        cols = _effective_feature_cols(merged, list(feature_cols))
        if effective_cols is None:
            effective_cols = cols
        elif cols != effective_cols:
            raise ValueError(
                f"Pooled fusion: feature set mismatch for {ticker}: {cols} vs {effective_cols}"
            )

        X_raw = merged[effective_cols].values
        if residual_target is not None:
            raise ValueError("Pooled fusion does not support residual_target.")
        y_raw = merged[[target_col]].values

        X_win, y_win = create_windows(X_raw, y_raw, forecast_horizon=forecast_horizon)
        X_tr, X_te, y_tr, y_te = split_data(X_win, y_win)
        X_train_parts.append(X_tr)
        X_test_parts.append(X_te)
        y_train_parts.append(y_tr)
        y_test_parts.append(y_te)

        usable_dates = merged.index[LOOKBACK_WINDOW + forecast_horizon - 1:]
        split_idx = int(len(X_win) * TRAIN_RATIO)
        test_dates_parts.append(usable_dates[split_idx:])

    X_train = np.concatenate(X_train_parts, axis=0)
    X_test = np.concatenate(X_test_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    X_train, X_test, y_train, y_test, feat_scaler, tgt_scaler = scale_window_splits(
        X_train, X_test, y_train, y_test
    )

    flat_dates: list = []
    for d in test_dates_parts:
        flat_dates.extend(pd.DatetimeIndex(d).tolist())
    test_dates = pd.DatetimeIndex(flat_dates)

    return {
        "merged_df": merged_ref,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feat_scaler": feat_scaler,
        "tgt_scaler": tgt_scaler,
        "test_dates": test_dates,
        "feature_cols": effective_cols,
        "n_assets_pooled": len(TICKERS),
    }


if __name__ == "__main__":
    result = run_fusion_pipeline()
    print(f"[Fusion] X_train shape: {result['X_train'].shape}")
    print(f"[Fusion] X_test  shape: {result['X_test'].shape}")
