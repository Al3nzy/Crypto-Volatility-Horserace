"""
Multimodal baselines for Q1-style horserace: naive persistence + SVR on the same
scaled windows as the main DL pipeline (no extra fusion).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

from config import (
    LOOKBACK_WINDOW,
    SVR_C,
    SVR_EPSILON,
    SVR_GAMMA,
    TRAIN_RATIO,
)
from src.evaluate import rmse
from src.fuse_data import create_windows


def build_naive_persistence_result(pipeline: dict, forecast_horizon: int) -> dict:
    """
    h-step persistence: predict vol at T using realized vol at forecast origin (T - forecast_horizon),
    aligned to the same windowed test set as CNN-BiLSTM-Attention.
    """
    merged = pipeline["merged_df"]
    feature_cols = pipeline["feature_cols"]
    vol = merged["Volatility"].values.astype(float)
    X_raw = merged[feature_cols].values.astype(float)
    y_raw = merged[["Volatility"]].values.astype(float)

    X_w, y_w = create_windows(X_raw, y_raw, forecast_horizon=forecast_horizon)
    split_idx = int(len(y_w) * TRAIN_RATIO)
    y_test_scaled = y_w[split_idx:]
    tgt = pipeline["tgt_scaler"]
    y_actual = tgt.inverse_transform(y_test_scaled).flatten()
    n_test = len(y_actual)

    naive = np.empty(n_test, dtype=float)
    for j in range(n_test):
        k = split_idx + j
        i = k
        T = i + LOOKBACK_WINDOW + forecast_horizon - 1
        # Forecast origin is T - forecast_horizon (i.e. i + LOOKBACK_WINDOW - 1)
        naive[j] = vol[T - forecast_horizon]

    dates = pipeline["test_dates"]
    if hasattr(dates, "__len__"):
        dates = dates[:n_test]

    return {
        "name": "Naive_Persistence",
        "predictions": naive,
        "actual": y_actual,
        "dates": dates,
        "rmse": rmse(y_actual, naive),
        "mae": mean_absolute_error(y_actual, naive),
    }


def build_svr_result(pipeline: dict) -> dict:
    """RBF SVR on flattened MinMax-scaled windows; same target scaler as DL."""
    X_train = pipeline["X_train"]
    X_test = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_test = pipeline["y_test"]
    tgt = pipeline["tgt_scaler"]

    n_feat = int(np.prod(X_train.shape[1:]))
    Xtr = X_train.reshape(X_train.shape[0], n_feat)
    Xte = X_test.reshape(X_test.shape[0], n_feat)
    ytr = y_train.ravel()

    svr = SVR(kernel="rbf", C=SVR_C, gamma=SVR_GAMMA, epsilon=SVR_EPSILON)
    svr.fit(Xtr, ytr)
    pred_scaled = svr.predict(Xte).reshape(-1, 1)
    y_pred = tgt.inverse_transform(pred_scaled).flatten()
    y_actual = tgt.inverse_transform(y_test).flatten()
    n = min(len(y_pred), len(y_actual))
    dates = pipeline["test_dates"][:n]

    return {
        "name": "SVR (RBF)",
        "predictions": y_pred[:n],
        "actual": y_actual[:n],
        "dates": dates,
        "rmse": rmse(y_actual[:n], y_pred[:n]),
        "mae": mean_absolute_error(y_actual[:n], y_pred[:n]),
    }
