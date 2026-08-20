"""
DL-only sensitivity recompute: fill Clean/Perturbed/Delta DoC without the full horserace.

Skips baselines, hybrids, DM tests, plots, ablations. For each ticker × horizon:
  fuse → train CNN-BiLSTM-Attn → missing/noise perturbations → write sensitivity CSV.

Usage (from repo root):
  python scripts/recompute_sensitivity_doc.py
  python scripts/recompute_sensitivity_doc.py --out-dir results2/results
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.collection_utils import load_local_env

load_local_env(PROJECT_ROOT)

from config import (
    EXPERIMENT_HORIZONS,
    FULL_FEATURES,
    SEED,
    SENSITIVITY_MISSING_RATES,
    SENSITIVITY_NOISE_STD_RATES,
    TICKERS,
)
from src.evaluate import sensitivity_perturbation_metrics
from src.fuse_data import run_fusion_pipeline
from src.model_cnn_lstm import build_model, train_model


def ticker_suffix(ticker: str) -> str:
    return ticker.replace("-", "_")


def run_sensitivity_for_pair(
    ticker: str,
    horizon: int,
    out_dir: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"  SENSITIVITY+DoC  {ticker}  h={horizon}")
    print("=" * 60)

    pipeline = run_fusion_pipeline(
        feature_cols=FULL_FEATURES,
        primary_ticker=ticker,
        forecast_horizon=horizon,
    )
    X_train = pipeline["X_train"]
    X_test = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_test = pipeline["y_test"]
    tgt = pipeline["tgt_scaler"]

    model = build_model(X_train.shape[1], X_train.shape[2])
    train_model(model, X_train, y_train, X_test, y_test)

    y_pred = tgt.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_actual = tgt.inverse_transform(y_test).flatten()

    sens_rows = []
    for mr in SENSITIVITY_MISSING_RATES:
        Xp = X_test.copy()
        mask = rng.random(Xp.shape) < mr
        Xp[mask] = 0.0
        yp = tgt.inverse_transform(model.predict(Xp, verbose=0)).flatten()
        row = sensitivity_perturbation_metrics(
            "CNN-BiLSTM-Attn", y_actual, y_pred, yp, f"missing_{mr:.2f}"
        )
        if row:
            sens_rows.append(row)

    for nr in SENSITIVITY_NOISE_STD_RATES:
        Xp = X_test + rng.normal(0, nr, size=X_test.shape)
        yp = tgt.inverse_transform(model.predict(Xp, verbose=0)).flatten()
        row = sensitivity_perturbation_metrics(
            "CNN-BiLSTM-Attn", y_actual, y_pred, yp, f"noise_{nr:.2f}"
        )
        if row:
            sens_rows.append(row)

    df = pd.DataFrame(sens_rows)
    suf = f"{ticker_suffix(ticker)}_h{horizon}"
    out_path = os.path.join(out_dir, f"sensitivity_metrics_{suf}.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Sensitivity+DoC] wrote {out_path}")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Recompute sensitivity metrics with DoC (DL-only).")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "results2", "results"),
        help="Directory for sensitivity_metrics_*.csv (default: results2/results)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional subset of tickers (default: config TICKERS)",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of horizons (default: EXPERIMENT_HORIZONS)",
    )
    args = parser.parse_args()

    tickers = args.tickers or list(TICKERS)
    horizons = args.horizons or list(EXPERIMENT_HORIZONS)
    out_dir = os.path.abspath(args.out_dir)

    print(f"[Sensitivity+DoC] tickers={tickers}")
    print(f"[Sensitivity+DoC] horizons={horizons}")
    print(f"[Sensitivity+DoC] out_dir={out_dir}")

    # One RNG stream for all pairs so perturbation masks are reproducible.
    rng = np.random.default_rng(SEED)

    for horizon in horizons:
        for ticker in tickers:
            run_sensitivity_for_pair(ticker, horizon, out_dir, rng)

    print("\n[Sensitivity+DoC] done.")


if __name__ == "__main__":
    main()
