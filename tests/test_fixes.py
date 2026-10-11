"""
Regression tests for the evaluation-protocol fixes. Run:  python tests/test_fixes.py
(no model training; uses the cached fused datasets).
"""
import os
import sys

os.environ.setdefault("CRYPTO_HORSERACE_PROFILE", "smoke")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import LOOKBACK_WINDOW as L, TRAIN_RATIO, VOLATILITY_WINDOW as W
from src.baselines import krp_forecasts
from src.evaluate import holm_adjust, run_diebold_mariano, build_dm_table
from src.fuse_data import purge_count, run_fusion_pipeline
from src.horserace_baselines import build_naive_persistence_result


def test_naive_uses_raw_targets():
    for h in (1, 3, 7):
        p = run_fusion_pipeline(forecast_horizon=h, primary_ticker="BTC-USD")
        r = build_naive_persistence_result(p, h)
        vol = p["merged_df"]["Volatility"].values
        T = p["split_idx"] + L + h - 1 + np.arange(len(p["X_test"]))
        assert np.allclose(r["actual"], vol[T]) and np.allclose(r["predictions"], vol[T - h])
        # naive forecasts of a persistent series cannot be 5x worse than the series level
        assert r["rmse"] < 0.5 * np.nanmean(vol[T]), r["rmse"]


def test_actual_series_identical_across_models():
    p = run_fusion_pipeline(forecast_horizon=3, primary_ticker="BTC-USD")
    r = build_naive_persistence_result(p, 3)
    tgt = p["tgt_scaler"].inverse_transform(p["y_test"]).flatten()
    assert np.allclose(r["actual"], tgt)          # the network's actual == naive's actual


def test_purge_removes_future_labels():
    for h in (1, 3, 7):
        p = run_fusion_pipeline(forecast_horizon=h, primary_ticker="BTC-USD")
        last_train_label_row = (len(p["X_train"]) - 1) + L + h - 1
        first_test_origin_row = p["split_idx"] + L - 1
        assert last_train_label_row <= first_test_origin_row, (h, last_train_label_row, first_test_origin_row)
        assert p["n_purged"] == purge_count(h)
        # test windows must not move when purging
        assert len(p["X_test"]) == p["n_windows"] - p["split_idx"]


def test_krp_matches_bruteforce():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.03, 300)
    for h in (1, 3, 7):
        f = krp_forecasts(r, h)
        T = 200
        t0 = T - h
        known = r[t0 - (W - h) + 1: t0 + 1]
        m2 = np.mean(known ** 2)
        vec = np.r_[known, np.zeros(h)]                    # unknown days: mean 0 ...
        ss = np.sum(known ** 2) + h * m2 - np.sum(known) ** 2 / W   # ... and E r^2 = m2
        assert abs(f[T] - np.sqrt(ss / (W - 1))) < 1e-12


def test_dm_properties():
    rng = np.random.default_rng(1)
    e = rng.normal(size=400) ** 2
    assert run_diebold_mariano(e, e, h=3)["p_value"] > 0.99 or np.isclose(run_diebold_mariano(e, e, h=3)["dm_stat"], 0)
    a = rng.normal(size=400) ** 2
    b = (1.5 * rng.normal(size=400)) ** 2
    assert run_diebold_mariano(a, b, h=1)["significant"]
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.all(adj >= np.array([0.01, 0.04, 0.03])) and np.all(adj <= 1)


def test_dm_refuses_mismatched_actuals():
    n = 50
    a = {"name": "A", "actual": np.arange(n, dtype=float), "predictions": np.arange(n) + 0.1, "dates": pd.date_range("2024-01-01", periods=n)}
    b = {"name": "B", "actual": np.arange(n, dtype=float) * 3, "predictions": np.arange(n) * 3 + 0.2, "dates": pd.date_range("2024-01-01", periods=n)}
    try:
        build_dm_table({"A": a, "B": b}, forecast_horizon=1, output_suffix="__test__")
    except ValueError:
        return
    raise AssertionError("DM table accepted models scored on different realized series")


def test_offline_market_cache_is_idempotent():
    from src.ingest_market import fetch_market_data
    n1 = len(fetch_market_data("BTC-USD", use_api=False))
    n2 = len(fetch_market_data("BTC-USD", use_api=False))
    assert n1 == n2 == 1812, (n1, n2)


def test_gate_never_worse_on_calibration_and_zero_for_useless_net():
    from src.protocol import choose_gate
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    lin = y + rng.normal(scale=0.1, size=200)
    useless = rng.normal(scale=2.0, size=200)           # pure noise correction
    lam, scores = choose_gate(y, lin, useless)
    assert lam == 0.0 and scores[0.0] <= min(scores.values()) + 1e-12
    helpful = (y - lin)                                   # the exact residual
    lam, _ = choose_gate(y, lin, helpful)
    assert lam == 1.0


def test_stationary_features_are_scale_free():
    import config
    from src.fuse_data import fuse_pillars
    m = fuse_pillars("BTC-USD")
    for c in config.MARKET_STATIONARY_COLS + config.ONCHAIN_STATIONARY_COLS:
        assert c in m.columns and np.isfinite(m[c]).all(), c
    # price level grows ~20x over the sample, the stationary transforms must not
    assert m["Close"].iloc[-1] / m["Close"].iloc[0] > 5
    assert abs(m["Range_HL"].iloc[:300].mean() / m["Range_HL"].iloc[-300:].mean() - 1) < 1.0


def test_residual_target_scaling_is_winsorised():
    p = run_fusion_pipeline(forecast_horizon=1, primary_ticker="BTC-USD",
                            residual_target=np.r_[np.zeros(100), 50.0, np.zeros(1800)][:1812])
    assert p["y_train"].max() <= 5.0 + 1e-9
    assert type(p["tgt_scaler"]).__name__ == "StandardScaler"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)} tests passed")
