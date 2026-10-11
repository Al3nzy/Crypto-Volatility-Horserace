"""
Training and evaluation protocols that sit on top of the model builders:

* seed ensembles (every neural forecast is the mean of several seeds, and the
  per-seed spread is recorded),
* rolling-origin evaluation that spans the Terra-LUNA and FTX episodes,
* cross-asset transfer with per-asset scaling.
"""
import gc

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (BASELINE_N_JOBS, CROSS_ASSET_SEEDS, EPOCHS, LOOKBACK_WINDOW, ROLLING_EPOCHS,
                    ROLLING_INCLUDE_ARIMA_HYBRID, ROLLING_SEEDS, ROLLING_START, ROLLING_STEP_DAYS,
                    TICKERS, TRAIN_VERBOSE, VALIDATION_SPLIT)
from src.baselines import garch_forecasts_at_origins, har_design
from src.fuse_data import purge_count, run_fusion_pipeline
from src.model_cnn_lstm import build_model, set_global_determinism, train_model


def _rmse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_result(name, pred, actual, dates):
    pred, actual = np.asarray(pred, dtype=float).flatten(), np.asarray(actual, dtype=float).flatten()
    n = min(len(pred), len(actual), len(dates))
    return {"name": name, "predictions": pred[:n], "actual": actual[:n], "dates": dates[:n],
            "rmse": _rmse(actual[:n], pred[:n]), "mae": float(mean_absolute_error(actual[:n], pred[:n]))}


def assert_common_actual(results: dict):
    """Every model must be scored against one identical realized series."""
    names = list(results)
    ref = np.asarray(results[names[0]]["actual"], dtype=float)
    for k in names[1:]:
        a = np.asarray(results[k]["actual"], dtype=float)
        n = min(len(a), len(ref))
        if len(a) != len(ref) or not np.allclose(a[:n], ref[:n], rtol=1e-6, atol=1e-10):
            raise AssertionError(f"Realized series of '{k}' differs from '{names[0]}'.")


# ──────────────────────────────────────────────────────────────
# Seed ensembles
# ──────────────────────────────────────────────────────────────
def fit_seed_ensemble(builder, X_train, y_train, seeds, X_eval, epochs: int = EPOCHS, verbose: int = None,
                      X_cal=None, label: str = "net"):
    """
    Train one network per seed (builder(seed) -> compiled model), predict X_eval.
    Returns (first_model, first_history, scaled predictions of shape (S, n, 1)); with X_cal given, also
    the scaled predictions on X_cal as a fourth value. Only the first seed's network is kept (for the
    interpretation analyses). Training progress is printed (Keras progress bar when verbose=1).
    """
    verbose = TRAIN_VERBOSE if verbose is None else verbose
    first_model, first_hist, preds, cal = None, None, [], []
    for i, seed in enumerate(seeds):
        print(f"\n[fit] {label}  seed {seed}  ({i + 1}/{len(seeds)})", flush=True)
        model = builder(seed)
        hist = train_model(model, X_train, y_train, epochs=epochs, verbose=verbose)
        preds.append(model.predict(X_eval, verbose=0))
        if X_cal is not None:
            cal.append(model.predict(X_cal, verbose=0))
        if i == 0:
            first_model, first_hist = model, hist
        else:
            del model
            gc.collect()
    if X_cal is not None:
        return first_model, first_hist, np.stack(preds), np.stack(cal)
    return first_model, first_hist, np.stack(preds)


GATE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def choose_gate(y_cal, linear_cal, net_cal, grid=GATE_GRID):
    """
    Shrinkage weight for a residual network. Picks lambda in `grid` minimising RMSE of
    linear + lambda * net on a calibration set that precedes the test period; ties go to the
    smaller lambda. lambda = 0 returns the linear stage unchanged.
    """
    y, l, n = (np.asarray(a, dtype=float).flatten() for a in (y_cal, linear_cal, net_cal))
    scores = {g: float(np.sqrt(np.mean((y - (l + g * n)) ** 2))) for g in grid}
    best = min(grid, key=lambda g: (scores[g], g))
    return best, scores


def ensemble_predict(scaled_preds: np.ndarray, tgt_scaler):
    """Mean over seeds (in target units) and the per-seed forecasts."""
    per_seed = np.stack([tgt_scaler.inverse_transform(p).flatten() for p in scaled_preds])
    return per_seed.mean(axis=0), per_seed


def seed_metric_rows(model_name, per_seed, actual, seeds, ticker, horizon):
    return [{"Ticker": ticker, "Horizon": horizon, "Model": model_name, "Seed": int(s),
             "RMSE": _rmse(actual, p), "MAE": float(mean_absolute_error(actual, p))}
            for s, p in zip(seeds, per_seed)]


# ──────────────────────────────────────────────────────────────
# Rolling-origin evaluation
# ──────────────────────────────────────────────────────────────
def _windows(X_raw, n_win, h):
    L = LOOKBACK_WINDOW
    return np.stack([X_raw[i: i + L] for i in range(n_win)])


def _fit_scaled(X_tr, y_tr):
    F = X_tr.shape[-1]
    fs, ts = MinMaxScaler(), MinMaxScaler()
    fs.fit(X_tr.reshape(-1, F))
    ts.fit(y_tr.reshape(-1, 1))
    return fs, ts


def _scale(fs, X):
    F = X.shape[-1]
    return fs.transform(X.reshape(-1, F)).reshape(X.shape)


def run_rolling_origin(merged_df: pd.DataFrame, feature_cols, horizon: int, linear_stages: dict,
                       step_days: int = ROLLING_STEP_DAYS, epochs: int = ROLLING_EPOCHS,
                       seeds=None, start: str = ROLLING_START, n_jobs: int = BASELINE_N_JOBS,
                       include_garch: bool = True, hybrid_stages=None):
    """
    Expanding-window, rolling-origin evaluation from `start` to the end of the sample.

    Windows whose target date is on/after `start` are forecast in consecutive folds of `step_days`
    windows. For each fold the networks are retrained from scratch on every earlier window (minus the
    h-1 purged windows), with min-max feature scalers fitted on that training set only. ARIMA and GARCH
    are re-estimated at every origin, HAR-RV at every fold.

    linear_stages: {"ARIMA": array, "KRP": array}; stage[t] is the h-step forecast of row t made at
    row t-h. hybrid_stages: which stages get a residual-learning hybrid (default: KRP, plus ARIMA when
    ROLLING_INCLUDE_ARIMA_HYBRID). Each hybrid is gated: lambda is chosen per fold on the last
    VALIDATION_SPLIT share of the fold's training windows (see choose_gate), and every forecast is
    floored at half the smallest training volatility so a network can never produce a negative one.

    Returns (results dict, fold table).
    """
    seeds = list(seeds or ROLLING_SEEDS)
    if hybrid_stages is None:
        hybrid_stages = ["KRP"] + (["ARIMA"] if ROLLING_INCLUDE_ARIMA_HYBRID else [])
    hybrid_stages = [s for s in hybrid_stages if s in linear_stages]
    L, h = LOOKBACK_WINDOW, int(horizon)
    vol = merged_df["Volatility"].values.astype(float)
    idx = merged_df.index
    n = len(vol)
    n_win = n - L - h + 1
    tgt_rows = np.arange(n_win) + L + h - 1
    tgt_dates = idx[tgt_rows]
    start_k = int(np.argmax(tgt_dates >= pd.Timestamp(start)))
    if tgt_dates[start_k] < pd.Timestamp(start):
        raise ValueError("Rolling start is after the end of the sample.")
    purge = purge_count(h)

    X_raw = merged_df[list(feature_cols)].values.astype(float)
    X_win = _windows(X_raw, n_win, h)
    y_vol = vol[tgt_rows]
    arima_full = linear_stages["ARIMA"]
    y_res = {k: np.nan_to_num(vol[tgt_rows] - linear_stages[k][tgt_rows], nan=0.0) for k in hybrid_stages}

    folds = [(s, min(s + step_days, n_win)) for s in range(start_k, n_win, step_days)]
    ks_all = np.arange(start_k, n_win)
    T_all = tgt_rows[ks_all]
    actual = vol[T_all]
    dates = idx[T_all]

    cnn_pred = np.full(len(ks_all), np.nan)
    hyb_pred = {k: np.full(len(ks_all), np.nan) for k in hybrid_stages}
    har_pred = np.full(len(ks_all), np.nan)
    X_har, y_har = har_design(vol, h)
    fold_rows = []
    import tensorflow as tf

    for f, (s, e) in enumerate(folds):
        tr = np.arange(0, s - purge)
        te = np.arange(s, e)
        sl = slice(s - start_k, e - start_k)
        n_val = max(8, int(round(VALIDATION_SPLIT * len(tr))))
        cal = tr[-n_val:]
        floor = 0.5 * float(np.min(y_vol[tr]))
        print(f"\n[Rolling h={h}] fold {f + 1}/{len(folds)}  train={len(tr)}  test={len(te)}", flush=True)

        A = np.column_stack([np.ones(len(tr)), X_har[tr]])
        coef = np.linalg.lstsq(A, y_har[tr], rcond=None)[0]
        har_pred[sl] = np.column_stack([np.ones(len(te)), X_har[te]]) @ coef

        fs, ts_v = _fit_scaled(X_win[tr], y_vol[tr])
        Xtr, Xte, Xcal = _scale(fs, X_win[tr]), _scale(fs, X_win[te]), _scale(fs, X_win[cal])
        ytr = ts_v.transform(y_vol[tr].reshape(-1, 1))
        _, _, pv = fit_seed_ensemble(
            lambda sd: build_model(L, Xtr.shape[2], seed=sd), Xtr, ytr, seeds, Xte, epochs=epochs,
            label=f"rolling h={h} fold {f + 1} standalone")
        cnn_pred[sl] = np.maximum(ensemble_predict(pv, ts_v)[0], floor)

        row = {"Fold": f, "FirstTargetDate": str(dates[s - start_k].date()),
               "LastTargetDate": str(dates[e - 1 - start_k].date()),
               "TrainWindows": int(len(tr)), "TestWindows": int(len(te))}
        for stage in hybrid_stages:
            yst = y_res[stage]
            tr_res = StandardScaler().fit(yst[tr].reshape(-1, 1))
            ytr_r = np.clip(tr_res.transform(yst[tr].reshape(-1, 1)), -5.0, 5.0)
            _, _, pr, pc = fit_seed_ensemble(
                lambda sd: build_model(L, Xtr.shape[2], seed=sd), Xtr, ytr_r, seeds, Xte,
                epochs=epochs, X_cal=Xcal, label=f"rolling h={h} fold {f + 1} {stage}-hybrid")
            net_te = ensemble_predict(pr, tr_res)[0]
            net_cal = ensemble_predict(pc, tr_res)[0]
            lam, _ = choose_gate(y_vol[cal], linear_stages[stage][tgt_rows[cal]], net_cal)
            hyb_pred[stage][sl] = np.maximum(linear_stages[stage][tgt_rows[te]] + lam * net_te, floor)
            row[f"Gate_{stage}"] = lam
        fold_rows.append(row)
        tf.keras.backend.clear_session()
        gc.collect()

    arima_pred = arima_full[T_all]
    results = {
        "Naive_Persistence": make_result("Naive_Persistence", vol[T_all - h], actual, dates),
        "ARIMA": make_result("ARIMA", arima_pred, actual, dates),
        "HAR-RV": make_result("HAR-RV", har_pred, actual, dates),
        "CNN-BiLSTM-Attn": make_result("CNN-BiLSTM-Attn", cnn_pred, actual, dates),
        "Hybrid (ARIMA+DL)": make_result("Hybrid (ARIMA+DL)", 0.5 * arima_pred + 0.5 * cnn_pred, actual, dates),
    }
    if "ARIMA" in hyb_pred:
        results["Residual Hybrid (ARIMA+DL-resid)"] = make_result(
            "Residual Hybrid (ARIMA+DL-resid)", hyb_pred["ARIMA"], actual, dates)
    if "KRP" in linear_stages:
        results["KRP"] = make_result("KRP", linear_stages["KRP"][T_all], actual, dates)
    if "KRP" in hyb_pred:
        results["Residual Hybrid (KRP+DL-resid)"] = make_result(
            "Residual Hybrid (KRP+DL-resid)", hyb_pred["KRP"], actual, dates)
    if include_garch:
        ret = merged_df["Log_Return"]
        origins = T_all - h
        results["GARCH"] = make_result("GARCH", garch_forecasts_at_origins(ret, origins, h, False, n_jobs), actual, dates)
        results["GJR-GARCH"] = make_result("GJR-GARCH", garch_forecasts_at_origins(ret, origins, h, True, n_jobs), actual, dates)
    return results, pd.DataFrame(fold_rows)


# ──────────────────────────────────────────────────────────────
# Cross-asset transfer with per-asset scaling
# ──────────────────────────────────────────────────────────────
def run_cross_asset_transfer(feature_cols, holdout: str, horizon: int, seeds, in_domain_rmse=None):
    """
    Train one network on the pooled training windows of every other asset and
    test it, zero-shot, on the held-out asset's test windows. Each asset's
    windows are scaled with that asset's own training-period min-max
    statistics before pooling (so price, volume, and activity channels with
    different absolute magnitudes are comparable), and the held-out asset's
    test windows and target are scaled with its own training statistics. The
    validation tail used for early stopping contains the last 15% of every
    training asset.
    """
    parts_main, parts_val = [], []
    n_feat = None
    for t in TICKERS:
        if t == holdout:
            continue
        p = run_fusion_pipeline(feature_cols=feature_cols, primary_ticker=t, forecast_horizon=horizon)
        if n_feat is None:
            n_feat = p["X_train"].shape[2]
        elif p["X_train"].shape[2] != n_feat:
            raise ValueError(f"Feature-count mismatch while pooling ({t}).")
        nv = int(round(VALIDATION_SPLIT * len(p["X_train"])))
        parts_main.append((p["X_train"][:-nv], p["y_train"][:-nv]))
        parts_val.append((p["X_train"][-nv:], p["y_train"][-nv:]))
    X_tr = np.concatenate([x for x, _ in parts_main] + [x for x, _ in parts_val])
    y_tr = np.concatenate([y for _, y in parts_main] + [y for _, y in parts_val])
    ph = run_fusion_pipeline(feature_cols=feature_cols, primary_ticker=holdout, forecast_horizon=horizon)
    if ph["X_test"].shape[2] != n_feat:
        raise ValueError("Held-out asset has a different effective feature set.")
    _, _, pv = fit_seed_ensemble(
        lambda sd: build_model(X_tr.shape[1], X_tr.shape[2], seed=sd), X_tr, y_tr, seeds, ph["X_test"],
        label=f"transfer holdout={holdout} h={horizon}")
    pred, _ = ensemble_predict(pv, ph["tgt_scaler"])
    actual = ph["tgt_scaler"].inverse_transform(ph["y_test"]).flatten()
    row = {"Holdout": holdout, "Horizon": horizon, "RMSE": _rmse(actual, pred),
           "MAE": float(mean_absolute_error(actual, pred)), "Scaling": "per_asset",
           "TrainAssets": "|".join(t for t in TICKERS if t != holdout), "N_train": int(len(X_tr)),
           "N_test": int(len(actual)), "Seeds": len(seeds)}
    if in_domain_rmse is not None:
        row["InDomain_RMSE"] = float(in_domain_rmse)
        row["Ratio_vs_InDomain"] = row["RMSE"] / float(in_domain_rmse)
    return row
