"""
main.py – Orchestrator for the Crypto Volatility Horserace.

Runs every experiment as an independent TASK (one per asset-horizon combo, rolling-origin
evaluation, ablation, cross-asset transfer). Each task is a separate Python process, so memory
held by TensorFlow is released when it ends, finished tasks are skipped on restart, and the
Keras progress bars stay visible on the console.

Phases inside a combo task:
  Phase 1: ETL (ingest three pillars → fuse & preprocess)
  Phase 2: Statistical baselines (ARIMA, GARCH)
  Phase 3: Deep-learning models (seed ensembles of CNN-BiLSTM-Attention and AHI)
  Phase 4: Benchmarking, interpretation, robustness, rolling-origin and cross-asset evaluation
"""
import os
import sys
import time

# Must be set before TensorFlow is imported anywhere in the process (see
# src/model_cnn_lstm.py, which sets the same pair defensively as well).
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.collection_utils import load_local_env

load_local_env(PROJECT_ROOT)

from config import (
    ACTIVE_DATA_EXPERIMENT,
    ASSET_ONCHAIN_CORE_FEATURE_COLS,
    ASSET_SENTIMENT_FEATURE_COLS,
    BASELINE_N_JOBS,
    ENABLE_GOOGLE_TRENDS,
    ENABLE_ONCHAIN_API,
    FAST_DEV_MODE,
    FULL_FEATURES,
    FULL_FEATURES_NO_RV,
    MARKET_FEATURE_COLS,
    RESULTS_DIR, TRAIN_RATIO,
    SHOCK_WINDOWS, RUN_HISTORY_FILE, RUN_ABLATIONS, RUN_ABLATIONS_FOR_ALL_TICKERS,
    RUN_DL_ABLATION_BASELINES,
    RUN_HAR_GJR_BASELINES,
    RUN_NAIVE_PERSISTENCE_BASELINE,
    RUN_SVR_BASELINE,
    TICKERS, PRIMARY_TICKER, MARKET_INTERVAL,
    START_DATE, END_DATE,
    LOOKBACK_WINDOW, FORECAST_HORIZON, EXPERIMENT_HORIZONS,
    TRANSACTION_COST_BPS, MC_DROPOUT_PASSES, MC_INTERVAL_ALPHA,
    SENSITIVITY_MISSING_RATES, SENSITIVITY_NOISE_STD_RATES,
    USE_POOLED_DL_TRAINING, USE_MARKET_API, USE_API, SEED,
    CORE_TICKERS, PROFILE, EPOCHS, DL_SEEDS, AHI_SEEDS, BASELINE_DL_SEEDS, ABLATION_SEEDS, ABLATION_HORIZONS, PERMUTATION_REPEATS,
    RUN_ROLLING_ORIGIN, ROLLING_START, ROLLING_STEP_DAYS, ROLLING_EPOCHS, ROLLING_SEEDS, CROSS_ASSET_HORIZONS,
    CROSS_ASSET_SEEDS, ROLLING_INCLUDE_ARIMA_HYBRID, FEATURE_SET, FEATURE_GROUPS, MARKET_RAW_COLS, ONCHAIN_RAW_COLS,
    WORKERS, RESUME, TRAIN_VERBOSE,
    VALIDATION_SPLIT, PURGE_TRAIN_OVERLAP, GARCH_ESTIMAND, ASSET_ONCHAIN_FEATURE_COLS, RV_FEATURE_COLS,
)

# ── Phase 1 imports ──
from src.data_collection import collect_feature_datasets
from src.fuse_data import run_fusion_pipeline, run_fusion_pipeline_pooled, _effective_feature_cols
from src.data_quality import run_data_preflight

# ── Phase 2 imports ──
from src.baselines import run_all_baselines, get_arima_forecasts_for_residuals, krp_forecasts

# ── Phase 3 imports ──
from src.model_cnn_lstm import (
    build_model, train_model, extract_attention_weights, set_global_determinism,
)
from src.model_dl_baselines import build_cnn_only, build_lstm_only, build_gru_only
from src.horserace_baselines import build_naive_persistence_result, build_svr_result, build_krp_result

# ── Phase 4 imports ──
from src.evaluate import (
    rmse, build_comparison_table,
    plot_predictions_overlay, plot_attention_heatmap,
    plot_attention_per_feature, plot_training_curves,
    build_shock_window_table, build_dm_table, append_run_history,
    build_regime_table, build_backtest_table, build_ratio_ci_table,
    mc_dropout_intervals, interval_coverage, sensitivity_perturbation_metrics,
)
from src.interpret import (
    attention_lag_profile, attention_summary, permutation_importance, pillar_of,
    split_conformal_interval, perturb,
)
from src.protocol import (
    make_result, assert_common_actual, fit_seed_ensemble, ensemble_predict, choose_gate,
    seed_metric_rows, run_rolling_origin, run_cross_asset_transfer,
)
from sklearn.metrics import mean_absolute_error


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _format_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _save(df: pd.DataFrame, name: str, index: bool = False):
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=index)
    return path


def _save_predictions(results: dict, name: str):
    rows = []
    for model_name, res in results.items():
        n = min(len(res["predictions"]), len(res["actual"]), len(res["dates"]))
        rows.append(pd.DataFrame({
            "Date": pd.DatetimeIndex(res["dates"])[:n], "Model": model_name,
            "Actual": np.asarray(res["actual"])[:n], "Prediction": np.asarray(res["predictions"])[:n],
        }))
    _save(pd.concat(rows), name)



def _save_predictions(results: dict, name: str):
    rows = []
    for model_name, res in results.items():
        n = min(len(res["predictions"]), len(res["actual"]), len(res["dates"]))
        rows.append(pd.DataFrame({
            "Date": pd.DatetimeIndex(res["dates"])[:n], "Model": model_name,
            "Actual": np.asarray(res["actual"])[:n], "Prediction": np.asarray(res["predictions"])[:n],
        }))
    _save(pd.concat(rows), name)


def feature_audit(merged_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Which of the requested input channels actually reached the models, and why not."""
    active = set(_effective_feature_cols(merged_df, list(FULL_FEATURES)))
    rows = []
    for c in FULL_FEATURES:
        if c not in merged_df.columns:
            status = "dropped_missing_column"
        elif c in active:
            status = "active"
        elif merged_df[c].notna().sum() == 0:
            status = "dropped_all_missing"
        else:
            status = "dropped_constant"
        rows.append({"Ticker": ticker, "Feature": c, "Pillar": pillar_of(c), "Status": status,
                     "UniqueValues": int(merged_df[c].nunique()) if c in merged_df.columns else 0})
    return pd.DataFrame(rows)



def _floor_from_target_scaler(tgt):
    """Forecasts of volatility are floored at half the smallest training volatility."""
    return 0.5 * float(tgt.data_min_[0])


def run_dl_experiment(feature_cols, experiment_name="full_multimodal", primary_ticker=None,
                      forecast_horizon=FORECAST_HORIZON, pipeline=None, seeds=None, label="standalone"):
    """Train a seed ensemble of the standalone CNN-BiLSTM-Attention network on volatility."""
    seeds = list(seeds or DL_SEEDS)
    if pipeline is None:
        pipeline = run_fusion_pipeline(feature_cols=feature_cols, primary_ticker=primary_ticker,
                                       forecast_horizon=forecast_horizon)
    X_train, X_test = pipeline["X_train"], pipeline["X_test"]
    tgt = pipeline["tgt_scaler"]
    model, history, sp = fit_seed_ensemble(
        lambda sd: build_model(X_train.shape[1], X_train.shape[2], seed=sd),
        X_train, pipeline["y_train"], seeds, X_test, label=f"{label} ({experiment_name})")
    pred, per_seed = ensemble_predict(sp, tgt)
    pred = np.maximum(pred, _floor_from_target_scaler(tgt))
    actual = tgt.inverse_transform(pipeline["y_test"]).flatten()
    result = make_result(f"CNN-BiLSTM-Attn-{experiment_name}", pred, actual, pipeline["test_dates"])
    return {"pipeline": pipeline, "model": model, "history": history, "result": result,
            "per_seed": per_seed, "seeds": seeds, "n_train_samples": int(X_train.shape[0]),
            "n_test_samples": int(X_test.shape[0]), "n_features": int(X_train.shape[2])}


def run_dl_baseline_ensemble(pipeline, variant, seeds):
    builders = {"cnn_only": (build_cnn_only, "CNN-only"), "lstm_only": (build_lstm_only, "LSTM-only"),
                "gru_only": (build_gru_only, "GRU-only")}
    fn, label = builders[variant]
    X_train, X_test = pipeline["X_train"], pipeline["X_test"]
    tgt = pipeline["tgt_scaler"]

    def _build(sd):
        set_global_determinism(sd)
        return fn(X_train.shape[1], X_train.shape[2])

    _, _, sp = fit_seed_ensemble(_build, X_train, pipeline["y_train"], seeds, X_test, label=label)
    pred, per_seed = ensemble_predict(sp, tgt)
    pred = np.maximum(pred, _floor_from_target_scaler(tgt))
    actual = tgt.inverse_transform(pipeline["y_test"]).flatten()
    return make_result(label, pred, actual, pipeline["test_dates"]), per_seed


def _predictor(entry):
    """Forecast function of a network entry: floor(offset + gate * network output)."""
    tgt, model = entry["tgt"], entry["model"]
    off, gate, floor = entry.get("offset_test"), entry.get("gate", 1.0), entry.get("floor")

    def f(X):
        y = tgt.inverse_transform(model.predict(X, verbose=0)).flatten()
        if off is not None:
            y = np.asarray(off, dtype=float)[: len(y)] + gate * y
        return np.maximum(y, floor) if floor is not None else y
    return f


def fit_residual_hybrid(label, linear_full, vol_vals, ticker, horizon, seeds):
    """
    Gated residual-learning hybrid. `linear_full[t]` is the h-step forecast of row t made at row
    t-h by a statistical stage. The network learns the (standardised, winsorised) residual of that
    stage; the forecast is floor(stage + lambda * network), where lambda in {0, .25, .5, .75, 1} is
    chosen on the last 15% of the training windows (lambda = 0 gives back the statistical stage).
    """
    residuals = np.nan_to_num(vol_vals - linear_full, nan=0.0)
    pipe = run_fusion_pipeline(feature_cols=FULL_FEATURES, residual_target=residuals,
                               primary_ticker=ticker, forecast_horizon=horizon)
    Xtr, Xte, tgt = pipe["X_train"], pipe["X_test"], pipe["tgt_scaler"]
    n_val = max(8, int(round(VALIDATION_SPLIT * len(Xtr))))
    model, _, sp, sp_cal = fit_seed_ensemble(
        lambda sd: build_model(Xtr.shape[1], Xtr.shape[2], seed=sd), Xtr, pipe["y_train"], seeds, Xte,
        X_cal=Xtr[-n_val:], label=label)
    resid_pred, per_seed = ensemble_predict(sp, tgt)
    net_cal, _ = ensemble_predict(sp_cal, tgt)
    rows_val = np.arange(len(Xtr) - n_val, len(Xtr)) + LOOKBACK_WINDOW + horizon - 1
    lam, scores = choose_gate(vol_vals[rows_val], linear_full[rows_val], net_cal)
    floor = 0.5 * float(np.min(vol_vals[: pipe["split_idx"] + LOOKBACK_WINDOW]))
    start = pipe["split_idx"] + LOOKBACK_WINDOW + horizon - 1
    sl = slice(start, start + len(resid_pred))
    offset_test, actual = linear_full[sl], vol_vals[sl]
    final = np.maximum(offset_test + lam * resid_pred, floor)
    ungated = np.maximum(offset_test + resid_pred, floor)
    result = make_result(label, final, actual, pipe["test_dates"])
    seed_rows = seed_metric_rows(label, np.maximum(offset_test[None, :] + lam * per_seed, floor), actual,
                                 seeds, ticker, horizon)
    gate_row = {"Ticker": ticker, "Horizon": horizon, "Model": label, "Lambda": lam,
                "ValRMSE_stage_only": scores[0.0], "ValRMSE_ungated": scores[1.0], "ValRMSE_gated": scores[lam],
                "TestRMSE_stage_only": float(np.sqrt(np.mean((actual - offset_test) ** 2))),
                "TestRMSE_ungated": float(np.sqrt(np.mean((actual - ungated) ** 2))),
                "TestRMSE_gated": result["rmse"], "NetCorrection_std": float(np.std(resid_pred))}
    return {"result": result, "model": model, "tgt": tgt, "Xtr": Xtr, "offset_test": offset_test,
            "seed_rows": seed_rows, "pipe": pipe, "gate": lam, "floor": floor, "gate_row": gate_row,
            "n_val": n_val, "rows_val": rows_val}


def run_interpretation_and_robustness(entries, X_test, y_actual, feature_cols, ticker, horizon, suf):
    """
    entries: {label: {model, tgt, offset_test, offset_val, X_val, y_val}} for the standalone network and AHI
    (X_val/y_val: the validation tail of the training windows, used as the conformal calibration set).
    Writes attention profiles, permutation importance, MC-dropout and conformal intervals, and the
    perturbation tests (all channels, and sentiment/on-chain channels only).
    """
    import matplotlib
    matplotlib.use("Agg")
    from src.interpret import plot_attention_lag_profile, plot_permutation_importance

    profiles, summaries, perm_frames, interval_rows, sens_rows = {}, [], [], [], []
    exo_idx = [j for j, f in enumerate(feature_cols) if pillar_of(f) in ("sentiment", "onchain")]
    channel_sets = {"all": list(range(len(feature_cols)))}
    if exo_idx:
        channel_sets["exogenous"] = exo_idx

    for label, e in entries.items():
        predict = _predictor(e)
        base = predict(X_test)

        # Attention (every test window, all heads, all query positions)
        attn = extract_attention_weights(e["model"], X_test)
        prof = attention_lag_profile(attn)
        profiles[label] = prof
        s = attention_summary(attn)
        s.update({"Ticker": ticker, "Horizon": horizon, "Model": label})
        summaries.append(s)

        # Permutation importance (by feature and by pillar)
        perm_frames.append(permutation_importance(predict, X_test, y_actual, feature_cols,
                                                  repeats=PERMUTATION_REPEATS, model_label=label))

        # Intervals: MC dropout (model uncertainty) and split-conformal (calibrated)
        _, lo, hi = mc_dropout_intervals(e["model"], X_test, e["tgt"], passes=MC_DROPOUT_PASSES,
                                         alpha=MC_INTERVAL_ALPHA, offset=e.get("offset_test"),
                                         gate=e.get("gate", 1.0))
        st = interval_coverage(y_actual, lo, hi)
        interval_rows.append({"Ticker": ticker, "Horizon": horizon, "Model": label, "Method": "mc_dropout",
                              "Nominal": 1 - MC_INTERVAL_ALPHA, "Coverage": st["coverage"], "AvgWidth": st["avg_width"]})
        p_val = _predictor({**e, "offset_test": e.get("offset_val")})(e["X_val"])
        clo, chi, q = split_conformal_interval(e["y_val"], p_val, base, alpha=MC_INTERVAL_ALPHA)
        st = interval_coverage(y_actual, clo, chi)
        interval_rows.append({"Ticker": ticker, "Horizon": horizon, "Model": label, "Method": "split_conformal",
                              "Nominal": 1 - MC_INTERVAL_ALPHA, "Coverage": st["coverage"], "AvgWidth": st["avg_width"]})

        # Perturbation tests
        for ch_name, ch in channel_sets.items():
            rng = np.random.default_rng(SEED)
            for mr in SENSITIVITY_MISSING_RATES:
                row = sensitivity_perturbation_metrics(label, y_actual, base,
                                                       predict(perturb(X_test, "missing", mr, ch, rng)), f"missing_{mr:.2f}")
                if row:
                    row.update({"Channels": ch_name, "Ticker": ticker, "Horizon": horizon}); sens_rows.append(row)
            for nr in SENSITIVITY_NOISE_STD_RATES:
                row = sensitivity_perturbation_metrics(label, y_actual, base,
                                                       predict(perturb(X_test, "noise", nr, ch, rng)), f"noise_{nr:.2f}")
                if row:
                    row.update({"Channels": ch_name, "Ticker": ticker, "Horizon": horizon}); sens_rows.append(row)

    _save(pd.DataFrame(summaries), f"attention_summary_{suf}.csv")
    long = pd.concat([p.assign(Model=k) for k, p in profiles.items()])
    _save(long, f"attention_lag_profile_{suf}.csv")
    plot_attention_lag_profile(profiles, os.path.join(RESULTS_DIR, f"attention_lag_profile_{suf}.png"),
                               title=f"Temporal attention, all test windows ({suf})")
    perm = pd.concat(perm_frames)
    perm.insert(0, "Horizon", horizon); perm.insert(0, "Ticker", ticker)
    _save(perm, f"permutation_importance_{suf}.csv")
    plot_permutation_importance(perm, os.path.join(RESULTS_DIR, f"permutation_importance_{suf}.png"),
                                title=f"Permutation importance on the test year ({suf})")
    _save(pd.DataFrame(interval_rows), f"interval_metrics_{suf}.csv")
    _save(pd.DataFrame(sens_rows), f"sensitivity_metrics_{suf}.csv")



ANALYSIS_REFERENCES = ["Naive_Persistence", "HAR-RV", "ARIMA", "KRP"]


# ──────────────────────────────────────────────────────────────
# Task: ablation (feature groups), standalone network, several seeds
# ──────────────────────────────────────────────────────────────
def ablation_feature_sets():
    g = FEATURE_GROUPS
    raw_full = MARKET_RAW_COLS + ASSET_SENTIMENT_FEATURE_COLS + ONCHAIN_RAW_COLS + RV_FEATURE_COLS
    sets = {
        "rv_only": g["lagged_vol"],
        "market_only": g["market"],
        "sentiment_only": g["sentiment"],
        "onchain_only": g["onchain"],
        "market_rv": g["market"] + g["lagged_vol"],
        "full_multimodal_no_rv": g["market"] + g["sentiment"] + g["onchain"],
        "full_minus_sentiment": g["market"] + g["onchain"] + g["lagged_vol"],
        "full_minus_onchain": g["market"] + g["sentiment"] + g["lagged_vol"],
        "full_multimodal": g["market"] + g["sentiment"] + g["onchain"] + g["lagged_vol"],
    }
    if FEATURE_SET == "stationary":
        # Same information as price/volume LEVELS, for a direct stationary-versus-raw comparison
        sets["market_raw_levels"] = MARKET_RAW_COLS
        sets["full_raw_levels"] = raw_full
    return sets


def run_ablation_suite(ticker, horizon, merged_df, suf):
    rows, run_rows = [], []
    sets = ablation_feature_sets()
    for k, (ab_name, ab_cols) in enumerate(sets.items(), 1):
        if not set(ab_cols).issubset(set(merged_df.columns)):
            continue
        print(f"\n[Ablation {k}/{len(sets)}] {ab_name}", flush=True)
        pipe = run_fusion_pipeline(feature_cols=ab_cols, primary_ticker=ticker, forecast_horizon=horizon)
        _, _, sp = fit_seed_ensemble(
            lambda sd: build_model(pipe["X_train"].shape[1], pipe["X_train"].shape[2], seed=sd),
            pipe["X_train"], pipe["y_train"], ABLATION_SEEDS, pipe["X_test"], label=f"ablation {ab_name}")
        pred, per_seed = ensemble_predict(sp, pipe["tgt_scaler"])
        actual = pipe["tgt_scaler"].inverse_transform(pipe["y_test"]).flatten()
        per_rmse = [float(np.sqrt(np.mean((actual - p) ** 2))) for p in per_seed]
        for sd, v in zip(ABLATION_SEEDS, per_rmse):
            run_rows.append({"Experiment": ab_name, "Ticker": ticker, "Horizon": horizon, "Seed": sd, "RMSE": v})
        rows.append({"Experiment": ab_name, "Ticker": ticker, "NumFeatures": int(pipe["X_train"].shape[2]),
                     "RMSE": float(np.mean(per_rmse)),
                     "RMSE_std": float(np.std(per_rmse, ddof=1)) if len(per_rmse) > 1 else float("nan"),
                     "RMSE_ens": float(np.sqrt(np.mean((actual - pred) ** 2))),
                     "MAE": float(np.mean(np.abs(actual - pred))), "N_seeds": len(per_rmse),
                     "Features": "|".join(pipe["feature_cols"])})
        print(f"[Ablation] {ab_name}: RMSE={rows[-1]['RMSE']:.6f} +/- {rows[-1]['RMSE_std']:.6f} (n={len(per_rmse)})", flush=True)
        _save(pd.DataFrame(rows).sort_values("RMSE"), f"ablation_results_{suf}.csv")    # saved after every config
        _save(pd.DataFrame(run_rows), f"ablation_runs_{suf}.csv")


# ──────────────────────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────────────────────
def _suf(ticker, horizon):
    return f"{ticker.replace('-', '_')}_h{horizon}"


def task_combo(ticker, horizon):
    """Hold-out evaluation of every model for one asset and horizon, with tables and analyses."""
    suf = _suf(ticker, horizon)
    combo_start = time.time()
    pipeline = run_fusion_pipeline(feature_cols=FULL_FEATURES, primary_ticker=ticker, forecast_horizon=horizon)
    X_test, test_dates = pipeline["X_test"], pipeline["test_dates"]
    feature_cols, merged_df = pipeline["feature_cols"], pipeline["merged_df"]
    _save(feature_audit(merged_df, ticker), f"feature_audit_{ticker.replace('-', '_')}.csv")

    print("\n> PHASE 2: statistical baselines", flush=True)
    arima_full_pred, vol_vals = get_arima_forecasts_for_residuals(merged_df["Volatility"], forecast_horizon=horizon,
                                                                  n_jobs=BASELINE_N_JOBS)
    krp_full = krp_forecasts(merged_df["Log_Return"].values, horizon)
    baseline_results = run_all_baselines(merged_df, forecast_horizon=horizon, include_har_gjr=RUN_HAR_GJR_BASELINES,
                                         n_jobs=BASELINE_N_JOBS, arima_full=arima_full_pred)

    print("\n> PHASE 3: neural models", flush=True)
    seed_rows = []
    dl_out = run_dl_experiment(FULL_FEATURES, "full", ticker, horizon, pipeline=pipeline)
    model, history, dl_result = dl_out["model"], dl_out["history"], dl_out["result"]
    y_actual = dl_result["actual"]
    seed_rows += seed_metric_rows("CNN-BiLSTM-Attn", dl_out["per_seed"], y_actual, dl_out["seeds"], ticker, horizon)
    print(f"[CNN-BiLSTM-Attn] ensemble RMSE={dl_result['rmse']:.6f}  MAE={dl_result['mae']:.6f}", flush=True)
    arima_pred = np.asarray(baseline_results["ARIMA"]["predictions"])
    n_align = min(len(arima_pred), len(dl_result["predictions"]))
    hybrid_result = make_result("Hybrid (ARIMA+DL)",
                                0.5 * arima_pred[:n_align] + 0.5 * np.asarray(dl_result["predictions"])[:n_align],
                                np.asarray(y_actual)[:n_align], np.asarray(test_dates)[:n_align])

    ahi = fit_residual_hybrid("Residual Hybrid (ARIMA+DL-resid)", arima_full_pred, vol_vals, ticker, horizon, AHI_SEEDS)
    seed_rows += ahi["seed_rows"]
    print(f"[AHI] RMSE={ahi['result']['rmse']:.6f}  gate lambda={ahi['gate']}", flush=True)
    ahik = fit_residual_hybrid("Residual Hybrid (KRP+DL-resid)", krp_full, vol_vals, ticker, horizon, DL_SEEDS)
    seed_rows += ahik["seed_rows"]
    print(f"[AHI-K] RMSE={ahik['result']['rmse']:.6f}  gate lambda={ahik['gate']}", flush=True)
    _save(pd.DataFrame([ahi["gate_row"], ahik["gate_row"]]), f"gate_{suf}.csv")

    horserace_ml = {}
    if RUN_NAIVE_PERSISTENCE_BASELINE:
        horserace_ml["Naive_Persistence"] = build_naive_persistence_result(pipeline, horizon)
    horserace_ml["KRP"] = build_krp_result(pipeline, horizon)
    if RUN_SVR_BASELINE:
        print("\n[Horserace] SVR (RBF) …", flush=True)
        horserace_ml["SVR (RBF)"] = build_svr_result(pipeline)
    if RUN_DL_ABLATION_BASELINES:
        for variant in ("cnn_only", "lstm_only", "gru_only"):
            r, per_seed = run_dl_baseline_ensemble(pipeline, variant, BASELINE_DL_SEEDS)
            horserace_ml[r["name"]] = r
            seed_rows += seed_metric_rows(r["name"], per_seed, r["actual"], BASELINE_DL_SEEDS, ticker, horizon)

    all_results = {**horserace_ml, "ARIMA": baseline_results["ARIMA"], "GARCH": baseline_results["GARCH"],
                   "CNN-BiLSTM-Attn": dl_result, "Hybrid (ARIMA+DL)": hybrid_result,
                   "Residual Hybrid (ARIMA+DL-resid)": ahi["result"], "Residual Hybrid (KRP+DL-resid)": ahik["result"]}
    for key in ("HAR-RV", "GJR-GARCH"):
        if key in baseline_results:
            all_results[key] = baseline_results[key]
    assert_common_actual(all_results)

    print("\n> PHASE 4: tables, interpretation, robustness", flush=True)
    _save_predictions(all_results, f"predictions_{suf}.csv")
    build_comparison_table(all_results, output_suffix=suf)
    shock_df = build_shock_window_table(all_results, SHOCK_WINDOWS, output_suffix=suf)
    build_regime_table(all_results, merged_df=merged_df, output_suffix=suf)
    build_dm_table(all_results, forecast_horizon=horizon, output_suffix=suf)
    build_ratio_ci_table(all_results, ANALYSIS_REFERENCES, horizon, output_suffix=suf)
    build_backtest_table(all_results, TRANSACTION_COST_BPS, output_suffix=suf)
    plot_predictions_overlay(all_results, test_dates, output_suffix=suf)
    plot_training_curves(history, output_suffix=suf)

    nv = int(round(VALIDATION_SPLIT * len(pipeline["X_train"])))
    entries = {
        "CNN-BiLSTM-Attn": {
            "model": model, "tgt": pipeline["tgt_scaler"], "offset_test": None, "offset_val": None, "gate": 1.0,
            "floor": _floor_from_target_scaler(pipeline["tgt_scaler"]), "X_val": pipeline["X_train"][-nv:],
            "y_val": pipeline["tgt_scaler"].inverse_transform(pipeline["y_train"][-nv:]).flatten()},
        "AHI": {
            "model": ahi["model"], "tgt": ahi["tgt"], "offset_test": ahi["offset_test"], "gate": ahi["gate"],
            "floor": ahi["floor"], "offset_val": arima_full_pred[ahi["rows_val"]], "X_val": ahi["Xtr"][-ahi["n_val"]:],
            "y_val": vol_vals[ahi["rows_val"]]},
        "AHI-K": {
            "model": ahik["model"], "tgt": ahik["tgt"], "offset_test": ahik["offset_test"], "gate": ahik["gate"],
            "floor": ahik["floor"], "offset_val": krp_full[ahik["rows_val"]], "X_val": ahik["Xtr"][-ahik["n_val"]:],
            "y_val": vol_vals[ahik["rows_val"]]},
    }
    run_interpretation_and_robustness(entries, X_test, y_actual, feature_cols, ticker, horizon, suf)

    elapsed = time.time() - combo_start
    _save(pd.concat([pd.DataFrame(seed_rows)]), f"seed_metrics_{suf}.csv")
    _save(pd.DataFrame([{"Ticker": ticker, "Horizon": horizon, "NTrain": dl_out["n_train_samples"],
                         "NTest": dl_out["n_test_samples"], "NPurged": pipeline["n_purged"],
                         "NFeatures": dl_out["n_features"], "Features": "|".join(feature_cols),
                         "FeatureSet": FEATURE_SET, "TestFirstTarget": str(pd.Timestamp(test_dates[0]).date()),
                         "TestLastTarget": str(pd.Timestamp(test_dates[-1]).date())}]), f"sample_manifest_{suf}.csv")
    append_run_history({
        "run_ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "profile": PROFILE, "feature_set": FEATURE_SET,
        "primary_ticker": ticker, "forecast_horizon": horizon, "n_train_samples": dl_out["n_train_samples"],
        "n_test_samples": dl_out["n_test_samples"], "n_features": dl_out["n_features"],
        "winner": min(all_results.keys(), key=lambda k: all_results[k]["rmse"]),
        "combo_execution_time_sec": round(elapsed, 2)}, RUN_HISTORY_FILE)


def task_rolling(ticker, horizon):
    """Rolling-origin evaluation across the Terra-LUNA and FTX episodes."""
    suf = _suf(ticker, horizon)
    pipeline = run_fusion_pipeline(feature_cols=FULL_FEATURES, primary_ticker=ticker, forecast_horizon=horizon)
    merged_df, feature_cols = pipeline["merged_df"], pipeline["feature_cols"]
    arima_full, _ = get_arima_forecasts_for_residuals(merged_df["Volatility"], forecast_horizon=horizon,
                                                      n_jobs=BASELINE_N_JOBS)
    krp_full = krp_forecasts(merged_df["Log_Return"].values, horizon)
    results, folds = run_rolling_origin(merged_df, feature_cols, horizon, {"ARIMA": arima_full, "KRP": krp_full})
    tag = f"rolling_{suf}"
    assert_common_actual(results)
    _save(folds, f"rolling_folds_{suf}.csv")
    _save_predictions(results, f"predictions_{tag}.csv")
    build_comparison_table(results, output_suffix=tag)
    build_shock_window_table(results, SHOCK_WINDOWS, output_suffix=tag)
    build_regime_table(results, merged_df=merged_df, output_suffix=tag)
    build_dm_table(results, forecast_horizon=horizon, output_suffix=tag)
    build_ratio_ci_table(results, ANALYSIS_REFERENCES, horizon, output_suffix=tag)
    plot_predictions_overlay(results, list(results.values())[0]["dates"], output_suffix=tag)


def task_ablation(ticker, horizon):
    suf = _suf(ticker, horizon)
    merged_df = run_fusion_pipeline(feature_cols=FULL_FEATURES, primary_ticker=ticker,
                                    forecast_horizon=horizon)["merged_df"]
    run_ablation_suite(ticker, horizon, merged_df, suf)


def task_cross(holdout, horizon):
    row = run_cross_asset_transfer(FULL_FEATURES, holdout, horizon, CROSS_ASSET_SEEDS)
    _save(pd.DataFrame([row]), f"cross_asset_{_suf(holdout, horizon)}.csv")


TASKS = {"combo": task_combo, "rolling": task_rolling, "ablation": task_ablation, "cross": task_cross}


def build_task_list():
    """(kind, ticker, horizon) in the order they run when WORKERS=1."""
    tasks = []
    for h in EXPERIMENT_HORIZONS:
        for t in TICKERS:
            tasks.append(("combo", t, h))
            if RUN_ROLLING_ORIGIN and not FAST_DEV_MODE:
                tasks.append(("rolling", t, h))
        if RUN_ABLATIONS and not FAST_DEV_MODE and h in ABLATION_HORIZONS:
            tasks.append(("ablation", PRIMARY_TICKER if PRIMARY_TICKER in TICKERS else TICKERS[0], h))
    if not FAST_DEV_MODE and len(TICKERS) > 1:
        for h in CROSS_ASSET_HORIZONS:
            if h in EXPERIMENT_HORIZONS:
                for t in TICKERS:
                    tasks.append(("cross", t, h))
    return tasks


# ──────────────────────────────────────────────────────────────
# Resume bookkeeping, lock, aggregation
# ──────────────────────────────────────────────────────────────
def _signature():
    import hashlib, json
    keys = dict(profile=PROFILE, feature_set=FEATURE_SET, epochs=EPOCHS, dl=DL_SEEDS, ahi=AHI_SEEDS,
                base=BASELINE_DL_SEEDS, abl=ABLATION_SEEDS, roll=[ROLLING_START, ROLLING_STEP_DAYS, ROLLING_EPOCHS,
                ROLLING_SEEDS, ROLLING_INCLUDE_ARIMA_HYBRID], cross=CROSS_ASSET_SEEDS, purge=PURGE_TRAIN_OVERLAP,
                garch=GARCH_ESTIMAND, ratio=TRAIN_RATIO, lookback=LOOKBACK_WINDOW)
    return hashlib.md5(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _marker(kind, ticker, horizon):
    return os.path.join(RESULTS_DIR, "_done", f"{kind}_{_suf(ticker, horizon)}.json")


def _is_done(kind, ticker, horizon):
    import json
    p = _marker(kind, ticker, horizon)
    if not (RESUME and os.path.exists(p)):
        return False
    try:
        return json.load(open(p)).get("signature") == _signature()
    except Exception:
        return False


def _mark_done(kind, ticker, horizon, seconds):
    import json
    os.makedirs(os.path.dirname(_marker(kind, ticker, horizon)), exist_ok=True)
    json.dump({"signature": _signature(), "seconds": round(seconds, 1),
               "finished_utc": datetime.utcnow().isoformat()}, open(_marker(kind, ticker, horizon), "w"))


def _acquire_lock():
    """Hold an OS-level lock on results/.run.lock for the life of the orchestrator (released on crash too)."""
    path = os.path.join(RESULTS_DIR, ".run.lock")
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"\n[STOP] Another run is already writing to {RESULTS_DIR}. Close it first "
              f"(or set CRYPTO_HORSERACE_RESULTS_DIR to a different folder).")
        sys.exit(2)
    return fh


def aggregate(total_seconds: float = 0.0):
    """Merge the per-task files into the aggregate files the paper tables are built from."""
    import glob, json
    def cat(pattern, name):
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
        if files:
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            _save(df, name)
            return df
        return None
    sd = cat("seed_metrics_*_USD_h*.csv", "seed_metrics.csv")
    if sd is not None:
        agg = sd.groupby(["Ticker", "Horizon", "Model"]).agg(
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), RMSE_min=("RMSE", "min"),
            RMSE_max=("RMSE", "max"), N=("Seed", "count")).reset_index()
        _save(agg, "seed_summary.csv")
    cat("feature_audit_*_USD.csv", "feature_audit.csv")
    cat("sample_manifest_*_USD_h*.csv", "sample_manifest.csv")
    cat("gate_*_USD_h*.csv", "gate_lambda.csv")
    ca = cat("cross_asset_*_USD_h*.csv", "cross_asset_generalization.csv")
    if ca is not None:
        ind = []
        for _, r in ca.iterrows():
            f = os.path.join(RESULTS_DIR, f"comparison_table_{r['Holdout'].replace('-', '_')}_h{int(r['Horizon'])}.csv")
            v = np.nan
            if os.path.exists(f):
                c = pd.read_csv(f).set_index("Model")
                if "CNN-BiLSTM-Attn" in c.index:
                    v = float(c.loc["CNN-BiLSTM-Attn", "RMSE"])
            ind.append(v)
        ca["InDomain_RMSE"] = ind
        ca["Ratio_vs_InDomain"] = ca["RMSE"] / ca["InDomain_RMSE"]
        _save(ca, "cross_asset_generalization.csv")
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "_done", "*.json"))):
        kind, rest = os.path.basename(f)[:-5].split("_", 1)
        rows.append({"Task": os.path.basename(f)[:-5], "Kind": kind,
                     "Seconds": json.load(open(f)).get("seconds")})
    if rows:
        _save(pd.DataFrame(rows), "timing_report.csv")
    _write_manifest(total_seconds)
    try:
        import subprocess
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", "build_paper_inputs.py"), RESULTS_DIR],
                       check=False)
    except Exception as exc:                                        # pragma: no cover
        print(f"[aggregate] paper-input tables not built: {exc}")


def _write_manifest(elapsed_sec: float):
    import json, platform, importlib
    libs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "arch", "tensorflow"):
        try:
            libs[name] = importlib.import_module(name).__version__
        except Exception:
            libs[name] = "n/a"
    manifest = {
        "profile": PROFILE, "feature_set": FEATURE_SET, "tickers": TICKERS, "horizons": EXPERIMENT_HORIZONS,
        "dl_seeds": DL_SEEDS, "ahi_seeds": AHI_SEEDS, "baseline_dl_seeds": BASELINE_DL_SEEDS,
        "ablation_seeds": ABLATION_SEEDS, "ablation_horizons": ABLATION_HORIZONS, "rolling_seeds": ROLLING_SEEDS,
        "rolling": {"enabled": RUN_ROLLING_ORIGIN, "start": ROLLING_START, "step_days": ROLLING_STEP_DAYS,
                    "epochs": ROLLING_EPOCHS, "arima_hybrid": ROLLING_INCLUDE_ARIMA_HYBRID},
        "cross_asset": {"horizons": CROSS_ASSET_HORIZONS, "seeds": CROSS_ASSET_SEEDS},
        "purge_train_overlap": PURGE_TRAIN_OVERLAP, "garch_estimand": GARCH_ESTIMAND, "train_ratio": TRAIN_RATIO,
        "lookback": LOOKBACK_WINDOW, "epochs": EPOCHS, "mc_dropout_passes": MC_DROPOUT_PASSES,
        "elapsed_sec_last_invocation": round(elapsed_sec, 1), "python": platform.python_version(),
        "platform": platform.platform(), "libraries": libs, "signature": _signature(),
    }
    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────
def _run_worker_inline(kind, ticker, horizon):
    t0 = time.time()
    set_global_determinism(SEED)
    TASKS[kind](ticker, horizon)
    _mark_done(kind, ticker, horizon, time.time() - t0)


def _spawn(kind, ticker, horizon, log_path=None, threads=None):
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), "--task", kind, "--ticker", ticker, "--horizon", str(horizon)]
    env = dict(os.environ)
    if threads:
        env["TF_NUM_INTRAOP_THREADS"] = str(threads)
        env["TF_NUM_INTEROP_THREADS"] = "2"
    if log_path:
        with open(log_path, "w") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return subprocess.run(cmd, env=env).returncode


def orchestrate(workers: int = WORKERS):
    print("=" * 66 + "\n  CRYPTO VOLATILITY HORSERACE\n" + "=" * 66)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lock = _acquire_lock()
    t_start = time.time()
    print(f"[Config] profile={PROFILE} | features={FEATURE_SET} | DL seeds={DL_SEEDS} | AHI seeds={AHI_SEEDS} | "
          f"workers={workers} | resume={RESUME} | results={RESULTS_DIR}")
    print("\n> PHASE 1: data ingestion and preflight", flush=True)
    if set(TICKERS) < set(CORE_TICKERS):
        # A restricted run (CRYPTO_HORSERACE_TICKERS) must not rewrite the cached feature files:
        # collection saves only the assets it is given, which would drop the others from disk.
        print(f"[Phase 1] skipped: TICKERS={TICKERS} is a subset of {CORE_TICKERS}; cached feature files are used as they are.")
        collected = None
    else:
        collected = collect_feature_datasets(tickers=TICKERS, use_api=USE_API, use_google_trends=ENABLE_GOOGLE_TRENDS,
                                             use_onchain_api=ENABLE_ONCHAIN_API, use_market_api=USE_MARKET_API)
        preflight = run_data_preflight(
            market_df=collected["market_df"], sentiment_df=collected["sentiment_df"], onchain_df=collected["onchain_df"],
            source_metadata={"sentiment": collected["metadata"]["sentiment"], "onchain": collected["metadata"]["onchain"]})
        print(f"[Preflight] overall_pass={preflight['overall_pass']} failures={preflight['failure_count']}", flush=True)

    tasks = build_task_list()
    pending = [t for t in tasks if not _is_done(*t)]
    print(f"\n[Plan] {len(tasks)} tasks, {len(tasks) - len(pending)} already finished, {len(pending)} to run", flush=True)
    failures = []
    done_secs = []
    if workers <= 1:
        for i, (kind, t, h) in enumerate(pending, 1):
            eta = (np.mean(done_secs) * (len(pending) - i + 1) / 60) if done_secs else float("nan")
            print("\n" + "#" * 66 + f"\n# TASK {i}/{len(pending)}: {kind}  {t}  h={h}   "
                  f"(elapsed {_format_hms(time.time() - t_start)}, rough time left {eta:.0f} min)\n" + "#" * 66, flush=True)
            t0 = time.time()
            rc = _spawn(kind, t, h)
            if rc != 0:
                failures.append((kind, t, h, rc))
                print(f"[FAILED] {kind} {t} h={h} (exit {rc}); continuing with the next task", flush=True)
            else:
                done_secs.append(time.time() - t0)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
        threads = max(1, (os.cpu_count() or 4) // workers)
        heavy = {"ablation": 0, "rolling": 1, "combo": 2, "cross": 3}
        ordered = sorted(pending, key=lambda x: heavy[x[0]])
        print(f"[Parallel] {workers} workers, {threads} TensorFlow threads each; logs in {RESULTS_DIR}/logs/", flush=True)
        def work(task):
            kind, t, h = task
            t0 = time.time()
            rc = _spawn(kind, t, h, os.path.join(RESULTS_DIR, "logs", f"{kind}_{_suf(t, h)}.log"), threads)
            return task, rc, time.time() - t0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, t) for t in ordered]
            for n, fu in enumerate(as_completed(futs), 1):
                (kind, t, h), rc, secs = fu.result()
                print(f"[{n}/{len(ordered)}] {'ok' if rc == 0 else 'FAILED'}  {kind} {t} h={h}  "
                      f"{_format_hms(secs)}  (elapsed {_format_hms(time.time() - t_start)})", flush=True)
                if rc != 0:
                    failures.append((kind, t, h, rc))
    print("\n> AGGREGATING", flush=True)
    aggregate(time.time() - t_start)
    print("\n" + "=" * 66 + f"\n  DONE in {_format_hms(time.time() - t_start)}   results: {RESULTS_DIR}")
    if failures:
        print("  FAILED TASKS (re-run the same command to retry only these):")
        for f in failures:
            print("   ", f)
    print("=" * 66)
    lock.close()
    return 1 if failures else 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Crypto volatility horserace")
    ap.add_argument("--task", choices=sorted(TASKS), help="(internal) run a single task in this process")
    ap.add_argument("--ticker")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--workers", type=int, default=WORKERS, help="parallel worker processes (1 = sequential, progress bars on console)")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if a.task:
        _run_worker_inline(a.task, a.ticker, a.horizon)
        return 0
    if a.aggregate_only:
        aggregate()
        return 0
    return orchestrate(a.workers)


if __name__ == "__main__":
    sys.exit(main())

