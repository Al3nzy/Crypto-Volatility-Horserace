"""
main.py – Orchestrator for the Crypto Volatility Horserace.

Executes all four phases sequentially:
  Phase 1: ETL (ingest three pillars → fuse & preprocess)
  Phase 2: Statistical baselines (ARIMA, GARCH)
  Phase 3: Deep-learning model (CNN-BiLSTM-Attention)
  Phase 4: Benchmarking & visualisation
"""
import os
import sys
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
    ENABLE_GOOGLE_TRENDS,
    ENABLE_ONCHAIN_API,
    FULL_FEATURES,
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
    REPRO_SEEDS, WALKFORWARD_MIN_TRAIN, WALKFORWARD_STEP,
    USE_POOLED_DL_TRAINING, SEED,
)

# ── Phase 1 imports ──
from src.data_collection import collect_feature_datasets
from src.fuse_data import run_fusion_pipeline, run_fusion_pipeline_pooled
from src.data_quality import run_data_preflight

# ── Phase 2 imports ──
from src.baselines import run_all_baselines, get_arima_forecasts_for_residuals

# ── Phase 3 imports ──
from src.model_cnn_lstm import (
    build_model, train_model, extract_attention_weights,
)
from src.model_dl_baselines import run_dl_baseline_variant
from src.horserace_baselines import build_naive_persistence_result, build_svr_result

# ── Phase 4 imports ──
from src.evaluate import (
    rmse, build_comparison_table,
    plot_predictions_overlay, plot_attention_heatmap,
    plot_attention_per_feature, plot_training_curves,
    build_shock_window_table, build_dm_table, append_run_history,
    build_regime_table, build_backtest_table,
    mc_dropout_intervals, interval_coverage, sensitivity_perturbation_metrics,
)
from sklearn.metrics import mean_absolute_error

ABLATION_FEATURE_SETS = {
    "market_only": MARKET_FEATURE_COLS,
    "sentiment_only": ASSET_SENTIMENT_FEATURE_COLS,
    "onchain_only": ASSET_ONCHAIN_CORE_FEATURE_COLS,
    "market_macro": MARKET_FEATURE_COLS + ["Market_FearGreed"],
    "market_onchain": MARKET_FEATURE_COLS + ASSET_ONCHAIN_CORE_FEATURE_COLS,
    "market_sentiment": MARKET_FEATURE_COLS + ASSET_SENTIMENT_FEATURE_COLS,
    "full_multimodal": FULL_FEATURES,
}


def run_dl_experiment(
    feature_cols,
    experiment_name="full_multimodal",
    primary_ticker: str | None = None,
    forecast_horizon: int = FORECAST_HORIZON,
    pipeline=None,
):
    """Train/evaluate one DL experiment for a selected feature subset."""
    if pipeline is None:
        pipeline = run_fusion_pipeline(
            feature_cols=feature_cols,
            primary_ticker=primary_ticker,
            forecast_horizon=forecast_horizon,
        )
    X_train = pipeline["X_train"]
    X_test = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_test = pipeline["y_test"]
    tgt_scaler = pipeline["tgt_scaler"]
    test_dates = pipeline["test_dates"]

    model = build_model(X_train.shape[1], X_train.shape[2])
    history = train_model(model, X_train, y_train, X_test, y_test)

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = tgt_scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = tgt_scaler.inverse_transform(y_test).flatten()

    result = {
        "name": f"CNN-BiLSTM-Attn-{experiment_name}",
        "predictions": y_pred,
        "actual": y_actual,
        "dates": test_dates[:len(y_actual)],
        "rmse": rmse(y_actual, y_pred),
        "mae": mean_absolute_error(y_actual, y_pred),
    }

    return {
        "pipeline": pipeline,
        "model": model,
        "history": history,
        "result": result,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[2]),
    }


def run_cross_asset_generalization(feature_cols, holdout_ticker, forecast_horizon):
    """
    Train on train windows from all non-holdout assets; test on holdout asset.
    """
    train_X, train_y = [], []
    holdout = run_fusion_pipeline(
        feature_cols=feature_cols,
        primary_ticker=holdout_ticker,
        forecast_horizon=forecast_horizon,
    )
    for t in TICKERS:
        if t == holdout_ticker:
            continue
        p = run_fusion_pipeline(
            feature_cols=feature_cols,
            primary_ticker=t,
            forecast_horizon=forecast_horizon,
        )
        train_X.append(p["X_train"])
        train_y.append(p["y_train"])
    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_test = holdout["X_test"]
    y_test = holdout["y_test"]

    model = build_model(X_train.shape[1], X_train.shape[2])
    train_model(model, X_train, y_train)
    pred_scaled = model.predict(X_test, verbose=0)
    pred = holdout["tgt_scaler"].inverse_transform(pred_scaled).flatten()
    actual = holdout["tgt_scaler"].inverse_transform(y_test).flatten()
    return {
        "Model": "CrossAsset_CNN-BiLSTM-Attn",
        "Holdout": holdout_ticker,
        "Horizon": forecast_horizon,
        "RMSE": rmse(actual, pred),
        "MAE": mean_absolute_error(actual, pred),
    }


def run_walkforward_dl(pipeline, feature_cols, ticker, horizon):
    """
    Lightweight rolling-origin evaluation over test windows.
    """
    X_all = np.concatenate([pipeline["X_train"], pipeline["X_test"]], axis=0)
    y_all = np.concatenate([pipeline["y_train"], pipeline["y_test"]], axis=0)
    split_idx = len(pipeline["X_train"])
    preds, actual = [], []
    for start in range(max(WALKFORWARD_MIN_TRAIN, split_idx), len(X_all), WALKFORWARD_STEP):
        end = min(start + WALKFORWARD_STEP, len(X_all))
        if end <= start:
            continue
        model = build_model(X_all.shape[1], X_all.shape[2])
        train_model(model, X_all[:start], y_all[:start])
        p = model.predict(X_all[start:end], verbose=0).flatten()
        preds.extend(p.tolist())
        actual.extend(y_all[start:end].flatten().tolist())
    if not preds:
        return None
    p_inv = pipeline["tgt_scaler"].inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    a_inv = pipeline["tgt_scaler"].inverse_transform(np.array(actual).reshape(-1, 1)).flatten()
    return {
        "Ticker": ticker,
        "Horizon": horizon,
        "RMSE": rmse(a_inv, p_inv),
        "MAE": mean_absolute_error(a_inv, p_inv),
        "N": len(a_inv),
    }


def main():
    print("=" * 60)
    print("  CRYPTO VOLATILITY HORSERACE – MULTIMODAL BENCHMARKING")
    print("=" * 60)

    # ================================================================
    # PHASE 1 – ETL Pipeline
    # ================================================================
    print("\n> PHASE 1: Data Ingestion & Fusion")
    print("-" * 40)
    print(
        f"[Config] CRYPTO_HORSERACE_DATA_EXPERIMENT={ACTIVE_DATA_EXPERIMENT} | "
        f"CRYPTO_HORSERACE_POOLED_DL={USE_POOLED_DL_TRAINING}"
    )

    # 1a. Collect/build three datatype panels
    collected = collect_feature_datasets(
        tickers=TICKERS,
        use_api=True,
        use_google_trends=ENABLE_GOOGLE_TRENDS,
        use_onchain_api=ENABLE_ONCHAIN_API,
    )
    market_df = collected["market_df"]
    sentiment_df = collected["sentiment_df"]
    onchain_df = collected["onchain_df"]
    sentiment_meta = collected["metadata"]["sentiment"]
    onchain_meta = collected["metadata"]["onchain"]

    preflight = run_data_preflight(
        market_df=market_df,
        sentiment_df=sentiment_df,
        onchain_df=onchain_df,
        source_metadata={
            "sentiment": sentiment_meta,
            "onchain": onchain_meta,
        },
    )
    print(
        f"[Preflight] overall_pass={preflight['overall_pass']} "
        f"failures={preflight['failure_count']}"
    )

    # 1b. Run pipeline per horizon and ticker (multi-asset)
    ticker_suffix = lambda t: t.replace("-", "_")
    all_asset_results = {}
    cross_asset_rows = []
    walkforward_rows = []
    repro_rows = []

    for horizon in EXPERIMENT_HORIZONS:
        print("\n" + "=" * 60)
        print(f"  FORECAST HORIZON: {horizon} day(s)")
        print("=" * 60)
        for ticker in TICKERS:
            suf = f"{ticker_suffix(ticker)}_h{horizon}"
            print("\n" + "=" * 60)
            print(f"  ASSET: {ticker}")
            print("=" * 60)

            # Pipeline for this ticker
            pipeline = run_fusion_pipeline(
                feature_cols=FULL_FEATURES,
                primary_ticker=ticker,
                forecast_horizon=horizon,
            )
            X_test = pipeline["X_test"]
            test_dates = pipeline["test_dates"]
            feature_cols = pipeline["feature_cols"]
            merged_df = pipeline["merged_df"]

            # Phase 2 – Statistical Baselines
            print("\n> PHASE 2: Econometric Baselines")
            baseline_results = run_all_baselines(
                merged_df,
                forecast_horizon=horizon,
                include_har_gjr=RUN_HAR_GJR_BASELINES,
            )

            # Phase 3 – Deep Learning
            print("\n> PHASE 3: CNN-BiLSTM-Attention Model")
            dl_out = run_dl_experiment(
                FULL_FEATURES,
                experiment_name="full",
                primary_ticker=ticker,
                forecast_horizon=horizon,
                pipeline=pipeline,
            )
            model = dl_out["model"]
            history = dl_out["history"]
            dl_result = dl_out["result"]
            y_actual = dl_result["actual"]
            print(f"[CNN-BiLSTM-Attn] RMSE={dl_result['rmse']:.6f}  MAE={dl_result['mae']:.6f}")

            # Phase 4 – Benchmarking & Visualisation
            print("\n> PHASE 4: Benchmarking & Visualisation")

            # Hybrid ensemble
            arima_pred = np.asarray(baseline_results["ARIMA"]["predictions"])
            dl_pred = np.asarray(dl_result["predictions"])
            n_align = min(len(arima_pred), len(dl_pred), len(y_actual))
            hybrid_pred = 0.5 * arima_pred[:n_align] + 0.5 * dl_pred[:n_align]
            actual_align = np.asarray(y_actual)[:n_align]
            dates_align = np.asarray(test_dates)[:n_align]
            hybrid_result = {
                "name": "Hybrid (ARIMA+DL)",
                "predictions": hybrid_pred,
                "actual": actual_align,
                "dates": dates_align,
                "rmse": rmse(actual_align, hybrid_pred),
                "mae": mean_absolute_error(actual_align, hybrid_pred),
            }
            print(f"[Hybrid] RMSE={hybrid_result['rmse']:.6f}  MAE={hybrid_result['mae']:.6f}")

            # Residual hybrid
            vol_series = merged_df["Volatility"]
            arima_full_pred, vol_vals = get_arima_forecasts_for_residuals(
                vol_series, forecast_horizon=horizon
            )
            residuals = vol_vals - arima_full_pred
            residuals = np.nan_to_num(residuals, nan=0.0)
            residual_pipeline = run_fusion_pipeline(
                feature_cols=FULL_FEATURES,
                residual_target=residuals,
                primary_ticker=ticker,
                forecast_horizon=horizon,
            )
            X_train_r, X_test_r = residual_pipeline["X_train"], residual_pipeline["X_test"]
            y_train_r, y_test_r = residual_pipeline["y_train"], residual_pipeline["y_test"]
            tgt_scaler_r = residual_pipeline["tgt_scaler"]
            test_dates_r = residual_pipeline["test_dates"]

            model_resid = build_model(X_train_r.shape[1], X_train_r.shape[2])
            train_model(model_resid, X_train_r, y_train_r, X_test_r, y_test_r)

            resid_pred_scaled = model_resid.predict(X_test_r, verbose=0)
            resid_pred = tgt_scaler_r.inverse_transform(resid_pred_scaled).flatten()
            n_win = len(residual_pipeline["X_train"]) + len(X_test_r)
            split_idx = int(n_win * TRAIN_RATIO)
            start_idx = split_idx + LOOKBACK_WINDOW + horizon - 1
            arima_test_for_resid = arima_full_pred[start_idx : start_idx + len(resid_pred)]
            actual_r = vol_vals[start_idx : start_idx + len(resid_pred)]
            residual_hybrid_pred = arima_test_for_resid + resid_pred
            residual_hybrid_result = {
                "name": "Residual Hybrid (ARIMA+DL-resid)",
                "predictions": residual_hybrid_pred,
                "actual": actual_r,
                "dates": test_dates_r,
                "rmse": rmse(actual_r, residual_hybrid_pred),
                "mae": mean_absolute_error(actual_r, residual_hybrid_pred),
            }
            print(f"[Residual Hybrid] RMSE={residual_hybrid_result['rmse']:.6f}  MAE={residual_hybrid_result['mae']:.6f}")

            horserace_ml = {}
            if RUN_NAIVE_PERSISTENCE_BASELINE:
                horserace_ml["Naive_Persistence"] = build_naive_persistence_result(
                    pipeline, horizon
                )
            if RUN_SVR_BASELINE:
                print("\n[Horserace] SVR (RBF) on flattened windows …")
                horserace_ml["SVR (RBF)"] = build_svr_result(pipeline)
            if RUN_DL_ABLATION_BASELINES:
                for variant in ("cnn_only", "lstm_only", "gru_only"):
                    print(f"\n[Horserace] DL baseline: {variant} …")
                    r = run_dl_baseline_variant(pipeline, variant)
                    horserace_ml[r["name"]] = {
                        k: v for k, v in r.items() if k != "model"
                    }

            all_results = {
                **horserace_ml,
                "ARIMA": baseline_results["ARIMA"],
                "GARCH": baseline_results["GARCH"],
                "CNN-BiLSTM-Attn": dl_result,
                "Hybrid (ARIMA+DL)": hybrid_result,
                "Residual Hybrid (ARIMA+DL-resid)": residual_hybrid_result,
            }
            for key in ("HAR-RV", "GJR-GARCH"):
                if key in baseline_results:
                    all_results[key] = baseline_results[key]
            all_asset_results[f"{ticker}_h{horizon}"] = all_results

            # Comparison table, stress windows, regime table, DM tests
            build_comparison_table(all_results, output_suffix=suf)
            shock_df = build_shock_window_table(all_results, SHOCK_WINDOWS, output_suffix=suf)
            build_regime_table(all_results, output_suffix=suf)
            build_dm_table(all_results, output_suffix=suf)
            build_backtest_table(all_results, TRANSACTION_COST_BPS, output_suffix=suf)

            # Plots (per-asset files)
            plot_predictions_overlay(all_results, test_dates, output_suffix=suf)
            plot_training_curves(history, output_suffix=suf)

            # Attention + uncertainty
            print("\n[Attention] Extracting attention weights …")
            attn_weights = extract_attention_weights(model, X_test[:50])
            plot_attention_heatmap(
                attn_weights, feature_cols,
                sample_idx=0, head_idx=0,
                title_suffix=" (Sample 0, Head 0)",
                output_suffix=suf,
            )
            ftx_start = pd.Timestamp("2022-11-05")
            ftx_mask = test_dates[:len(y_actual)] >= ftx_start
            if ftx_mask.any():
                ftx_idx = np.argmax(ftx_mask)
                if ftx_idx < 50:
                    plot_attention_heatmap(
                        attn_weights, feature_cols,
                        sample_idx=ftx_idx, head_idx=0,
                        title_suffix=" (FTX Crash Window)",
                        output_suffix=suf,
                    )
            plot_attention_per_feature(attn_weights, feature_cols, X_test[:50], output_suffix=suf)
            _, mc_l, mc_u = mc_dropout_intervals(
                model, X_test, pipeline["tgt_scaler"], passes=MC_DROPOUT_PASSES, alpha=MC_INTERVAL_ALPHA
            )
            interval_stats = interval_coverage(y_actual, mc_l, mc_u)
            pd.DataFrame([{
                "Ticker": ticker, "Horizon": horizon,
                "Coverage": interval_stats["coverage"],
                "AvgWidth": interval_stats["avg_width"],
            }]).to_csv(os.path.join(RESULTS_DIR, f"interval_metrics_{suf}.csv"), index=False)

            # Sensitivity (missingness + noise on test windows)
            sens_rows = []
            rng = np.random.default_rng(SEED)
            for mr in SENSITIVITY_MISSING_RATES:
                Xp = X_test.copy()
                mask = rng.random(Xp.shape) < mr
                Xp[mask] = 0.0
                yp = pipeline["tgt_scaler"].inverse_transform(model.predict(Xp, verbose=0)).flatten()
                row = sensitivity_perturbation_metrics("CNN-BiLSTM-Attn", y_actual, dl_result["predictions"], yp, f"missing_{mr:.2f}")
                if row:
                    sens_rows.append(row)
            for nr in SENSITIVITY_NOISE_STD_RATES:
                Xp = X_test + rng.normal(0, nr, size=X_test.shape)
                yp = pipeline["tgt_scaler"].inverse_transform(model.predict(Xp, verbose=0)).flatten()
                row = sensitivity_perturbation_metrics("CNN-BiLSTM-Attn", y_actual, dl_result["predictions"], yp, f"noise_{nr:.2f}")
                if row:
                    sens_rows.append(row)
            if sens_rows:
                pd.DataFrame(sens_rows).to_csv(os.path.join(RESULTS_DIR, f"sensitivity_metrics_{suf}.csv"), index=False)

            # Optional ablations (only for PRIMARY_TICKER by default to save compute)
            run_ablations_here = RUN_ABLATIONS and (
                RUN_ABLATIONS_FOR_ALL_TICKERS or ticker == PRIMARY_TICKER
            )
            if run_ablations_here:
                print("\n[Ablation] Running DL ablation suite …")
                ablation_rows = []
                ablation_results_for_shock = {}
                for ab_name, ab_cols in ABLATION_FEATURE_SETS.items():
                    # skip feature-set variants with unavailable columns
                    if not set(ab_cols).issubset(set(feature_cols)):
                        continue
                    out = run_dl_experiment(
                        ab_cols,
                        experiment_name=ab_name,
                        primary_ticker=ticker,
                        forecast_horizon=horizon,
                    )
                    r = out["result"]
                    ablation_rows.append({
                        "Experiment": ab_name,
                        "Ticker": ticker,
                        "NumFeatures": len(ab_cols),
                        "RMSE": r["rmse"],
                        "MAE": r["mae"],
                    })
                    ablation_results_for_shock[f"DL-{ab_name}"] = r
                    print(f"[Ablation] {ab_name}: RMSE={r['rmse']:.6f}  MAE={r['mae']:.6f}")
                if ablation_rows:
                    ab_path = os.path.join(RESULTS_DIR, f"ablation_results_{suf}.csv")
                    pd.DataFrame(ablation_rows).sort_values("RMSE").to_csv(ab_path, index=False)
                    ab_shock_df = build_shock_window_table(
                        ablation_results_for_shock,
                        SHOCK_WINDOWS,
                        output_suffix=f"ablation_{suf}",
                    )
                    if not ab_shock_df.empty:
                        ab_shock_df.to_csv(
                            os.path.join(RESULTS_DIR, f"ablation_shock_window_metrics_{suf}.csv"),
                            index=False,
                        )
                    print(f"[Ablation] Saved -> {ab_path}")

            # Walk-forward DL metrics
            wf = run_walkforward_dl(pipeline, FULL_FEATURES, ticker, horizon)
            if wf:
                walkforward_rows.append(wf)

            # Reproducibility matrix (subset for compute)
            if ticker == PRIMARY_TICKER and horizon == FORECAST_HORIZON:
                for seed in REPRO_SEEDS:
                    np.random.seed(seed)
                    out_seed = run_dl_experiment(
                        FULL_FEATURES, experiment_name=f"seed_{seed}",
                        primary_ticker=ticker, forecast_horizon=horizon
                    )
                    repro_rows.append({
                        "Ticker": ticker, "Horizon": horizon, "Seed": seed,
                        "RMSE": out_seed["result"]["rmse"], "MAE": out_seed["result"]["mae"],
                    })

            # Run history (one row per ticker)
            run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            run_row = {
                "run_ts_utc": run_ts,
                "market_interval": MARKET_INTERVAL,
                "tickers": "|".join(TICKERS),
                "primary_ticker": ticker,
                "forecast_horizon": horizon,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "n_train_samples": dl_out["n_train_samples"],
                "n_test_samples": dl_out["n_test_samples"],
                "n_features": dl_out["n_features"],
                "arima_rmse": all_results["ARIMA"]["rmse"],
                "arima_mae": all_results["ARIMA"]["mae"],
                "garch_rmse": all_results["GARCH"]["rmse"],
                "garch_mae": all_results["GARCH"]["mae"],
                "dl_rmse": all_results["CNN-BiLSTM-Attn"]["rmse"],
                "dl_mae": all_results["CNN-BiLSTM-Attn"]["mae"],
                "hybrid_rmse": all_results["Hybrid (ARIMA+DL)"]["rmse"],
                "hybrid_mae": all_results["Hybrid (ARIMA+DL)"]["mae"],
                "residual_hybrid_rmse": all_results["Residual Hybrid (ARIMA+DL-resid)"]["rmse"],
                "residual_hybrid_mae": all_results["Residual Hybrid (ARIMA+DL-resid)"]["mae"],
                "winner": min(all_results.keys(), key=lambda k: all_results[k]["rmse"]),
                "shock_rows": int(len(shock_df)) if isinstance(shock_df, pd.DataFrame) else 0,
            }
            append_run_history(run_row, RUN_HISTORY_FILE)

            # Cross-asset generalization (per holdout)
            if horizon == FORECAST_HORIZON:
                cross_asset_rows.append(
                    run_cross_asset_generalization(FULL_FEATURES, ticker, horizon)
                )

    if USE_POOLED_DL_TRAINING:
        pooled_rows = []
        for horizon in EXPERIMENT_HORIZONS:
            pooled = run_fusion_pipeline_pooled(
                feature_cols=FULL_FEATURES,
                forecast_horizon=horizon,
            )
            model_p = build_model(pooled["X_train"].shape[1], pooled["X_train"].shape[2])
            train_model(
                model_p,
                pooled["X_train"],
                pooled["y_train"],
                pooled["X_test"],
                pooled["y_test"],
            )
            pred_s = model_p.predict(pooled["X_test"], verbose=0)
            pred = pooled["tgt_scaler"].inverse_transform(pred_s).flatten()
            actual = pooled["tgt_scaler"].inverse_transform(pooled["y_test"]).flatten()
            pooled_rows.append({
                "Horizon": horizon,
                "N_train": int(pooled["X_train"].shape[0]),
                "N_test": int(pooled["X_test"].shape[0]),
                "N_features": int(pooled["X_train"].shape[2]),
                "N_assets_pooled": int(pooled["n_assets_pooled"]),
                "RMSE": rmse(actual, pred),
                "MAE": mean_absolute_error(actual, pred),
            })
        pd.DataFrame(pooled_rows).to_csv(
            os.path.join(RESULTS_DIR, "pooled_dl_metrics.csv"), index=False
        )
        print(f"[Pooled DL] Saved -> {os.path.join(RESULTS_DIR, 'pooled_dl_metrics.csv')}")

    # Aggregate advanced reports
    if cross_asset_rows:
        pd.DataFrame(cross_asset_rows).to_csv(os.path.join(RESULTS_DIR, "cross_asset_generalization.csv"), index=False)
    if walkforward_rows:
        pd.DataFrame(walkforward_rows).to_csv(os.path.join(RESULTS_DIR, "walkforward_metrics.csv"), index=False)
    if repro_rows:
        rdf = pd.DataFrame(repro_rows)
        rdf.to_csv(os.path.join(RESULTS_DIR, "reproducibility_runs.csv"), index=False)
        agg = rdf.groupby(["Ticker", "Horizon"]).agg(
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
            N=("Seed", "count"),
        ).reset_index()
        agg.to_csv(os.path.join(RESULTS_DIR, "reproducibility_summary.csv"), index=False)

    # Summary
    print("\n" + "=" * 60)
    print("  HORSERACE COMPLETE (MULTI-ASSET)")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Assets: {', '.join(TICKERS)}")
    print("=" * 60)

    return all_asset_results


if __name__ == "__main__":
    main()
