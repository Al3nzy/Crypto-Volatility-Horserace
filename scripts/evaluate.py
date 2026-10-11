"""
Evaluation & Benchmarking Module (Section 3.4).

Computes metrics, builds comparison table, and generates all visualisations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import (RESULTS_DIR, LOOKBACK_WINDOW, DM_HLN_CORRECTION, BOOTSTRAP_RESAMPLES,
                    BOOTSTRAP_BLOCK_MIN, SEED)

plt.rcParams.update({
    "figure.figsize": (14, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def directional_accuracy(y_true, y_pred, eps: float = 1e-12) -> float:
    """
    Fraction of consecutive steps where sign(Δactual) matches sign(Δpred).
    Only counts steps where |Δactual| > eps (avoid zero-change noise).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))
    if n < 2:
        return float("nan")
    dt = np.diff(y_true[:n])
    dp = np.diff(y_pred[:n])
    valid = np.abs(dt) > eps
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(dt[valid]) == np.sign(dp[valid])))


def directional_counts(y_true, y_pred, eps: float = 1e-12):
    """(hits, n) behind directional_accuracy, for a binomial test against 0.5."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))
    if n < 2:
        return 0, 0
    dt, dp = np.diff(y_true[:n]), np.diff(y_pred[:n])
    valid = np.abs(dt) > eps
    return int(np.sum(np.sign(dt[valid]) == np.sign(dp[valid]))), int(valid.sum())


def test_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    n = min(len(y_true), len(y_pred))
    if n < 2:
        return float("nan")
    return float(r2_score(y_true[:n], y_pred[:n]))


# ──────────────────────────────────────────────────────────────
# Comparative table
# ──────────────────────────────────────────────────────────────
def build_comparison_table(results: dict, output_suffix: str = "") -> pd.DataFrame:
    """
    results: dict of {model_name: {"rmse", "mae", "predictions", "actual", ...}}
    Adds DoC (directional accuracy) and R2 on the test slice.
    output_suffix: e.g. "_BTC_USD" for per-asset files in multi-asset runs.
    """
    rows = []
    for name, res in results.items():
        pred = np.asarray(res.get("predictions", []))
        act = np.asarray(res.get("actual", []))
        doc = directional_accuracy(act, pred)
        r2v = test_r2(act, pred)
        hits, n_doc = directional_counts(act, pred)
        from scipy import stats as _st
        doc_p = float(_st.binomtest(hits, n_doc, 0.5).pvalue) if n_doc > 0 else float("nan")
        rows.append({
            "Model": name,
            "RMSE": res["rmse"],
            "MAE": res["mae"],
            "DoC": doc,
            "R2": r2v,
            "DoC_N": n_doc,
            "DoC_p_vs_0.5": doc_p,
        })
    df = pd.DataFrame(rows).set_index("Model")
    df.sort_values("RMSE", inplace=True)

    suf = f"_{output_suffix}" if output_suffix else ""
    print("\n" + "=" * 50)
    print(f"       COMPARATIVE PERFORMANCE TABLE{suf}")
    print("=" * 50)
    print(df.to_string())
    print("=" * 50 + "\n")

    table_path = os.path.join(RESULTS_DIR, f"comparison_table{suf}.csv")
    df.to_csv(table_path)
    return df


def build_shock_window_table(results: dict, shock_windows: list, output_suffix: str = "") -> pd.DataFrame:
    """
    Compute RMSE/MAE inside named stress windows for each model.
    shock_windows: [(start_date, end_date, label), ...]
    Prints winner per window for research narrative (DL vs baselines during crises).
    """
    rows = []
    for model_name, res in results.items():
        dates = pd.to_datetime(res.get("dates", []))
        preds = np.asarray(res.get("predictions", []))
        actual = np.asarray(res.get("actual", []))
        n = min(len(dates), len(preds), len(actual))
        if n == 0:
            continue

        dates = dates[:n]
        preds = preds[:n]
        actual = actual[:n]

        for start, end, label in shock_windows:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            if not mask.any():
                continue
            rows.append({
                "Window": label,
                "Model": model_name,
                "RMSE": rmse(actual[mask], preds[mask]),
                "MAE": mean_absolute_error(actual[mask], preds[mask]),
                "N": int(mask.sum()),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["Window", "RMSE"], inplace=True)
        suf = f"_{output_suffix}" if output_suffix else ""
        out_path = os.path.join(RESULTS_DIR, f"shock_window_metrics{suf}.csv")
        out.to_csv(out_path, index=False)
        print(f"[Eval] Saved shock-window metrics -> {out_path}")
        # Print winner per shock window (for research narrative)
        print("\n--- SHOCK-WINDOW WINNERS (crisis periods) ---")
        for label in out["Window"].unique():
            sub = out[out["Window"] == label]
            winner = sub.iloc[0]
            print(f"  {label}: {winner['Model']} (RMSE={winner['RMSE']:.6f})")
        print("--------------------------------------------\n")
    return out


def run_diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy on squared-error loss.

    e1, e2: squared forecast errors of the two models (same length).
    The long-run variance of the loss differential uses a Bartlett kernel with
    truncation lag h-1, the textbook choice for h-step-ahead forecasts (the
    differential of an h-step error is MA(h-1) under the null). With
    DM_HLN_CORRECTION the Harvey-Leybourne-Newbold small-sample factor
        sqrt((n + 1 - 2h + h(h-1)/n) / n)
    rescales the statistic and the Student-t(n-1) reference distribution is
    used. Returns dm_stat, two-sided p_value, and the 5% significance flag.
    """
    from scipy import stats

    d = np.asarray(e1, dtype=float) - np.asarray(e2, dtype=float)
    n = len(d)
    if n < 3:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant": False}

    d_bar = float(np.mean(d))
    dc = d - d_bar
    gamma_0 = float(np.mean(dc ** 2))
    max_lag = min(max(h - 1, 0), n - 1)
    hac_sum = 0.0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1)
        hac_sum += 2.0 * w * float(np.mean(dc[k:] * dc[:-k]))
    var_d = gamma_0 + hac_sum
    if var_d <= 0:
        var_d = 1e-12

    dm_stat = d_bar / np.sqrt(var_d / n)
    if DM_HLN_CORRECTION:
        corr = (n + 1 - 2 * h + h * (h - 1) / n) / n
        dm_stat *= np.sqrt(max(corr, 1e-12))
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=max(1, n - 1)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "significant": bool(p_value < 0.05)}


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjustment of a vector of p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def build_dm_table(results: dict, forecast_horizon: int = 1, output_suffix: str = "") -> pd.DataFrame:
    """
    Pairwise Diebold-Mariano tests (squared errors) for all model pairs.
    Negative DM stat means model1 is better; positive means model2 is better.
    forecast_horizon: number of steps ahead (sets Newey-West HAC lag truncation).
    """
    names = list(results.keys())
    pairs = []
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            a1 = np.asarray(results[m1]["actual"])
            a2 = np.asarray(results[m2]["actual"])
            p1 = np.asarray(results[m1]["predictions"])
            p2 = np.asarray(results[m2]["predictions"])
            n = min(len(a1), len(a2), len(p1), len(p2))
            # Every model must be scored against the same realized series.
            if not np.allclose(a1[:n], a2[:n], rtol=1e-6, atol=1e-10):
                raise ValueError(f"DM: realized series of {m1} and {m2} differ; refusing to test.")
            e1 = (a1[:n] - p1[:n]) ** 2
            e2 = (a1[:n] - p2[:n]) ** 2
            dm = run_diebold_mariano(e1, e2, h=forecast_horizon)
            better = m1 if dm["dm_stat"] < 0 else m2
            pairs.append({
                "Model_A": m1, "Model_B": m2,
                "DM_stat": dm["dm_stat"], "p_value": dm["p_value"],
                "Significant": dm["significant"], "Better": better,
            })
    df = pd.DataFrame(pairs)
    if not df.empty:
        df["p_holm"] = holm_adjust(df["p_value"].values)
        df["Significant_holm"] = df["p_holm"] < 0.05
        print("\n--- DIEBOLD-MARIANO TESTS (H0: equal predictive accuracy) ---")
        for _, row in df.iterrows():
            sig = " *" if row["Significant"] else ""
            print(f"  {row['Model_A']} vs {row['Model_B']}: DM={row['DM_stat']:.3f}, p={row['p_value']:.4f}{sig} -> {row['Better']} better")
        print("  (* p<0.05)")
        print("----------------------------------------------------------------\n")
        suf = f"_{output_suffix}" if output_suffix else ""
        df.to_csv(os.path.join(RESULTS_DIR, f"dm_tests{suf}.csv"), index=False)
    return df


def append_run_history(run_row: dict, history_file: str):
    """Append one run summary row to persistent CSV history."""
    row_df = pd.DataFrame([run_row])
    if os.path.exists(history_file):
        prev = pd.read_csv(history_file)
        combined = pd.concat([prev, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(history_file, index=False)
    print(f"[Eval] Appended run history -> {history_file}")


# ──────────────────────────────────────────────────────────────
# Prediction overlay plot
# ──────────────────────────────────────────────────────────────
def plot_predictions_overlay(results: dict, test_dates=None, save=True, output_suffix: str = ""):
    """
    Overlay actual volatility vs. all model predictions.
    Each model uses its own 'dates' array to handle different lengths.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot actual from each model (use the longest one for the backdrop)
    actual_plotted = False
    for name, res in results.items():
        model_dates = res.get("dates")
        if model_dates is None:
            continue
        actual = res.get("actual")
        n = min(len(model_dates), len(actual)) if actual is not None else 0
        if actual is not None and not actual_plotted:
            ax.plot(model_dates[:n], actual[:n], color="black",
                    linewidth=1.4, label="Actual Volatility")
            actual_plotted = True

    colors = {
        "ARIMA": "#1f77b4", "GARCH": "#ff7f0e", "CNN-BiLSTM-Attn": "#2ca02c",
        "Hybrid (ARIMA+DL)": "#9467bd",
        "Residual Hybrid (ARIMA+DL-resid)": "#8c564b",
    }
    for name, res in results.items():
        model_dates = res.get("dates")
        preds = res["predictions"]
        if model_dates is None:
            continue
        n = min(len(model_dates), len(preds))
        c = colors.get(name, "grey")
        ax.plot(model_dates[:n], preds[:n], linewidth=1.0, alpha=0.85,
                label=name, color=c)

    # Annotate crash windows
    crash_windows = [
        ("2022-05-05", "2022-05-20", "LUNA"),
        ("2022-11-05", "2022-11-20", "FTX"),
    ]
    # Only annotate stress windows that overlap the plotted forecast range;
    # otherwise axvspan silently stretches the x-axis to dates with no data.
    all_dates = []
    for res in results.values():
        d = res.get("dates")
        if d is not None and len(d):
            all_dates.append(pd.to_datetime(d))
    lo = min(d.min() for d in all_dates) if all_dates else None
    hi = max(d.max() for d in all_dates) if all_dates else None
    for start, end, label in crash_windows:
        try:
            s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
            if lo is None or e_ts < lo or s_ts > hi:
                continue
            ax.axvspan(s_ts, e_ts, alpha=0.12, color="red")
            ax.text(s_ts, ax.get_ylim()[1] * 0.95, label,
                    fontsize=9, color="red", fontweight="bold")
        except Exception:
            pass

    title = "Volatility Forecast: Actual vs. Model Predictions"
    if output_suffix:
        title += f" ({output_suffix})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (σ)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save:
        suf = f"_{output_suffix}" if output_suffix else ""
        path = os.path.join(RESULTS_DIR, f"predictions_overlay{suf}.png")
        fig.savefig(path, dpi=150)
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Attention heatmap
# ──────────────────────────────────────────────────────────────
def plot_attention_heatmap(attn_weights: np.ndarray,
                           feature_cols: list,
                           sample_idx: int = 0,
                           head_idx: int = 0,
                           title_suffix: str = "",
                           save=True,
                           output_suffix: str = ""):
    """
    Visualise attention weights for a single sample & head.
    attn_weights shape: (batch, heads, window, window)
    We average across the query dimension to get per-timestep importance.
    """
    w = attn_weights[sample_idx, head_idx]          # (window, window)
    avg_importance = w.mean(axis=0)                  # (window,)

    # Build a matrix: rows = timesteps, cols = feature groups
    # For a richer heatmap we tile importance across features
    n_steps = len(avg_importance)
    n_feats = len(feature_cols)

    # Create (timestep x feature) importance via outer product with uniform
    heatmap_data = np.outer(avg_importance, np.ones(n_feats))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=feature_cols,
        yticklabels=[f"t-{n_steps - 1 - i}" for i in range(n_steps)],
        cmap="YlOrRd", linewidths=0.3, ax=ax,
    )
    ax.set_title(f"Attention Weights Heatmap {title_suffix}", fontsize=13)
    ax.set_xlabel("Features")
    ax.set_ylabel("Lookback Timestep")
    plt.tight_layout()

    if save:
        suf = f"_{output_suffix}" if output_suffix else ""
        fname = f"attention_heatmap{title_suffix.replace(' ', '_')}{suf}.png"
        path = os.path.join(RESULTS_DIR, fname)
        fig.savefig(path, dpi=150)
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)


def plot_attention_per_feature(attn_weights: np.ndarray,
                                feature_cols: list,
                                X_test_window: np.ndarray,
                                sample_idx: int = 0,
                                save=True,
                                output_suffix: str = ""):
    """
    More granular: weight each feature's value by its attention score
    to show which pillar the model relied on.
    """
    # Average across all heads and query positions
    # attn_weights: (batch, heads, window, window)
    avg_w = attn_weights[sample_idx].mean(axis=0).mean(axis=0)  # (window,)

    # Weighted feature importance
    sample = X_test_window[sample_idx]  # (window, features)
    weighted = sample * avg_w[:, None]  # broadcast
    feat_importance = np.abs(weighted).mean(axis=0)  # (features,)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(feature_cols, feat_importance, color=sns.color_palette("viridis", len(feature_cols)))
    ax.set_xlabel("Attention-Weighted Feature Importance")
    ax.set_title("Feature Importance via Attention Mechanism")
    plt.tight_layout()

    if save:
        suf = f"_{output_suffix}" if output_suffix else ""
        path = os.path.join(RESULTS_DIR, f"feature_importance{suf}.png")
        fig.savefig(path, dpi=150)
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────
def plot_training_curves(history, save=True, output_suffix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss (MSE)")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].legend()

    plt.tight_layout()
    if save:
        suf = f"_{output_suffix}" if output_suffix else ""
        path = os.path.join(RESULTS_DIR, f"training_curves{suf}.png")
        fig.savefig(path, dpi=150)
        print(f"[Plot] Saved -> {path}")
    plt.close(fig)


def build_regime_table(results: dict, merged_df: pd.DataFrame | None = None, output_suffix: str = "") -> pd.DataFrame:
    """
    Regime-aware evaluation using test-period realized volatility and market returns:
    - calm: vol <= p33
    - high_vol: vol > p67 (top tercile of the evaluation sample; not a named event)
    - bull: p33 < vol <= p67 AND market return >= 0
    - bear: p33 < vol <= p67 AND market return < 0

    merged_df: the fused DataFrame with a Log_Return column for proper
    bull/bear classification using actual market direction (not Δvolatility).
    """
    # Pre-compute market returns from merged_df if available
    market_ret_series = None
    if merged_df is not None and "Log_Return" in merged_df.columns:
        market_ret_series = merged_df["Log_Return"]

    rows = []
    for model_name, res in results.items():
        dates = pd.to_datetime(res.get("dates", []))
        preds = np.asarray(res.get("predictions", []))
        actual = np.asarray(res.get("actual", []))
        n = min(len(dates), len(preds), len(actual))
        if n < 5:
            continue
        dates = dates[:n]
        preds = preds[:n]
        actual = actual[:n]

        # Use market Log_Return for bull/bear direction (not Δvolatility)
        if market_ret_series is not None:
            ret = market_ret_series.reindex(dates).fillna(0.0).values
        else:
            # Fallback: use change in volatility (less accurate but functional)
            actual_series = pd.Series(actual, index=dates)
            ret = actual_series.diff().fillna(0.0).values

        q33, q67 = np.quantile(actual, [0.33, 0.67])

        regime = np.where(
            actual <= q33, "calm",
            np.where(actual > q67, "high_vol", np.where(ret >= 0, "bull", "bear"))
        )
        for rg in ["calm", "bull", "bear", "high_vol"]:
            mask = regime == rg
            if mask.sum() == 0:
                continue
            rows.append({
                "Regime": rg,
                "Model": model_name,
                "RMSE": rmse(actual[mask], preds[mask]),
                "MAE": mean_absolute_error(actual[mask], preds[mask]),
                "N": int(mask.sum()),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["Regime", "RMSE"], inplace=True)
        suf = f"_{output_suffix}" if output_suffix else ""
        path = os.path.join(RESULTS_DIR, f"regime_metrics{suf}.csv")
        out.to_csv(path, index=False)
        print(f"[Eval] Saved regime metrics -> {path}")
    return out


def compute_backtest_metrics(
    actual_vol: np.ndarray,
    pred_vol: np.ndarray,
    cost_bps: float = 10.0,
):
    """
    Simple volatility timing utility:
    target exposure ~ inverse predicted volatility.
    """
    eps = 1e-8
    n = min(len(actual_vol), len(pred_vol))
    if n < 3:
        return {}
    rv = np.asarray(actual_vol[:n])
    pv = np.asarray(pred_vol[:n])

    signal = 1.0 / np.maximum(pv, eps)
    signal = signal / np.nanmean(signal)
    signal = np.clip(signal, 0.0, 2.0)

    # Proxy return from realized vol changes (thesis utility proxy, not PnL truth)
    proxy_ret = -np.diff(rv, prepend=rv[0])
    strat_ret_gross = signal * proxy_ret
    turnover = np.abs(np.diff(signal, prepend=signal[0]))
    costs = turnover * (cost_bps / 10000.0)
    strat_ret = strat_ret_gross - costs

    mean_r = np.nanmean(strat_ret)
    std_r = np.nanstd(strat_ret) + eps
    # Sortino: root-mean-squared deviation from zero target (not std of negatives)
    downside_std = np.sqrt(np.nanmean(np.minimum(strat_ret, 0) ** 2)) + eps

    equity = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.maximum(peak, eps)

    return {
        "Sharpe": float(mean_r / std_r * np.sqrt(252)),
        "Sortino": float(mean_r / downside_std * np.sqrt(252)),
        "MaxDrawdown": float(drawdown.min()),
        "AvgTurnover": float(np.nanmean(turnover)),
    }


def build_backtest_table(results: dict, cost_bps: float, output_suffix: str = "") -> pd.DataFrame:
    rows = []
    for model_name, res in results.items():
        m = compute_backtest_metrics(
            np.asarray(res.get("actual", [])),
            np.asarray(res.get("predictions", [])),
            cost_bps=cost_bps,
        )
        if not m:
            continue
        rows.append({"Model": model_name, **m})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values("Sharpe", ascending=False, inplace=True)
        suf = f"_{output_suffix}" if output_suffix else ""
        path = os.path.join(RESULTS_DIR, f"backtest_metrics{suf}.csv")
        out.to_csv(path, index=False)
        print(f"[Eval] Saved backtest metrics -> {path}")
    return out


def mc_dropout_intervals(model, X_test: np.ndarray, target_scaler, passes: int = 50, alpha: float = 0.10,
                         offset=None, gate: float = 1.0):
    """
    Monte-Carlo dropout intervals for DL predictions.
    Keeps Dropout active for stochastic sampling while freezing
    BatchNormalization to prevent test-data leakage into moving statistics.
    """
    import tensorflow as tf

    preds = []
    # A Functional model's layer list is not a sequential pipeline: layers may
    # have multiple inputs (e.g. MultiHeadAttention) and skip connections.
    # Call the model itself so Keras follows its graph, while freezing BatchNorm
    # layers to keep training=True from updating their moving statistics.
    batch_norm_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    trainable_states = [layer.trainable for layer in batch_norm_layers]
    try:
        for layer in batch_norm_layers:
            layer.trainable = False

        @tf.function
        def _mc_forward(x):
            return model(x, training=True)  # Dropout ON; BatchNorm frozen above.

        for _ in range(passes):
            y = _mc_forward(X_test).numpy()
            y = target_scaler.inverse_transform(y).flatten()
            if offset is not None:      # e.g. the ARIMA forecast when the net models the residual
                y = np.asarray(offset, dtype=float).flatten()[: len(y)] + gate * y
            preds.append(y)
    finally:
        for layer, trainable in zip(batch_norm_layers, trainable_states):
            layer.trainable = trainable

    arr = np.asarray(preds)  # (passes, n)
    mean_pred = arr.mean(axis=0)
    lower = np.quantile(arr, alpha / 2.0, axis=0)
    upper = np.quantile(arr, 1 - alpha / 2.0, axis=0)
    return mean_pred, lower, upper


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    y = np.asarray(y_true)
    l = np.asarray(lower)
    u = np.asarray(upper)
    n = min(len(y), len(l), len(u))
    if n == 0:
        return {"coverage": np.nan, "avg_width": np.nan}
    inside = (y[:n] >= l[:n]) & (y[:n] <= u[:n])
    return {"coverage": float(inside.mean()), "avg_width": float(np.mean(u[:n] - l[:n]))}


def sensitivity_perturbation_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_perturbed: np.ndarray,
    label: str,
):
    n = min(len(y_true), len(y_pred_clean), len(y_pred_perturbed))
    if n == 0:
        return None
    yt = y_true[:n]
    yc = y_pred_clean[:n]
    yp = y_pred_perturbed[:n]
    clean_rmse = rmse(yt, yc)
    pert_rmse = rmse(yt, yp)
    clean_doc = directional_accuracy(yt, yc)
    pert_doc = directional_accuracy(yt, yp)
    return {
        "Model": model_name,
        "Scenario": label,
        "Clean_RMSE": clean_rmse,
        "Perturbed_RMSE": pert_rmse,
        "Delta_RMSE": pert_rmse - clean_rmse,
        "Clean_DoC": clean_doc,
        "Perturbed_DoC": pert_doc,
        "Delta_DoC": pert_doc - clean_doc,
    }


# ──────────────────────────────────────────────────────────────
# Moving-block bootstrap for RMSE ratios
# ──────────────────────────────────────────────────────────────
def block_bootstrap_rmse_ratio(err_model, err_ref, horizon: int = 1, n_boot: int = None,
                               seed: int = SEED):
    """
    95% percentile interval for RMSE(model) / RMSE(reference).

    err_*: forecast errors on the same test rows. Rows are resampled in
    overlapping blocks of length max(BOOTSTRAP_BLOCK_MIN, 2h) (moving-block
    bootstrap), which keeps the serial dependence that makes consecutive
    errors of a persistent target non-independent. The same resampled row
    indices are applied to both models, so the interval reflects the paired
    difference. Returns (ratio, lo, hi, block_length).
    """
    e1 = np.asarray(err_model, dtype=float).flatten()
    e2 = np.asarray(err_ref, dtype=float).flatten()
    n = min(len(e1), len(e2))
    e1, e2 = e1[:n], e2[:n]
    B = int(n_boot or BOOTSTRAP_RESAMPLES)
    L = int(max(BOOTSTRAP_BLOCK_MIN, 2 * horizon))
    L = min(L, n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    s1 = np.sqrt(np.mean(e1[idx] ** 2, axis=1))
    s2 = np.sqrt(np.mean(e2[idx] ** 2, axis=1))
    ratios = s1 / s2
    point = float(np.sqrt(np.mean(e1 ** 2)) / np.sqrt(np.mean(e2 ** 2)))
    return point, float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)), L


def build_ratio_ci_table(results: dict, references, horizon: int, output_suffix: str = ""):
    """RMSE ratio of every model to each reference model with bootstrap intervals."""
    rows = []
    for ref in references:
        if ref not in results:
            continue
        a_ref = np.asarray(results[ref]["actual"], dtype=float)
        p_ref = np.asarray(results[ref]["predictions"], dtype=float)
        for name, res in results.items():
            if name == ref:
                continue
            a = np.asarray(res["actual"], dtype=float)
            p = np.asarray(res["predictions"], dtype=float)
            n = min(len(a), len(a_ref), len(p), len(p_ref))
            point, lo, hi, L = block_bootstrap_rmse_ratio(a[:n] - p[:n], a_ref[:n] - p_ref[:n], horizon=horizon)
            rows.append({"Model": name, "Reference": ref, "RMSE_ratio": point,
                         "CI_low": lo, "CI_high": hi, "Block": L, "N": n,
                         "B": int(BOOTSTRAP_RESAMPLES), "CI_excludes_1": bool(lo > 1 or hi < 1)})
    out = pd.DataFrame(rows)
    if not out.empty:
        suf = f"_{output_suffix}" if output_suffix else ""
        out.to_csv(os.path.join(RESULTS_DIR, f"ratio_ci{suf}.csv"), index=False)
    return out
