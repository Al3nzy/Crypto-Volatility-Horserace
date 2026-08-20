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
from config import RESULTS_DIR, LOOKBACK_WINDOW

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
        rows.append({
            "Model": name,
            "RMSE": res["rmse"],
            "MAE": res["mae"],
            "DoC": doc,
            "R2": r2v,
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
    Diebold-Mariano test for equal predictive accuracy.
    H0: both methods have equal forecast accuracy.
    e1, e2: squared forecast errors from model 1 and 2 (same length).
    h: forecast horizon (1 for one-step-ahead).
    Returns: {"dm_stat": float, "p_value": float, "significant": bool}
    """
    d = e1 - e2  # loss differential (squared errors)
    n = len(d)
    d_bar = np.mean(d)
    # HAC variance (Newey-West for serial correlation)
    gamma0 = np.var(d)
    gammas = []
    for k in range(1, min(h, n)):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0
        gammas.append(gamma_k * 2)
    var_d = gamma0 + sum(gammas) if gammas else gamma0
    if var_d <= 0:
        var_d = 1e-10
    dm_stat = d_bar / np.sqrt(var_d / n)
    # Two-tailed p-value (approximate N(0,1))
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "significant": p_value < 0.05}


def build_dm_table(results: dict, output_suffix: str = "") -> pd.DataFrame:
    """
    Pairwise Diebold-Mariano tests (squared errors) for all model pairs.
    Negative DM stat means model1 is better; positive means model2 is better.
    """
    names = list(results.keys())
    pairs = []
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            a1 = np.asarray(results[m1]["actual"])
            p1 = np.asarray(results[m1]["predictions"])
            p2 = np.asarray(results[m2]["predictions"])
            n = min(len(a1), len(p1), len(p2))
            e1 = (a1[:n] - p1[:n]) ** 2
            e2 = (a1[:n] - p2[:n]) ** 2
            dm = run_diebold_mariano(e1, e2)
            better = m1 if dm["dm_stat"] < 0 else m2
            pairs.append({
                "Model_A": m1, "Model_B": m2,
                "DM_stat": dm["dm_stat"], "p_value": dm["p_value"],
                "Significant": dm["significant"], "Better": better,
            })
    df = pd.DataFrame(pairs)
    if not df.empty:
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
    for start, end, label in crash_windows:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.12, color="red")
            ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.95, label,
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


def build_regime_table(results: dict, output_suffix: str = "") -> pd.DataFrame:
    """
    Regime-aware evaluation using test-period realized volatility and returns:
    - calm: vol <= p33
    - bear: return < 0 and p33 < vol <= p67
    - bull: return >= 0 and p33 < vol <= p67
    - crisis: vol > p67
    """
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

        actual_series = pd.Series(actual, index=dates)
        ret = actual_series.diff().fillna(0.0).values
        q33, q67 = np.quantile(actual, [0.33, 0.67])

        regime = np.where(
            actual <= q33, "calm",
            np.where(actual > q67, "crisis", np.where(ret >= 0, "bull", "bear"))
        )
        for rg in ["calm", "bull", "bear", "crisis"]:
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
    downside_std = np.nanstd(np.minimum(strat_ret, 0)) + eps

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


def mc_dropout_intervals(model, X_test: np.ndarray, target_scaler, passes: int = 50, alpha: float = 0.10):
    """
    Monte-Carlo dropout intervals for DL predictions.
    """
    preds = []
    for _ in range(passes):
        y = model(X_test, training=True).numpy()
        y = target_scaler.inverse_transform(y).flatten()
        preds.append(y)
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
