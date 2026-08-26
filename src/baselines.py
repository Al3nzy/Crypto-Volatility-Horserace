"""
Statistical Baseline Module (Section 3.2) – ARIMA & GARCH(1,1).

Both models operate on the **univariate volatility series** from the merged
dataset so the comparison against the multimodal DL model is fair.

PERFORMANCE NOTE
-----------------
All four rolling-refit routines below (`run_arima`, `run_garch`,
`run_gjr_garch`, `get_arima_forecasts_for_residuals`) refit a fresh
statistical model at every single step of a rolling-origin forecast. Each
step's fit only depends on history up to that point, not on any other
step's result, so the steps are embarrassingly parallel: running them
concurrently across CPU cores changes only wall-clock time, never the
numbers. `n_jobs` controls how many worker processes joblib uses
(`n_jobs=1` reproduces the original strictly-sequential behaviour;
`n_jobs=-1` uses all available cores). This is wired to
`config.BASELINE_N_JOBS` by default.
"""
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from config import BASELINE_N_JOBS, LOOKBACK_WINDOW, TRAIN_RATIO

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _arima_step(vals: np.ndarray, t0: int, forecast_horizon: int, order):
    """
    Fit ARIMA(order) on vals[:t0+1] and forecast `forecast_horizon` steps
    ahead. Falls back to last-observed-value persistence on any fit failure
    (identical fallback semantics to the original sequential implementation).
    """
    history = vals[: t0 + 1]
    try:
        model = ARIMA(history, order=order)
        fitted = model.fit()
        fc = fitted.forecast(steps=forecast_horizon)
        yhat = float(fc[-1]) if hasattr(fc, "__len__") else float(fc)
    except Exception:
        yhat = float(history[-1])
    return yhat


def _garch_step(scaled_ret: pd.Series, t0: int, forecast_horizon: int, asymmetric: bool):
    """
    Fit GARCH(1,1) (or GJR-GARCH when asymmetric=True) on scaled_ret up to
    and including t0, forecast forecast_horizon steps ahead. Same fallback
    semantics as the original sequential implementation.
    """
    train_window = scaled_ret.iloc[: t0 + 1]
    try:
        kwargs = dict(vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
        if asymmetric:
            kwargs["o"] = 1
        am = arch_model(train_window, **kwargs)
        res = am.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=forecast_horizon)
        var_h = forecast.variance.values[-1, forecast_horizon - 1]
        sigma = np.sqrt(var_h) / 100.0
    except Exception:
        sigma = train_window.iloc[-14:].std() / 100.0
    return sigma


# ──────────────────────────────────────────────────────────────
# ARIMA baseline
# ──────────────────────────────────────────────────────────────
def run_arima(vol_series: pd.Series, forecast_horizon: int = 1, order=(5, 1, 0), n_jobs: int = 1):
    """
    Rolling h-step-ahead ARIMA forecast on the window-aligned test portion.
    """
    vals = vol_series.values
    n = len(vals)
    n_windows = n - LOOKBACK_WINDOW - forecast_horizon + 1
    if n_windows < 30:
        print("[ARIMA] Not enough rows for windowed forecasting.")
        return None
    split_idx = int(n_windows * TRAIN_RATIO)
    idx = vol_series.index

    ks = list(range(split_idx, n_windows))
    t0s = [k + LOOKBACK_WINDOW - 1 for k in ks]
    target_idxs = [t0 + forecast_horizon for t0 in t0s]

    print(f"[ARIMA] Forecasting {len(ks)} steps (h={forecast_horizon}) using n_jobs={n_jobs} …")
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_arima_step)(vals, t0, forecast_horizon, order) for t0 in t0s
    )
    predictions = np.array(predictions)
    actuals = vals[target_idxs]
    test_dates = idx[target_idxs]

    rmse_val = _rmse(actuals, predictions)
    mae_val = mean_absolute_error(actuals, predictions)
    print(f"[ARIMA] RMSE={rmse_val:.6f}  MAE={mae_val:.6f}")
    return {
        "name": "ARIMA",
        "predictions": predictions,
        "actual": actuals,
        "dates": pd.DatetimeIndex(test_dates),
        "rmse": rmse_val,
        "mae": mae_val,
    }


# ──────────────────────────────────────────────────────────────
# GARCH(1,1) baseline
# ──────────────────────────────────────────────────────────────
def run_garch(returns_series: pd.Series, vol_series: pd.Series, forecast_horizon: int = 1, n_jobs: int = 1):
    """
    Rolling h-step-ahead GARCH(1,1) conditional volatility forecast.
    Input: log-return series and actual realized volatility series.
    The model predicts conditional variance at step h -> sqrt for sigma.
    """
    vol_vals = vol_series.values
    n = len(returns_series)
    n_windows = n - LOOKBACK_WINDOW - forecast_horizon + 1
    split_idx = int(n_windows * TRAIN_RATIO)
    scaled_ret = returns_series * 100
    idx = vol_series.index

    ks = list(range(split_idx, n_windows))
    t0s = [k + LOOKBACK_WINDOW - 1 for k in ks]
    target_idxs = [t0 + forecast_horizon for t0 in t0s]

    print(f"[GARCH] Forecasting {len(ks)} steps (h={forecast_horizon}) using n_jobs={n_jobs} …")
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_garch_step)(scaled_ret, t0, forecast_horizon, False) for t0 in t0s
    )
    predictions = np.array(predictions)
    actuals = vol_vals[target_idxs]
    test_dates = idx[target_idxs]

    rmse_val = _rmse(actuals, predictions)
    mae_val = mean_absolute_error(actuals, predictions)
    print(f"[GARCH] RMSE={rmse_val:.6f}  MAE={mae_val:.6f}")
    return {
        "name": "GARCH(1,1)",
        "predictions": predictions,
        "actual": actuals,
        "dates": pd.DatetimeIndex(test_dates),
        "rmse": rmse_val,
        "mae": mae_val,
    }


# ──────────────────────────────────────────────────────────────
# GJR-GARCH(1,1,1) – asymmetric volatility (leverage) term
# ──────────────────────────────────────────────────────────────
def run_gjr_garch(returns_series: pd.Series, vol_series: pd.Series, forecast_horizon: int = 1, n_jobs: int = 1):
    """
    Rolling h-step-ahead GJR-GARCH conditional volatility (arch: GARCH + o=1).
    """
    vol_vals = vol_series.values
    n = len(returns_series)
    n_windows = n - LOOKBACK_WINDOW - forecast_horizon + 1
    split_idx = int(n_windows * TRAIN_RATIO)
    scaled_ret = returns_series * 100
    idx = vol_series.index

    ks = list(range(split_idx, n_windows))
    t0s = [k + LOOKBACK_WINDOW - 1 for k in ks]
    target_idxs = [t0 + forecast_horizon for t0 in t0s]

    print(f"[GJR-GARCH] Forecasting {len(ks)} steps (h={forecast_horizon}) using n_jobs={n_jobs} …")
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_garch_step)(scaled_ret, t0, forecast_horizon, True) for t0 in t0s
    )
    predictions = np.array(predictions)
    actuals = vol_vals[target_idxs]
    test_dates = idx[target_idxs]

    rmse_val = _rmse(actuals, predictions)
    mae_val = mean_absolute_error(actuals, predictions)
    print(f"[GJR-GARCH] RMSE={rmse_val:.6f}  MAE={mae_val:.6f}")
    return {
        "name": "GJR-GARCH",
        "predictions": predictions,
        "actual": actuals,
        "dates": pd.DatetimeIndex(test_dates),
        "rmse": rmse_val,
        "mae": mae_val,
    }


# ──────────────────────────────────────────────────────────────
# HAR-RV: OLS on daily / weekly / monthly volatility components
# ──────────────────────────────────────────────────────────────
def run_har_rv(merged_df: pd.DataFrame, forecast_horizon: int = 1):
    """
    Heterogeneous Autoregressive model of realized volatility for h-step-ahead forecast.
    Features at forecast origin t0 = k + LOOKBACK_WINDOW - 1:
      RV_d = vol[t0]
      RV_w = mean(vol[max(0, t0 - 4) : t0 + 1])   (5-day average up to t0)
      RV_m = mean(vol[max(0, t0 - 21) : t0 + 1])  (22-day average up to t0)
    Target = vol[t0 + forecast_horizon].
    """
    vol = merged_df["Volatility"].values.astype(float)
    n = len(vol)
    n_windows = n - LOOKBACK_WINDOW - forecast_horizon + 1
    if n_windows < 30:
        print("[HAR-RV] Not enough rows for HAR regression.")
        return None

    X_har = np.zeros((n_windows, 3))
    y_targets = np.zeros(n_windows)
    for k in range(n_windows):
        t0 = k + LOOKBACK_WINDOW - 1
        T = t0 + forecast_horizon
        rv_d = vol[t0]
        rv_w = np.mean(vol[max(0, t0 - 4) : t0 + 1])
        rv_m = np.mean(vol[max(0, t0 - 21) : t0 + 1])
        X_har[k] = [rv_d, rv_w, rv_m]
        y_targets[k] = vol[T]

    split_idx = int(n_windows * TRAIN_RATIO)
    X_train, X_test = X_har[:split_idx], X_har[split_idx:]
    y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]
    X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_b = np.column_stack([np.ones(len(X_test)), X_test])
    coef, _, rank, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
    if rank < X_train_b.shape[1]:
        print("[HAR-RV] Degenerate design; skipping.")
        return None
    pred = X_test_b @ coef

    idx = merged_df.index
    test_dates = [idx[k + LOOKBACK_WINDOW + forecast_horizon - 1] for k in range(split_idx, n_windows)]

    rmse_v = _rmse(y_test, pred)
    mae_v = mean_absolute_error(y_test, pred)
    print(f"[HAR-RV] RMSE={rmse_v:.6f}  MAE={mae_v:.6f}")
    return {
        "name": "HAR-RV",
        "predictions": pred,
        "actual": y_test,
        "dates": pd.DatetimeIndex(test_dates),
        "rmse": rmse_v,
        "mae": mae_v,
    }


# ──────────────────────────────────────────────────────────────
# ARIMA forecasts for residual hybrid (aligned with pipeline windows)
# ──────────────────────────────────────────────────────────────
def get_arima_forecasts_for_residuals(
    vol_series: pd.Series,
    forecast_horizon: int = 1,
    order=(5, 1, 0),
    min_history=20,
    n_jobs: int = 1,
):
    """
    Rolling h-step-ahead ARIMA forecasts for the series (except warmup).
    Used to compute residuals for the residual hybrid (ARIMA + DL-on-residuals).
    Returns (arima_pred, actual) arrays of length len(vol_series).
    preds[t] = h-step forecast for vol at t, using data up to t - forecast_horizon.

    This refits ARIMA at (almost) every index in the series, which is by far
    the single most expensive routine in the whole baseline suite (it is not
    restricted to the test split). It is parallelized across n_jobs workers;
    with n_jobs=1 it reproduces the original sequential results exactly.
    """
    vals = vol_series.values
    n = len(vals)
    preds = np.full(n, np.nan)
    warmup = min_history + forecast_horizon
    preds[:warmup] = np.mean(vals[:min_history])  # warmup fallback

    t_list = list(range(warmup, n))
    origins = [t - forecast_horizon for t in t_list]

    print(
        f"[ARIMA-Residual] Forecasting {len(t_list)} steps (h={forecast_horizon}) "
        f"using n_jobs={n_jobs} …"
    )
    results = Parallel(n_jobs=n_jobs)(
        delayed(_arima_step)(vals, origin, forecast_horizon, order) for origin in origins
    )
    for t, val in zip(t_list, results):
        preds[t] = val

    return preds, vals


# ──────────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────────
def run_all_baselines(
    merged_df: pd.DataFrame,
    forecast_horizon: int = 1,
    include_har_gjr: bool = True,
    n_jobs: int = BASELINE_N_JOBS,
):
    """
    Run ARIMA, GARCH(1,1), and optionally HAR-RV + GJR-GARCH on merged panel for given horizon.
    """
    vol_series = merged_df["Volatility"]
    ret_series = merged_df["Log_Return"]

    arima_res = run_arima(vol_series, forecast_horizon=forecast_horizon, n_jobs=n_jobs)
    garch_res = run_garch(ret_series, vol_series, forecast_horizon=forecast_horizon, n_jobs=n_jobs)

    out = {}
    if arima_res is not None:
        out["ARIMA"] = arima_res
    if garch_res is not None:
        out["GARCH"] = garch_res

    if include_har_gjr:
        har_res = run_har_rv(merged_df, forecast_horizon=forecast_horizon)
        if har_res is not None:
            out["HAR-RV"] = har_res

        gjr_res = run_gjr_garch(ret_series, vol_series, forecast_horizon=forecast_horizon, n_jobs=n_jobs)
        if gjr_res is not None:
            out["GJR-GARCH"] = gjr_res

    return out
