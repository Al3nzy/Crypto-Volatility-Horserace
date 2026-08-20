"""
Statistical Baseline Module (Section 3.2) – ARIMA & GARCH(1,1).

Both models operate on the **univariate volatility series** from the merged
dataset so the comparison against the multimodal DL model is fair.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from config import LOOKBACK_WINDOW, TRAIN_RATIO

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ──────────────────────────────────────────────────────────────
# ARIMA baseline
# ──────────────────────────────────────────────────────────────
def run_arima(vol_series: pd.Series, forecast_horizon: int = 1, order=(5, 1, 0)):
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

    predictions = []
    actuals = []
    test_dates = []
    idx = vol_series.index

    print(f"[ARIMA] Forecasting {n_windows - split_idx} steps (h={forecast_horizon}) …")
    for k in range(split_idx, n_windows):
        t0 = k + LOOKBACK_WINDOW - 1
        target_idx = t0 + forecast_horizon
        history = list(vals[: t0 + 1])
        try:
            model = ARIMA(history, order=order)
            fitted = model.fit()
            fc = fitted.forecast(steps=forecast_horizon)
            yhat = float(fc[-1]) if hasattr(fc, "__len__") else float(fc)
        except Exception:
            yhat = history[-1]  # fallback: persist last value
        predictions.append(yhat)
        actuals.append(vals[target_idx])
        test_dates.append(idx[target_idx])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
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
def run_garch(returns_series: pd.Series, vol_series: pd.Series, forecast_horizon: int = 1):
    """
    Rolling h-step-ahead GARCH(1,1) conditional volatility forecast.
    Input: log-return series and actual realized volatility series.
    The model predicts conditional variance at step h -> sqrt for sigma.
    """
    ret_vals = returns_series.values
    vol_vals = vol_series.values
    n = len(ret_vals)
    n_windows = n - LOOKBACK_WINDOW - forecast_horizon + 1
    split_idx = int(n_windows * TRAIN_RATIO)
    scaled_ret = returns_series * 100

    predictions = []
    actuals = []
    test_dates = []
    idx = vol_series.index

    print(f"[GARCH] Forecasting {n_windows - split_idx} steps (h={forecast_horizon}) …")
    for k in range(split_idx, n_windows):
        t0 = k + LOOKBACK_WINDOW - 1
        target_idx = t0 + forecast_horizon
        train_window = scaled_ret.iloc[: t0 + 1]
        try:
            am = arch_model(train_window, vol="Garch", p=1, q=1,
                            mean="Constant", dist="Normal")
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=forecast_horizon)
            var_h = forecast.variance.values[-1, forecast_horizon - 1]
            sigma = np.sqrt(var_h) / 100.0
        except Exception:
            sigma = train_window.iloc[-14:].std() / 100.0
        predictions.append(sigma)
        actuals.append(vol_vals[target_idx])
        test_dates.append(idx[target_idx])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
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
def run_gjr_garch(returns_series: pd.Series, vol_series: pd.Series, forecast_horizon: int = 1):
    """
    Rolling h-step-ahead GJR-GARCH conditional volatility (arch: GARCH + o=1).
    """
    ret_vals = returns_series.values
    vol_vals = vol_series.values
    n = len(ret_vals)
    n_windows = n - LOOKBACK_WINDOW - forecast_horizon + 1
    split_idx = int(n_windows * TRAIN_RATIO)
    scaled_ret = returns_series * 100

    predictions = []
    actuals = []
    test_dates = []
    idx = vol_series.index

    print(f"[GJR-GARCH] Forecasting {n_windows - split_idx} steps (h={forecast_horizon}) …")
    for k in range(split_idx, n_windows):
        t0 = k + LOOKBACK_WINDOW - 1
        target_idx = t0 + forecast_horizon
        train_window = scaled_ret.iloc[: t0 + 1]
        try:
            am = arch_model(
                train_window,
                vol="Garch",
                p=1,
                o=1,
                q=1,
                mean="Constant",
                dist="Normal",
            )
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=forecast_horizon)
            var_h = forecast.variance.values[-1, forecast_horizon - 1]
            sigma = np.sqrt(var_h) / 100.0
        except Exception:
            sigma = train_window.iloc[-14:].std() / 100.0
        predictions.append(sigma)
        actuals.append(vol_vals[target_idx])
        test_dates.append(idx[target_idx])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
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
def get_arima_forecasts_for_residuals(vol_series: pd.Series, forecast_horizon: int = 1, order=(5, 1, 0), min_history=20):
    """
    Rolling h-step-ahead ARIMA forecasts for the series (except warmup).
    Used to compute residuals for the residual hybrid (ARIMA + DL-on-residuals).
    Returns (arima_pred, actual) arrays of length len(vol_series).
    preds[t] = h-step forecast for vol at t, using data up to t - forecast_horizon.
    """
    vals = vol_series.values
    n = len(vals)
    preds = np.full(n, np.nan)
    warmup = min_history + forecast_horizon
    preds[:warmup] = np.mean(vals[:min_history])  # warmup fallback

    for t in range(warmup, n):
        origin = t - forecast_horizon
        try:
            model = ARIMA(vals[: origin + 1], order=order)
            fitted = model.fit()
            fc = fitted.forecast(steps=forecast_horizon)
            preds[t] = float(fc[-1]) if hasattr(fc, "__len__") else float(fc)
        except Exception:
            preds[t] = vals[origin]

    return preds, vals


# ──────────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────────
def run_all_baselines(
    merged_df: pd.DataFrame,
    forecast_horizon: int = 1,
    include_har_gjr: bool = True,
):
    """
    Run ARIMA, GARCH(1,1), and optionally HAR-RV + GJR-GARCH on merged panel for given horizon.
    """
    vol_series = merged_df["Volatility"]
    ret_series = merged_df["Log_Return"]

    arima_res = run_arima(vol_series, forecast_horizon=forecast_horizon)
    garch_res = run_garch(ret_series, vol_series, forecast_horizon=forecast_horizon)

    out = {}
    if arima_res is not None:
        out["ARIMA"] = arima_res
    if garch_res is not None:
        out["GARCH"] = garch_res

    if include_har_gjr:
        har_res = run_har_rv(merged_df, forecast_horizon=forecast_horizon)
        if har_res is not None:
            out["HAR-RV"] = har_res

        gjr_res = run_gjr_garch(ret_series, vol_series, forecast_horizon=forecast_horizon)
        if gjr_res is not None:
            out["GJR-GARCH"] = gjr_res

    return out
