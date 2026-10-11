"""
Interpretability, calibration and robustness utilities.

* temporal attention profile (what the attention layer actually does)
* permutation importance by feature and by information pillar (what the
  forecast actually depends on, measured as the RMSE increase when a channel
  is shuffled across test windows)
* split-conformal intervals (calibrated counterpart of MC dropout)
* input perturbation tests for any forecaster (standalone network or AHI)
"""
import numpy as np
import pandas as pd

from config import (ASSET_ONCHAIN_FEATURE_COLS, ASSET_SENTIMENT_FEATURE_COLS, MARKET_FEATURE_COLS,
                    MARKET_STATIONARY_COLS, ONCHAIN_STATIONARY_COLS, RV_FEATURE_COLS, SEED)


def pillar_of(feature: str) -> str:
    if feature in MARKET_FEATURE_COLS or feature in MARKET_STATIONARY_COLS:
        return "market"
    if feature in ASSET_SENTIMENT_FEATURE_COLS:
        return "sentiment"
    if feature in ASSET_ONCHAIN_FEATURE_COLS or feature in ONCHAIN_STATIONARY_COLS:
        return "onchain"
    if feature in RV_FEATURE_COLS:
        return "lagged_vol"
    return "other"


# ──────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────
def attention_lag_profile(attn: np.ndarray) -> pd.DataFrame:
    """
    attn: (n, heads, T, T) attention probabilities (rows = queries, sum to 1).
    Averages over samples, heads, and query positions to obtain the weight
    given to each of the T lookback days, with dispersion across samples.
    Lag 0 is the most recent day.
    """
    per_sample = attn.mean(axis=(1, 2))              # (n, T) weight on each key
    T = per_sample.shape[1]
    mean = per_sample.mean(axis=0)
    std = per_sample.std(axis=0)
    prof = pd.DataFrame({
        "Lag": np.arange(T)[::-1],                   # last column is t-0
        "MeanWeight": mean, "StdAcrossWindows": std,
        "UniformWeight": 1.0 / T,
    }).sort_values("Lag").reset_index(drop=True)
    return prof


def attention_summary(attn: np.ndarray) -> dict:
    per_sample = attn.mean(axis=(1, 2))
    T = per_sample.shape[1]
    mean = per_sample.mean(axis=0)
    ent = -(mean * np.log(mean + 1e-12)).sum() / np.log(T)
    return {
        "Windows": int(attn.shape[0]), "Heads": int(attn.shape[1]), "Lookback": int(T),
        "UniformWeight": 1.0 / T, "MinWeight": float(mean.min()), "MaxWeight": float(mean.max()),
        "MaxOverMin": float(mean.max() / mean.min()),
        "NormalizedEntropy": float(ent),              # 1.0 = perfectly uniform
        "WeightLast3Days": float(mean[-3:].sum()),
        "UniformLast3Days": 3.0 / T,
    }


# ──────────────────────────────────────────────────────────────
# Permutation importance
# ──────────────────────────────────────────────────────────────
def permutation_importance(predict_fn, X: np.ndarray, y_true: np.ndarray, feature_names,
                           repeats: int = 10, seed: int = SEED, model_label: str = "model"):
    """
    predict_fn(X) -> forecasts in target units. For each channel the channel
    is shuffled across windows (each window keeps its own time structure for
    the other channels) and the RMSE increase is recorded. Pillar rows shuffle
    all channels of the pillar jointly with the same window permutation.
    """
    rng = np.random.default_rng(seed)
    base = float(np.sqrt(np.mean((y_true - predict_fn(X)) ** 2)))
    n = X.shape[0]

    def _delta(cols):
        vals = []
        for _ in range(repeats):
            perm = rng.permutation(n)
            Xp = X.copy()
            Xp[:, :, cols] = X[perm][:, :, cols]
            vals.append(float(np.sqrt(np.mean((y_true - predict_fn(Xp)) ** 2))) - base)
        return float(np.mean(vals)), float(np.std(vals))

    rows = []
    for j, f in enumerate(feature_names):
        m, sd = _delta([j])
        rows.append({"Model": model_label, "Level": "feature", "Name": f, "Pillar": pillar_of(f),
                     "BaseRMSE": base, "DeltaRMSE": m, "DeltaRMSE_std": sd, "RelIncrease": m / base})
    for p in sorted({pillar_of(f) for f in feature_names}):
        cols = [j for j, f in enumerate(feature_names) if pillar_of(f) == p]
        m, sd = _delta(cols)
        rows.append({"Model": model_label, "Level": "pillar", "Name": p, "Pillar": p,
                     "BaseRMSE": base, "DeltaRMSE": m, "DeltaRMSE_std": sd, "RelIncrease": m / base})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Split-conformal intervals
# ──────────────────────────────────────────────────────────────
def split_conformal_interval(y_cal: np.ndarray, p_cal: np.ndarray, p_test: np.ndarray, alpha: float = 0.10):
    """
    Symmetric split-conformal interval: the (1-alpha) finite-sample-corrected
    quantile of |y - p| on a calibration set is added to and subtracted from
    each test forecast. Calibration rows are the validation tail of the
    training windows (chronologically before the test rows).
    """
    r = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(p_cal, dtype=float))
    n = len(r)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    q = float(np.sort(r)[k - 1])
    p_test = np.asarray(p_test, dtype=float)
    return p_test - q, p_test + q, q


# ──────────────────────────────────────────────────────────────
# Input perturbation
# ──────────────────────────────────────────────────────────────
def perturb(X: np.ndarray, kind: str, level: float, channels, rng) -> np.ndarray:
    """
    kind="missing": zero a random fraction `level` of window entries in `channels`.
    kind="noise"  : add N(0, level^2) in min-max-scaled units to `channels`.
    """
    Xp = X.copy()
    ch = np.asarray(channels, dtype=int)
    sub = Xp[:, :, ch]
    if kind == "missing":
        mask = rng.random(sub.shape) < level
        sub[mask] = 0.0
    elif kind == "noise":
        sub = sub + rng.normal(0.0, level, size=sub.shape)
    else:
        raise ValueError(kind)
    Xp[:, :, ch] = sub
    return Xp


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────
def plot_attention_lag_profile(profiles: dict, path: str, title: str = ""):
    """Bars: mean attention weight per lookback day for each model, with the uniform level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.8 / max(1, len(profiles))
    for i, (label, prof) in enumerate(profiles.items()):
        x = np.arange(len(prof)) + i * width
        ax.bar(x, prof["MeanWeight"], width=width, label=label, hatch=["", "//", "..", "xx"][i % 4], alpha=0.85,
               yerr=prof["StdAcrossWindows"], capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(1.0 / len(next(iter(profiles.values()))), color="k", ls="--", lw=1, label="Uniform")
    ax.set_xticks(np.arange(len(prof)) + width * (len(profiles) - 1) / 2)
    ax.set_xticklabels([f"t-{int(l)}" for l in prof["Lag"]], fontsize=7, rotation=45)
    ax.set_ylabel("Mean attention weight")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_permutation_importance(df: pd.DataFrame, path: str, title: str = ""):
    """Horizontal bars of RMSE increase (%) when a pillar or feature is shuffled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = df["Model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, level in zip(axes, ["pillar", "feature"]):
        sub = df[df["Level"] == level]
        names = list(dict.fromkeys(sub.sort_values("RelIncrease", ascending=False)["Name"]))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            s = sub[sub["Model"] == m].set_index("Name").reindex(names)
            ax.barh(np.arange(len(names)) + i * width, 100 * s["RelIncrease"], height=width, label=m,
                    xerr=100 * s["DeltaRMSE_std"] / s["BaseRMSE"], hatch=["", "//", ".."][i % 3], alpha=0.85,
                    error_kw={"lw": 0.6})
        ax.set_yticks(np.arange(len(names)) + width * (len(models) - 1) / 2)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE increase when shuffled (%)", fontsize=8)
        ax.set_title(f"By {level}", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
