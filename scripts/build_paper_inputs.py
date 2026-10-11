"""
Aggregate a results directory into the summary tables used by the paper.

Usage:  python scripts/build_paper_inputs.py [results_dir]
Writes  <results_dir>/paper_inputs/*.csv  and  <results_dir>/paper_inputs/KEY_NUMBERS.md

Nothing here is hand-edited: every number in the manuscript can be traced to
one of these files.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = os.path.join(RES, "paper_inputs")
os.makedirs(OUT, exist_ok=True)


def _tag(path, prefix):
    m = re.match(prefix + r"_(.+)_USD_h(\d+)\.csv", os.path.basename(path))
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load(prefix, rolling=False, **kw):
    pat = f"{prefix}_rolling_*_USD_h*.csv" if rolling else f"{prefix}_*_USD_h*.csv"
    frames = []
    for p in sorted(glob.glob(os.path.join(RES, pat))):
        stem = os.path.basename(p)
        if not rolling and ("rolling" in stem or "ablation" in stem):
            continue
        a, h = _tag(p.replace("_rolling", ""), prefix)
        if a is None:
            continue
        d = pd.read_csv(p, **kw)
        d["Asset"], d["h"] = a, h
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(df, name):
    if df is not None and len(df):
        df.to_csv(os.path.join(OUT, name), index=False)
    return df


key = []

for label, rolling in (("holdout", False), ("rolling", True)):
    C = load("comparison_table", rolling)
    if C.empty:
        continue
    C["rank"] = C.groupby(["Asset", "h"]).RMSE.rank()
    mean = C.groupby(["h", "Model"]).agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), DoC=("DoC", "mean"),
                                         R2=("R2", "mean"), MeanRank=("rank", "mean")).reset_index()
    wins = C[C["rank"] == 1].groupby(["h", "Model"]).size().rename("Wins").reset_index()
    mean = mean.merge(wins, on=["h", "Model"], how="left").fillna({"Wins": 0})
    save(mean, f"{label}_cross_asset_mean.csv")
    save(C, f"{label}_comparison_all.csv")
    P = C.pivot_table(index=["Asset", "h"], columns="Model", values="RMSE")
    rel = []
    for ref in ("Naive_Persistence", "KRP", "HAR-RV", "ARIMA"):
        if ref in P.columns:
            r = (P.div(P[ref], axis=0) - 1) * 100
            r = r.reset_index().melt(id_vars=["Asset", "h"], var_name="Model", value_name="PctVsRef")
            r["Reference"] = ref
            rel.append(r)
    if rel:
        R = pd.concat(rel)
        save(R, f"{label}_rmse_vs_references.csv")
        g = R[R.Model.str.startswith("Residual Hybrid")].groupby(["Reference", "Model", "h"]).PctVsRef.agg(["mean", lambda x: int((x < 0).sum())])
        g.columns = ["MeanPct", "CasesBetter"]
        save(g.reset_index(), f"{label}_ahi_vs_references.csv")
    best = mean.sort_values(["h", "RMSE"]).groupby("h").head(3)
    key.append(f"## {label}: three lowest cross-asset RMSE per horizon\n" + best.to_string(index=False) + "\n")

# ---- Pairwise tests
for label, rolling in (("holdout", False), ("rolling", True)):
    D = load("dm_tests", rolling)
    if D.empty:
        continue
    D["Significant"] = D["Significant"].astype(bool)
    s = D.groupby("h").agg(Pairs=("p_value", "size"), SigRaw=("Significant", "sum"),
                           SigHolm=("Significant_holm", "sum")).reset_index()
    save(s, f"{label}_dm_summary.csv")
    save(D, f"{label}_dm_all.csv")

# ---- Ratio CIs
for label, rolling in (("holdout", False), ("rolling", True)):
    Rc = load("ratio_ci", rolling)
    save(Rc, f"{label}_ratio_ci_all.csv")

# ---- Regimes and stress windows (rolling covers LUNA/FTX)
for label, rolling in (("holdout", False), ("rolling", True)):
    G = load("regime_metrics", rolling)
    if not G.empty:
        m = G.groupby(["h", "Regime", "Model"]).RMSE.mean().unstack("Regime")
        if {"high_vol", "calm"} <= set(m.columns):
            m["HighOverCalm"] = m["high_vol"] / m["calm"]
        save(m.reset_index(), f"{label}_regime_mean.csv")
    S = load("shock_window_metrics", rolling)
    if not S.empty:
        save(S.groupby(["h", "Window", "Model"]).agg(RMSE=("RMSE", "mean"), N=("N", "sum")).reset_index(),
             f"{label}_shock_windows_mean.csv")
        save(S, f"{label}_shock_windows_all.csv")

# ---- Seeds
p = os.path.join(RES, "seed_summary.csv")
if os.path.exists(p):
    sd = pd.read_csv(p)
    sd["CV"] = sd.RMSE_std / sd.RMSE_mean
    save(sd, "seed_summary_with_cv.csv")

# ---- Ablation
ab = []
for f in sorted(glob.glob(os.path.join(RES, "ablation_results_*_USD_h*.csv"))):
    h = int(re.search(r"_h(\d+)\.csv", f).group(1))
    d = pd.read_csv(f); d["h"] = h; ab.append(d)
if ab:
    A = pd.concat(ab)
    A["RelToBestPct"] = A.groupby("h").RMSE.transform(lambda x: (x / x.min() - 1) * 100)
    A["RMSE_over_seedstd"] = (A.RMSE - A.groupby("h").RMSE.transform("min")) / A.RMSE_std
    save(A, "ablation_all.csv")

# ---- Interpretation
for prefix in ("permutation_importance", "attention_summary", "attention_lag_profile", "interval_metrics", "sensitivity_metrics"):
    frames = []
    for f in sorted(glob.glob(os.path.join(RES, f"{prefix}_*_USD_h*.csv"))):
        if "rolling" in f:
            continue
        d = pd.read_csv(f)
        a, h = _tag(f, prefix)
        d["Asset"], d["h"] = a, h
        frames.append(d)
    if frames:
        X = pd.concat(frames, ignore_index=True)
        save(X, f"{prefix}_all.csv")
        if prefix == "permutation_importance":
            save(X.groupby(["h", "Model", "Level", "Name"]).RelIncrease.mean().reset_index(), "permutation_importance_mean.csv")
        if prefix == "interval_metrics":
            save(X.groupby(["h", "Model", "Method"]).agg(Coverage=("Coverage", "mean"), AvgWidth=("AvgWidth", "mean")).reset_index(),
                 "interval_summary.csv")
        if prefix == "sensitivity_metrics":
            X["Ratio"] = X.Perturbed_RMSE / X.Clean_RMSE
            save(X.groupby(["Model", "Channels", "Scenario"]).agg(
                RatioMean=("Ratio", "mean"), RatioMedian=("Ratio", "median"), dDoC=("Delta_DoC", "mean")).reset_index(),
                "sensitivity_summary.csv")

# ---- Cross-asset, audit, manifest copies
for name in ("cross_asset_generalization.csv", "feature_audit.csv", "sample_manifest.csv", "timing_report.csv"):
    p = os.path.join(RES, name)
    if os.path.exists(p):
        pd.read_csv(p).to_csv(os.path.join(OUT, name), index=False)

with open(os.path.join(OUT, "KEY_NUMBERS.md"), "w") as fh:
    fh.write("# Key numbers (auto-generated)\n\n" + "\n".join(key))
print(f"Wrote paper inputs to {OUT}")
