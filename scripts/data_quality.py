"""
Data provenance and preflight validation for paper-ready runs.
"""
from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from config import (
    ASSET_ONCHAIN_FEATURE_COLS,
    ASSET_ONCHAIN_CORE_FEATURE_COLS,
    ASSET_ONCHAIN_OPTIONAL_FEATURE_COLS,
    ASSET_SENTIMENT_FEATURE_COLS,
    ASSET_SENTIMENT_PREFLIGHT_OPTIONAL_COLS,
    ASSET_SENTIMENT_PREFLIGHT_REQUIRED_COLS,
    ENABLE_COINMARKETCAP,
    DATA_PREFLIGHT_JSON_FILE,
    DATA_QUALITY_REPORT_FILE,
    DATA_SOURCE_REPORT_FILE,
    MAX_PREFLIGHT_MISSING_RATE,
    MIN_PREFLIGHT_COVERAGE,
    PAPER_MODE,
    TICKERS,
)
from src.collection_utils import normalize_dates


def _normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
    out["Date"] = normalize_dates(out["Date"])
    return out


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _constant_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> list[str]:
    constants = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            constants.append(col)
        elif series.nunique() <= 1:
            constants.append(col)
    return constants


def _pillar_quality_rows(
    pillar: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    metadata: dict,
    expected_dates_by_asset: dict[str, set],
    required_feature_cols: list[str] | None = None,
    optional_feature_cols: list[str] | None = None,
) -> list[dict]:
    df = _normalize_panel(df)
    rows = []
    per_asset = metadata.get("per_asset", {})
    required_feature_cols = required_feature_cols or feature_cols
    optional_feature_cols = optional_feature_cols or []
    for ticker in TICKERS:
        asset_df = df[df["Asset"] == ticker].copy()
        expected_dates = expected_dates_by_asset.get(ticker, set())
        expected_rows = int(len(expected_dates))
        overlap_count = int(asset_df["Date"].isin(expected_dates).sum()) if expected_dates else int(len(asset_df))
        source_meta = per_asset.get(ticker, {})
        constant_cols = _constant_features(asset_df, required_feature_cols)
        optional_constant_cols = _constant_features(asset_df, optional_feature_cols)
        missing_cells = int(asset_df[required_feature_cols].isna().sum().sum()) if not asset_df.empty else 0
        total_cells = int(len(asset_df) * len(required_feature_cols))
        optional_missing_cells = (
            int(asset_df[optional_feature_cols].isna().sum().sum())
            if (not asset_df.empty and optional_feature_cols)
            else 0
        )
        optional_total_cells = int(len(asset_df) * len(optional_feature_cols))
        row = {
            "pillar": pillar,
            "asset": ticker,
            "source_type": source_meta.get("source_type", "unknown"),
            "is_real": bool(source_meta.get("is_real", False)),
            "row_count": int(len(asset_df)),
            "expected_rows": expected_rows,
            "coverage_ratio": _safe_ratio(overlap_count, expected_rows),
            "missing_rate": _safe_ratio(missing_cells, total_cells),
            "optional_missing_rate": _safe_ratio(optional_missing_cells, optional_total_cells),
            "constant_feature_count": len(constant_cols),
            "constant_features": "|".join(constant_cols),
            "optional_constant_feature_count": len(optional_constant_cols),
            "optional_constant_features": "|".join(optional_constant_cols),
            "source_components": "|".join(source_meta.get("source_components", [])),
            "errors": json.dumps(source_meta.get("errors", {}), ensure_ascii=True),
        }
        row["passes_preflight"] = (
            row["coverage_ratio"] >= MIN_PREFLIGHT_COVERAGE
            and row["missing_rate"] <= MAX_PREFLIGHT_MISSING_RATE
            and (pillar == "market" or row["is_real"])
            and row["constant_feature_count"] < len(required_feature_cols)
        )
        rows.append(row)
    return rows


def run_data_preflight(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    onchain_df: pd.DataFrame,
    source_metadata: dict,
) -> dict:
    """
    Validate the collected data before modeling and persist provenance reports.
    """
    market_df = _normalize_panel(market_df)
    sentiment_df = _normalize_panel(sentiment_df)
    onchain_df = _normalize_panel(onchain_df)

    market_meta = {"per_asset": {}}
    expected_dates_by_asset = {}
    for ticker in TICKERS:
        asset_market = market_df[market_df["Asset"] == ticker].copy()
        expected_dates_by_asset[ticker] = set(asset_market["Date"].tolist())
        comps = ["yfinance"]
        if ENABLE_COINMARKETCAP and "CMC_Close" in asset_market.columns:
            comps.append("coinmarketcap")
        market_meta["per_asset"][ticker] = {
            "source_type": "api_yfinance_coinmarketcap" if len(comps) > 1 else "api_yfinance",
            "source_components": comps,
            "is_real": not asset_market.empty,
            "errors": {},
        }

    quality_rows = []
    market_required = ["Open", "High", "Low", "Close", "Volume", "Log_Return", "Volatility"]
    market_optional = [c for c in ["CMC_Close", "CMC_Volume"] if c in market_df.columns]
    quality_rows.extend(
        _pillar_quality_rows(
            "market",
            market_df,
            market_required + market_optional,
            market_meta,
            expected_dates_by_asset,
            required_feature_cols=market_required,
            optional_feature_cols=market_optional,
        )
    )
    quality_rows.extend(
        _pillar_quality_rows(
            "sentiment",
            sentiment_df,
            ASSET_SENTIMENT_PREFLIGHT_REQUIRED_COLS + ASSET_SENTIMENT_PREFLIGHT_OPTIONAL_COLS,
            source_metadata.get("sentiment", {}),
            expected_dates_by_asset,
            required_feature_cols=ASSET_SENTIMENT_PREFLIGHT_REQUIRED_COLS,
            optional_feature_cols=ASSET_SENTIMENT_PREFLIGHT_OPTIONAL_COLS,
        )
    )
    quality_rows.extend(
        _pillar_quality_rows(
            "onchain",
            onchain_df,
            ASSET_ONCHAIN_FEATURE_COLS,
            source_metadata.get("onchain", {}),
            expected_dates_by_asset,
            required_feature_cols=ASSET_ONCHAIN_CORE_FEATURE_COLS,
            optional_feature_cols=ASSET_ONCHAIN_OPTIONAL_FEATURE_COLS,
        )
    )

    quality_df = pd.DataFrame(quality_rows)
    source_df = quality_df[
        ["pillar", "asset", "source_type", "is_real", "source_components", "errors"]
    ].copy()
    source_df.to_csv(DATA_SOURCE_REPORT_FILE, index=False)
    quality_df.to_csv(DATA_QUALITY_REPORT_FILE, index=False)

    paper_failures = quality_df[
        (quality_df["pillar"].isin(["sentiment", "onchain"]))
        & (~quality_df["passes_preflight"])
    ].copy()
    summary = {
        "paper_mode": PAPER_MODE,
        "overall_pass": paper_failures.empty,
        "failure_count": int(len(paper_failures)),
        "failures": paper_failures.to_dict(orient="records"),
    }
    with open(DATA_PREFLIGHT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    if PAPER_MODE and not paper_failures.empty:
        fail_list = ", ".join(
            f"{row['pillar']}:{row['asset']}[{row['source_type']}]"
            for _, row in paper_failures.iterrows()
        )
        raise RuntimeError(
            "Paper mode preflight failed. Non-real or low-quality inputs detected: "
            f"{fail_list}. See results/data_quality_report.csv and results/data_preflight_report.json."
        )

    return summary
