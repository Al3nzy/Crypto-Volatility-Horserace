"""
Pillar 3 – Asset-specific on-chain data.

Priority:
1) asset-specific raw on-chain CSVs,
2) optional vendor API only if explicitly enabled,
3) synthetic asset-specific fallback.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from config import (
    ENABLE_ONCHAIN_API,
    END_DATE,
    FEATURE_ONCHAIN_FILE,
    ONCHAIN_PROVIDER,
    ONCHAIN_DATA_FILE,
    RAW_ONCHAIN_COINMETRICS_FILE,
    RAW_ONCHAIN_DIR,
    RAW_ONCHAIN_LEGACY_DIR,
    RAW_ONCHAIN_LEGACY_FILE,
    SEED,
    START_DATE,
    TICKERS,
)
from src.asset_utils import get_asset_meta, normalize_ticker
from src.collection_utils import normalize_dates


GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics"
COINMETRICS_COMMUNITY_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
COINMETRICS_ENTERPRISE_URL = "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
SUPPORTED_API_PROVIDERS = {"glassnode", "coinmetrics_community"}
ONCHAIN_COLS = [
    "Active_Addresses",
    "Tx_Count",
    "Transfer_Volume_USD",
    "Hash_Rate",
    "Network_Cap_USD",
]

GLASSNODE_METRICS = {
    "Active_Addresses": "addresses/active_count",
    "Tx_Count": "transactions/count",
    "Transfer_Volume_USD": "transactions/transfers_volume_sum",
}
COINMETRICS_METRICS = {
    "Active_Addresses": "AdrActCnt",
    "Tx_Count": "TxCnt",
}
COINMETRICS_OPTIONAL_METRICS = {
    "Transfer_Volume_USD": "TxTfrValAdjUSD",
}
COINMETRICS_EXTENDED_METRICS = {
    "Hash_Rate": "HashRt",
    "Network_Cap_USD": "CapMrktCurUSD",
}


def _coinmetrics_request(url: str, headers: dict | None = None) -> urllib.request.Request:
    h = {"User-Agent": "crypto-horserace/1.0"}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h)


def _coinmetrics_endpoint_and_headers() -> tuple[str, dict | None]:
    key = os.environ.get("COINMETRICS_API_KEY", "").strip()
    if key:
        return COINMETRICS_ENTERPRISE_URL, {"Authorization": f"Bearer {key}"}
    return COINMETRICS_COMMUNITY_URL, None

SYNTHETIC_BASE_LEVELS = {
    "BTC-USD": (850_000, 275_000, 5_500_000_000.0),
    "ETH-USD": (500_000, 700_000, 4_200_000_000.0),
    "DOGE-USD": (85_000, 65_000, 450_000_000.0),
    "XRP-USD": (120_000, 140_000, 900_000_000.0),
    "SOL-USD": (170_000, 300_000, 1_300_000_000.0),
    "SHIB-USD": (95_000, 110_000, 350_000_000.0),
}


def _finalize_onchain_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = normalize_dates(df["Date"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    for col in ONCHAIN_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["Active_Addresses"] = pd.to_numeric(df["Active_Addresses"], errors="coerce")
    df["Tx_Count"] = pd.to_numeric(df["Tx_Count"], errors="coerce")
    df["Transfer_Volume_USD"] = pd.to_numeric(df["Transfer_Volume_USD"], errors="coerce")
    df["Hash_Rate"] = pd.to_numeric(df["Hash_Rate"], errors="coerce")
    df["Network_Cap_USD"] = pd.to_numeric(df["Network_Cap_USD"], errors="coerce")
    return df[["Date", "Asset"] + ONCHAIN_COLS].sort_values(["Asset", "Date"])


def _load_raw_onchain(tickers: list[str]) -> pd.DataFrame | None:
    candidate_paths = []
    for path in [RAW_ONCHAIN_COINMETRICS_FILE, RAW_ONCHAIN_LEGACY_FILE]:
        if os.path.exists(path):
            candidate_paths.append(path)

    for raw_dir in [RAW_ONCHAIN_DIR, RAW_ONCHAIN_LEGACY_DIR]:
        if os.path.isdir(raw_dir):
            for name in os.listdir(raw_dir):
                if name.lower().endswith(".csv") and "template" not in name.lower():
                    candidate_paths.append(os.path.join(raw_dir, name))

    frames = []
    for path in candidate_paths:
        raw = pd.read_csv(path)
        if "Asset" not in raw.columns:
            continue
        raw["Asset"] = raw["Asset"].map(normalize_ticker)
        raw = raw[raw["Asset"].isin(tickers)]
        if raw.empty:
            continue
        frames.append(_finalize_onchain_frame(raw))

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby(["Date", "Asset"], as_index=False).last()
    return merged


def _fetch_glassnode_metric(
    asset_symbol: str,
    metric_path: str,
    request_timeout: int = 60,
) -> tuple[pd.DataFrame | None, str | None]:
    api_key = os.environ.get("GLASSNODE_API_KEY", "").strip()
    if not api_key:
        return None, "missing_api_key"

    params = {
        "a": asset_symbol,
        "api_key": api_key,
        "s": int(pd.Timestamp(START_DATE).timestamp()),
        "u": int(pd.Timestamp(END_DATE).timestamp()),
        "i": "24h",
    }
    url = f"{GLASSNODE_BASE_URL}/{metric_path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-horserace/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[Pillar 3] Glassnode API error for {asset_symbol}: HTTP {e.code}")
        return None, f"http_{e.code}"
    except Exception as e:
        print(f"[Pillar 3] Glassnode API error for {asset_symbol}: {e}")
        return None, str(e)

    if not isinstance(payload, list) or not payload:
        return None, "empty_response"

    out = pd.DataFrame(payload)
    if "t" not in out.columns or "v" not in out.columns:
        return None, "invalid_schema"
    out["Date"] = pd.to_datetime(out["t"], unit="s", errors="coerce").dt.normalize()
    return out[["Date", "v"]].dropna(subset=["Date"]), None


def _fetch_glassnode_onchain(
    ticker: str,
    request_timeout: int = 60,
) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch a common comparable metric subset from Glassnode."""
    meta = get_asset_meta(ticker)
    asset_symbol = meta.get("glassnode_asset")
    if not asset_symbol:
        return None, "unsupported_asset"

    merged = None
    errors = {}
    for col, metric_path in GLASSNODE_METRICS.items():
        metric_df, metric_error = _fetch_glassnode_metric(
            asset_symbol=asset_symbol,
            metric_path=metric_path,
            request_timeout=request_timeout,
        )
        if metric_error:
            errors[col] = metric_error
            continue
        metric_df = metric_df.rename(columns={"v": col})
        merged = metric_df if merged is None else merged.merge(metric_df, on="Date", how="outer")

    if merged is None or merged.empty:
        if errors:
            return None, json.dumps(errors, ensure_ascii=True)
        return None, "empty_response"

    merged["Asset"] = ticker
    keep_cols = ["Date", "Asset"] + ONCHAIN_COLS
    for col in ONCHAIN_COLS:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged[keep_cols].dropna(subset=["Date"]), None


def _fetch_coinmetrics_onchain(
    ticker: str,
    request_timeout: int = 60,
) -> tuple[pd.DataFrame | None, str | None]:
    meta = get_asset_meta(ticker)
    asset_symbol = meta.get("coinmetrics_asset")
    if not asset_symbol:
        return None, "unsupported_asset"

    merged = None
    errors = {}
    all_metrics = {
        **COINMETRICS_METRICS,
        **COINMETRICS_OPTIONAL_METRICS,
        **COINMETRICS_EXTENDED_METRICS,
    }
    base_url, extra_headers = _coinmetrics_endpoint_and_headers()

    for final_col, raw_col in all_metrics.items():
        params = {
            "assets": asset_symbol,
            "metrics": raw_col,
            "start_time": START_DATE,
            "end_time": END_DATE,
            "frequency": "1d",
            "page_size": 10000,
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = _coinmetrics_request(url, extra_headers)
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            errors[final_col] = f"http_{e.code}"
            continue
        except Exception as e:
            errors[final_col] = str(e)
            continue

        rows = payload.get("data", [])
        if not rows:
            errors[final_col] = "empty_response"
            continue

        metric_df = pd.DataFrame(rows)
        if "time" not in metric_df.columns or raw_col not in metric_df.columns:
            errors[final_col] = "invalid_schema"
            continue

        metric_df["Date"] = normalize_dates(metric_df["time"])
        metric_df[final_col] = pd.to_numeric(metric_df[raw_col], errors="coerce")
        metric_df = metric_df[["Date", final_col]].dropna(subset=["Date"])
        merged = metric_df if merged is None else merged.merge(metric_df, on="Date", how="outer")

    if merged is None or merged.empty:
        if errors:
            print(f"[Pillar 3] Coin Metrics Community unavailable for {asset_symbol}: {errors}")
            return None, json.dumps(errors, ensure_ascii=True)
        return None, "empty_response"

    merged["Asset"] = ticker
    for col in ONCHAIN_COLS:
        if col not in merged.columns:
            merged[col] = np.nan
    if errors:
        print(f"[Pillar 3] Coin Metrics partial coverage for {asset_symbol}: {errors}")
    return merged[["Date", "Asset"] + ONCHAIN_COLS].dropna(subset=["Date"]), None


def _generate_synthetic_onchain(
    tickers: list[str],
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fallback on-chain features that remain asset-specific."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    frames = []

    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng(SEED + 100 + i)
        base_addr, base_tx, base_vol = SYNTHETIC_BASE_LEVELS.get(
            ticker, (120_000, 80_000, 600_000_000.0)
        )
        panel = pd.DataFrame({"Date": dates, "Asset": ticker})

        if market_df is not None and "Asset" in market_df.columns:
            market = market_df[market_df["Asset"] == ticker].copy()
            market = market.reset_index()[["Date", "Log_Return", "Volume"]]
            market["Date"] = pd.to_datetime(market["Date"]).dt.normalize()
            panel = panel.merge(market, on="Date", how="left")
        else:
            panel["Log_Return"] = 0.0
            panel["Volume"] = 0.0

        panel["Log_Return"] = panel["Log_Return"].fillna(0.0)
        panel["Volume"] = panel["Volume"].ffill().fillna(0.0)
        turnover = np.log1p(panel["Volume"])
        move_mag = np.abs(panel["Log_Return"]).clip(0, 0.25)

        panel["Active_Addresses"] = np.clip(
            base_addr * (1 + 0.15 * move_mag + 0.02 * turnover / (turnover.std() + 1e-6))
            + rng.normal(0, base_addr * 0.04, len(panel)),
            1_000,
            None,
        ).round()
        panel["Tx_Count"] = np.clip(
            base_tx * (1 + 0.20 * move_mag + 0.03 * turnover / (turnover.std() + 1e-6))
            + rng.normal(0, base_tx * 0.05, len(panel)),
            500,
            None,
        ).round()
        panel["Transfer_Volume_USD"] = np.clip(
            base_vol * (1 + 0.80 * move_mag) + panel["Volume"] * rng.uniform(0.4, 1.0),
            10_000,
            None,
        )
        panel["Hash_Rate"] = np.clip(
            panel["Active_Addresses"] * 2e15 * (1 + 0.1 * move_mag)
            + rng.normal(0, panel["Active_Addresses"] * 1e13, len(panel)),
            1e12,
            None,
        )
        panel["Network_Cap_USD"] = np.clip(
            panel["Transfer_Volume_USD"] * rng.uniform(0.08, 0.22, len(panel))
            * (1 + 0.5 * np.abs(panel["Log_Return"])),
            1e6,
            None,
        )
        frames.append(panel[["Date", "Asset"] + ONCHAIN_COLS])

    return pd.concat(frames, ignore_index=True)


def load_onchain_data(
    tickers: list[str] | None = None,
    use_api: bool = ENABLE_ONCHAIN_API,
    market_df: pd.DataFrame | None = None,
    return_metadata: bool = False,
    request_timeout: int = 60,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Load asset-specific on-chain features keyed by Date and Asset.
    Priority:
    1) raw asset-specific exports
    2) optional vendor API if configured
    3) synthetic fallback
    """
    tickers = tickers or TICKERS
    metadata = {
        "pillar": "onchain",
        "per_asset": {},
    }

    raw = _load_raw_onchain(tickers)
    raw_assets = set()
    if raw is not None:
        print("[Pillar 3] Loading asset-specific raw on-chain CSV …")
        raw = _finalize_onchain_frame(raw)
        raw_assets = set(raw["Asset"].unique())

    frames = [raw] if raw is not None else []

    if raw is not None:
        for ticker in tickers:
            if ticker in raw_assets:
                asset_rows = raw[raw["Asset"] == ticker]
                metadata["per_asset"][ticker] = {
                    "source_type": "raw_archive",
                    "source_components": ["raw_archive"],
                    "is_real": True,
                    "errors": {},
                    "row_count": int(len(asset_rows)),
                    "date_min": str(asset_rows["Date"].min().date()) if not asset_rows.empty else None,
                    "date_max": str(asset_rows["Date"].max().date()) if not asset_rows.empty else None,
                }

    if use_api:
        if ONCHAIN_PROVIDER in SUPPORTED_API_PROVIDERS:
            print(f"[Pillar 3] Fetching asset-specific on-chain data ({ONCHAIN_PROVIDER}) …")
        else:
            print(
                "[Pillar 3] No free/default on-chain API configured; "
                "expecting raw local exports."
            )
    for ticker in tickers:
        if ticker in raw_assets:
            continue
        errors = {}
        source_components = []
        source_type = "missing" if raw is not None else "synthetic"
        asset_df = None

        if use_api and ONCHAIN_PROVIDER in SUPPORTED_API_PROVIDERS:
            if ONCHAIN_PROVIDER == "glassnode":
                asset_df, api_error = _fetch_glassnode_onchain(
                    ticker,
                    request_timeout=request_timeout,
                )
            elif ONCHAIN_PROVIDER == "coinmetrics_community":
                asset_df, api_error = _fetch_coinmetrics_onchain(
                    ticker,
                    request_timeout=request_timeout,
                )
            else:
                asset_df, api_error = None, f"unsupported_provider:{ONCHAIN_PROVIDER}"
            if api_error:
                errors[ONCHAIN_PROVIDER] = api_error
            if asset_df is not None and not asset_df.empty:
                source_type = "api"
                source_components.append(ONCHAIN_PROVIDER)
        elif use_api and ONCHAIN_PROVIDER not in SUPPORTED_API_PROVIDERS:
            errors["onchain_api"] = f"provider_disabled:{ONCHAIN_PROVIDER}"

        if source_type == "synthetic":
            asset_df = _generate_synthetic_onchain([ticker], market_df=market_df)

        if asset_df is not None and not asset_df.empty:
            frames.append(asset_df)
        metadata["per_asset"][ticker] = {
            "source_type": source_type,
            "source_components": source_components,
            "is_real": source_type != "synthetic",
            "errors": errors,
        }

    if not frames:
        out = pd.DataFrame(columns=["Date", "Asset"] + ONCHAIN_COLS)
    else:
        out = _finalize_onchain_frame(pd.concat(frames, ignore_index=True))
    if all(not meta["is_real"] for meta in metadata["per_asset"].values()):
        print("[Pillar 3] Falling back to synthetic asset-specific on-chain …")
    for ticker in tickers:
        asset_rows = out[out["Asset"] == ticker]
        metadata["per_asset"][ticker]["row_count"] = int(len(asset_rows))
        metadata["per_asset"][ticker]["date_min"] = (
            str(asset_rows["Date"].min().date()) if not asset_rows.empty else None
        )
        metadata["per_asset"][ticker]["date_max"] = (
            str(asset_rows["Date"].max().date()) if not asset_rows.empty else None
        )
    return (out, metadata) if return_metadata else out


def save_onchain_data(df: pd.DataFrame) -> str:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = normalize_dates(out["Date"])
    for path in [FEATURE_ONCHAIN_FILE, ONCHAIN_DATA_FILE]:
        out.to_csv(path, index=False)
    print(f"[Pillar 3] Saved -> {ONCHAIN_DATA_FILE}  ({len(out)} rows)")
    return ONCHAIN_DATA_FILE


if __name__ == "__main__":
    onchain_df = load_onchain_data()
    save_onchain_data(onchain_df)
