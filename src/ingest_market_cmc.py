"""
CoinMarketCap Pro API – daily OHLCV aligned with Yahoo Finance panels.

Requires env COINMARKETCAP_API_KEY. Uses /v1/cryptocurrency/ohlcv/historical
with backward pagination (time_end) to cover long ranges within API limits.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from config import END_DATE, START_DATE
from src.asset_utils import get_asset_meta
from src.collection_utils import normalize_dates

CMC_OHLCV_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
CMC_MAX_COUNT = 5000


def _cmc_headers() -> dict[str, str]:
    key = os.environ.get("COINMARKETCAP_API_KEY", "").strip()
    return {
        "Accept": "application/json",
        "X-CMC_PRO_API_KEY": key,
    }


def fetch_cmc_daily_ohlcv(
    ticker: str,
    request_timeout: int = 60,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Return DataFrame with Date, CMC_Close, CMC_Volume (quote volume in USD space).
    """
    key = os.environ.get("COINMARKETCAP_API_KEY", "").strip()
    if not key:
        return None, "missing_api_key"

    meta = get_asset_meta(ticker)
    cmc_id = meta.get("coinmarketcap_id")
    if cmc_id is None:
        return None, "missing_coinmarketcap_id"

    end_ts = pd.Timestamp(END_DATE).normalize() + pd.Timedelta(days=1)
    start_ts = pd.Timestamp(START_DATE).normalize()
    cursor_end = end_ts
    all_rows: list[dict] = []

    while cursor_end > start_ts:
        params = {
            "id": str(cmc_id),
            "convert": "USD",
            "count": str(CMC_MAX_COUNT),
            "time_end": cursor_end.isoformat(),
        }
        url = f"{CMC_OHLCV_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=_cmc_headers())
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace") if e.fp else ""
            return None, f"http_{e.code}:{body[:200]}"
        except Exception as e:
            return None, str(e)

        status = payload.get("status", {})
        if status.get("error_code", 0) != 0:
            return None, status.get("error_message", "cmc_error")

        data = payload.get("data") or {}
        quotes = data.get("quotes") or []
        if not quotes:
            break

        for q in quotes:
            t_open = q.get("time_open")
            if not t_open:
                continue
            dt = pd.to_datetime(t_open, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            dt = dt.tz_localize(None).normalize()
            quote = q.get("quote") or {}
            usd = quote.get("USD") or {}
            close = usd.get("close")
            vol = usd.get("volume")
            if close is None:
                continue
            all_rows.append(
                {
                    "Date": dt,
                    "CMC_Close": float(close),
                    "CMC_Volume": float(vol) if vol is not None else np.nan,
                }
            )

        earliest = min(pd.to_datetime(q.get("time_open"), utc=True) for q in quotes if q.get("time_open"))
        earliest = earliest.tz_localize(None).normalize()
        if earliest >= cursor_end.normalize():
            break
        cursor_end = earliest - pd.Timedelta(days=1)
        if len(quotes) < CMC_MAX_COUNT:
            break

    if not all_rows:
        return None, "empty_response"

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["Date"]).sort_values("Date")
    df["Date"] = normalize_dates(df["Date"])
    df = df[(df["Date"] >= str(start_ts.date())) & (df["Date"] <= END_DATE)]
    return df, None


def build_cmc_yf_validation_rows(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per (Date, Asset): log close ratio and relative volume gap for QA reporting.
    """
    df = market_df.copy()
    if "CMC_Close" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = normalize_dates(df["Date"])
    sub = df.dropna(subset=["Close", "CMC_Close"]).copy()
    if sub.empty:
        return pd.DataFrame()

    sub["log_close_ratio"] = np.log(sub["Close"] / sub["CMC_Close"])
    sub["vol_ratio_yf_cmc"] = np.where(
        (sub["Volume"] > 0) & (sub["CMC_Volume"] > 0),
        sub["Volume"] / sub["CMC_Volume"],
        np.nan,
    )
    return sub[
        ["Date", "Asset", "Close", "CMC_Close", "log_close_ratio", "Volume", "CMC_Volume", "vol_ratio_yf_cmc"]
    ]
