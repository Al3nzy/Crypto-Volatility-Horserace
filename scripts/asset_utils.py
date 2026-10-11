"""
Helpers for working with canonical asset tickers and aliases.
"""
from __future__ import annotations

from config import ASSET_METADATA, TICKERS


def get_asset_meta(ticker: str) -> dict:
    if ticker not in ASSET_METADATA:
        raise KeyError(f"Unknown asset ticker: {ticker}")
    return ASSET_METADATA[ticker]


def normalize_ticker(value: str | None) -> str | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    upper = raw.upper()
    if upper in ASSET_METADATA:
        return upper

    for ticker, meta in ASSET_METADATA.items():
        candidates = {ticker.upper(), meta["symbol"].upper(), meta["name"].upper()}
        candidates.update(alias.upper() for alias in meta.get("aliases", []))
        if upper in candidates:
            return ticker
    return None


def supported_tickers(include_optional: bool = False) -> list[str]:
    if include_optional:
        return list(ASSET_METADATA.keys())
    return list(TICKERS)


def ticker_suffix(ticker: str) -> str:
    return ticker.replace("-", "_")
