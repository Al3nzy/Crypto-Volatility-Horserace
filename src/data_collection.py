"""
Canonical datatype-first data-collection orchestration.

This module keeps market, sentiment, and on-chain collection separate while
still producing the final daily feature panels consumed by fusion/training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ENABLE_COINMARKETCAP,
    ENABLE_GOOGLE_TRENDS,
    ENABLE_ONCHAIN_API,
    MARKET_CMC_FEATURE_COLS,
    MARKET_CMC_VALIDATION_FILE,
    TICKERS,
)
from src.ingest_market import fetch_market_data, save_market_data
from src.ingest_market_cmc import build_cmc_yf_validation_rows, fetch_cmc_daily_ohlcv
from src.ingest_onchain import load_onchain_data, save_onchain_data
from src.ingest_sentiment import load_sentiment_data, save_sentiment_data


def collect_market_dataset(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or TICKERS
    market_frames = []
    for ticker in tickers:
        df_t = fetch_market_data(ticker=ticker)
        df_t = df_t.reset_index()
        df_t["Asset"] = ticker
        if ENABLE_COINMARKETCAP:
            cmc_df, cmc_err = fetch_cmc_daily_ohlcv(ticker)
            if cmc_df is not None and not cmc_df.empty:
                df_t = df_t.merge(cmc_df, on="Date", how="left")
            else:
                print(f"[Pillar 1] CoinMarketCap OHLCV unavailable for {ticker}: {cmc_err}")
                for col in MARKET_CMC_FEATURE_COLS:
                    df_t[col] = np.nan
        market_frames.append(df_t)

    market_df = pd.concat(market_frames, ignore_index=True)
    market_df = market_df.sort_values(["Asset", "Date"]).reset_index(drop=True)

    if ENABLE_COINMARKETCAP and all(c in market_df.columns for c in MARKET_CMC_FEATURE_COLS):
        if market_df["CMC_Close"].isna().all():
            print(
                "[Pillar 1] CoinMarketCap columns are all empty; dropping them "
                "(check COINMARKETCAP_API_KEY)."
            )
            market_df = market_df.drop(columns=list(MARKET_CMC_FEATURE_COLS), errors="ignore")
        else:
            ok_per_asset = market_df.groupby("Asset")["CMC_Close"].apply(
                lambda s: bool(s.notna().any())
            )
            if not bool(ok_per_asset.all()):
                print(
                    "[Pillar 1] CoinMarketCap incomplete across assets; dropping CMC columns "
                    "for a consistent cross-asset schema."
                )
                market_df = market_df.drop(columns=list(MARKET_CMC_FEATURE_COLS), errors="ignore")

    if ENABLE_COINMARKETCAP and all(c in market_df.columns for c in MARKET_CMC_FEATURE_COLS):
        val = build_cmc_yf_validation_rows(market_df)
        if not val.empty:
            val.to_csv(MARKET_CMC_VALIDATION_FILE, index=False)
            print(f"[Pillar 1] CMC vs Yahoo validation -> {MARKET_CMC_VALIDATION_FILE}")

    save_market_data(market_df)
    return market_df


def collect_feature_datasets(
    tickers: list[str] | None = None,
    use_api: bool = True,
    request_timeout: int = 20,
    use_google_trends: bool = ENABLE_GOOGLE_TRENDS,
    use_onchain_api: bool = ENABLE_ONCHAIN_API,
) -> dict:
    tickers = tickers or TICKERS
    market_df = collect_market_dataset(tickers=tickers)

    sentiment_df, sentiment_meta = load_sentiment_data(
        tickers=tickers,
        market_df=market_df,
        return_metadata=True,
        use_api=use_api,
        request_timeout=request_timeout,
        use_google_trends=use_google_trends and use_api,
    )
    save_sentiment_data(sentiment_df)

    onchain_df, onchain_meta = load_onchain_data(
        tickers=tickers,
        market_df=market_df,
        return_metadata=True,
        use_api=use_api and use_onchain_api,
        request_timeout=request_timeout,
    )
    save_onchain_data(onchain_df)

    return {
        "market_df": market_df,
        "sentiment_df": sentiment_df,
        "onchain_df": onchain_df,
        "metadata": {
            "sentiment": sentiment_meta,
            "onchain": onchain_meta,
        },
    }
