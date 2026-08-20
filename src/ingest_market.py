"""
Pillar 1 – Market Data Ingestion via yfinance.
Downloads daily OHLCV and computes realised volatility.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    FEATURE_MARKET_FILE,
    MARKET_DATA_FILE,
    PRIMARY_TICKER, START_DATE, END_DATE,
    MARKET_INTERVAL, INTRADAY_PERIOD,
    RAW_MARKET_FILE,
    VOLATILITY_WINDOW, LOG_RETURNS,
)
from src.collection_utils import normalize_dates


def fetch_market_data(ticker: str | None = None) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and derive volatility."""
    ticker = ticker or PRIMARY_TICKER

    intraday_aliases = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if MARKET_INTERVAL in intraday_aliases:
        print(
            f"[Pillar 1] Downloading {ticker} interval={MARKET_INTERVAL} "
            f"period={INTRADAY_PERIOD} …"
        )
        df = yf.download(
            ticker,
            period=INTRADAY_PERIOD,
            interval=MARKET_INTERVAL,
            auto_adjust=True,
        )
    else:
        print(
            f"[Pillar 1] Downloading {ticker} interval={MARKET_INTERVAL} "
            f"from {START_DATE} to {END_DATE} …"
        )
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval=MARKET_INTERVAL,
            auto_adjust=True,
        )

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Log-returns
    if LOG_RETURNS:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    else:
        df["Log_Return"] = df["Close"].pct_change()

    # Realised volatility (rolling std of returns)
    df["Volatility"] = df["Log_Return"].rolling(window=VOLATILITY_WINDOW).std()

    df.dropna(inplace=True)
    return df


def save_market_data(df: pd.DataFrame) -> str:
    out = df.copy()
    if out.index.name == "Date":
        out = out.reset_index()
    if "Date" in out.columns:
        out["Date"] = normalize_dates(out["Date"])

    for path in [RAW_MARKET_FILE, FEATURE_MARKET_FILE, MARKET_DATA_FILE]:
        out.to_csv(path, index=False)
    print(f"[Pillar 1] Saved -> {MARKET_DATA_FILE}  ({len(out)} rows)")
    return MARKET_DATA_FILE


if __name__ == "__main__":
    market_df = fetch_market_data()
    save_market_data(market_df)
