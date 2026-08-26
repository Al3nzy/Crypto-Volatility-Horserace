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


def fetch_market_data(ticker: str | None = None, use_api: bool = True) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance, or read from the local cache directly
    when use_api=False (or when the download fails/returns empty). Since
    START_DATE/END_DATE are fixed historical dates, a cached local CSV
    covering the same range is identical to a fresh download, so skipping
    the network call on repeat runs changes nothing about the resulting
    data -- only how long Phase 1 takes.
    """
    ticker = ticker or PRIMARY_TICKER
    df = None

    intraday_aliases = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if not use_api:
        print(f"[Pillar 1] use_api=False for {ticker}; using local cache …")
    try:
        if not use_api:
            pass
        elif MARKET_INTERVAL in intraday_aliases:
            print(
                f"[Pillar 1] Downloading {ticker} interval={MARKET_INTERVAL} "
                f"period={INTRADAY_PERIOD} …"
            )
            df = yf.download(
                ticker,
                period=INTRADAY_PERIOD,
                interval=MARKET_INTERVAL,
                auto_adjust=True,
                progress=False,
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
                progress=False,
            )
    except Exception as e:
        print(f"[Pillar 1] yfinance download failed for {ticker}: {e}")
        df = None

    # Fallback to local files if download failed or returned empty (essential for Code Ocean)
    if df is None or df.empty:
        for local_path in [FEATURE_MARKET_FILE, RAW_MARKET_FILE, MARKET_DATA_FILE]:
            if os.path.exists(local_path):
                print(f"[Pillar 1] Loading cached market data for {ticker} from {local_path} …")
                local_df = pd.read_csv(local_path, parse_dates=["Date"])
                if "Asset" in local_df.columns:
                    local_df = local_df[local_df["Asset"] == ticker]
                if not local_df.empty:
                    local_df.set_index("Date", inplace=True)
                    df = local_df
                    break

    if df is None or df.empty:
        raise ValueError(f"No market data available for {ticker} (network unavailable and no local cache found).")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    available_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[available_cols].copy()

    # Log-returns
    if "Log_Return" not in df.columns:
        if LOG_RETURNS:
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        else:
            df["Log_Return"] = df["Close"].pct_change()

    # Realised volatility (rolling std of returns)
    if "Volatility" not in df.columns:
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
