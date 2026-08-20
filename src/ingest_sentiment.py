"""
Pillar 2 – Asset-specific sentiment data.

Priority:
1) Asset-specific raw archives in CSV form.
2) Coin-tagged news (CryptoPanic) + Google Trends attention.
3) Synthetic asset-specific fallback.

Fear & Greed remains in the dataset only as a market-wide control feature.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from config import (
    ENABLE_FINBERT_ON_INGEST,
    ENABLE_GOOGLE_TRENDS,
    END_DATE,
    FEATURE_SENTIMENT_FILE,
    FEATURE_SENTIMENT_FINBERT_FILE,
    RAW_SENTIMENT_ARCHIVE_DIR,
    RAW_SENTIMENT_CORPUS_DIR,
    RAW_SENTIMENT_CRYPTOPANIC_FILE,
    RAW_SENTIMENT_DIR,
    RAW_SENTIMENT_LEGACY_FILE,
    SEED,
    SENTIMENT_DATA_FILE,
    START_DATE,
    TICKERS,
)
from src.asset_utils import get_asset_meta, normalize_ticker
from src.collection_utils import normalize_dates

try:
    from pytrends.dailydata import get_daily_data
except Exception:  # pragma: no cover - optional dependency at runtime
    get_daily_data = None

FNG_API_URL = "https://api.alternative.me/fng/"
CRYPTOPANIC_API_URL = "https://cryptopanic.com/api/v1/posts/"

SENTIMENT_COLS = [
    "News_Sentiment",
    "News_Volume",
    "Social_Sentiment",
    "Social_Volume",
    "Attention_Index",
    "Sentiment_Disagreement",
    "Market_FearGreed",
    "FinBERT_Polarity",
]

POSITIVE_TERMS = {
    "adoption", "approve", "approved", "breakout", "bullish", "gain", "growth",
    "higher", "partnership", "pump", "rally", "record", "recovery", "surge",
    "uptrend", "upgrade", "whale buy",
}
NEGATIVE_TERMS = {
    "bearish", "crackdown", "crash", "decline", "dump", "exploit", "fear",
    "hack", "lawsuit", "liquidation", "lower", "outflow", "selloff", "slump",
    "token unlock", "whale sell",
}


def _fetch_fear_greed_index(limit: int = 10000, request_timeout: int = 30) -> pd.Series | None:
    """Fetch market-wide Fear & Greed control and map to [-1, 1]."""
    url = f"{FNG_API_URL}?limit={limit}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=request_timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"[Pillar 2] Fear & Greed API error: {e}")
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    records = []
    for row in rows:
        ts = int(row.get("timestamp", 0))
        value = float(row.get("value", 50))
        records.append(
            {
                "Date": pd.Timestamp(ts, unit="s").normalize(),
                "Market_FearGreed": max(-1.0, min(1.0, (value / 50.0) - 1.0)),
            }
        )

    out = pd.DataFrame(records).drop_duplicates("Date").sort_values("Date")
    return out.set_index("Date")["Market_FearGreed"]


def _score_text_sentiment(text: str | None) -> float:
    """Small lexical scorer used when raw sources do not provide labels."""
    if not text:
        return 0.0

    lowered = str(text).lower()
    positive = sum(1 for term in POSITIVE_TERMS if term in lowered)
    negative = sum(1 for term in NEGATIVE_TERMS if term in lowered)
    tokens = max(1, len(re.findall(r"[a-zA-Z]+", lowered)))
    score = (positive - negative) / max(3.0, np.sqrt(tokens))
    return float(np.clip(score, -1.0, 1.0))


def _source_bucket(source_value: str | None) -> str:
    source = str(source_value or "").lower()
    social_markers = ("reddit", "twitter", "x.com", "social", "telegram", "discord")
    if any(marker in source for marker in social_markers):
        return "social"
    return "news"


def _standardize_attention(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - minimum) / (maximum - minimum) * 100.0).astype(float)


def _append_cryptopanic_corpus_row(record: dict) -> None:
    os.makedirs(RAW_SENTIMENT_CORPUS_DIR, exist_ok=True)
    path = os.path.join(RAW_SENTIMENT_CORPUS_DIR, "cryptopanic_corpus.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_finbert_daily_merge(tickers: list[str]) -> pd.DataFrame | None:
    if not os.path.isfile(FEATURE_SENTIMENT_FINBERT_FILE):
        return None
    fb = pd.read_csv(FEATURE_SENTIMENT_FINBERT_FILE, parse_dates=["Date"])
    fb["Date"] = normalize_dates(fb["Date"])
    fb["Asset"] = fb["Asset"].astype(str)
    fb = fb[fb["Asset"].isin(tickers)]
    if fb.empty or "FinBERT_Polarity" not in fb.columns:
        return None
    return fb[["Date", "Asset", "FinBERT_Polarity"]].drop_duplicates(["Date", "Asset"])


def _finalize_sentiment_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = normalize_dates(df["Date"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    for col in SENTIMENT_COLS:
        if col not in df.columns:
            df[col] = 0.0
    volume_cols = ["News_Volume", "Social_Volume"]
    for col in volume_cols:
        df[col] = df[col].fillna(0).astype(int)
    for col in set(SENTIMENT_COLS) - set(volume_cols):
        df[col] = df[col].fillna(0.0).astype(float)
    return df[["Date", "Asset"] + SENTIMENT_COLS].sort_values(["Asset", "Date"])


def _aggregate_raw_sentiment(df_raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame | None:
    """Aggregate asset-specific raw archives into daily features."""
    asset_col = next((c for c in df_raw.columns if c.lower() in {"asset", "ticker", "symbol", "coin"}), None)
    if asset_col is None:
        return None

    date_col = next(
        (
            c
            for c in df_raw.columns
            if c.lower() in {"date", "datetime", "published_at", "created_at", "timestamp", "time"}
        ),
        None,
    )
    if date_col is None:
        return None

    source_col = next((c for c in df_raw.columns if c.lower() in {"source", "platform", "channel"}), None)
    sentiment_col = next(
        (
            c
            for c in df_raw.columns
            if c.lower() in {"sentiment", "sentimentscore", "sentiment_score", "score", "compound"}
        ),
        None,
    )
    text_cols = [c for c in df_raw.columns if c.lower() in {"title", "headline", "text", "body", "content"}]
    engagement_col = next(
        (
            c
            for c in df_raw.columns
            if c.lower() in {"engagement", "upvotes", "comments", "likes", "retweets", "importance"}
        ),
        None,
    )
    attention_col = next(
        (
            c
            for c in df_raw.columns
            if c.lower() in {"attention_index", "google_trends", "trends", "attention"}
        ),
        None,
    )

    df = df_raw.copy()
    df["Asset"] = df[asset_col].map(normalize_ticker)
    df["Date"] = normalize_dates(df[date_col])
    df = df[df["Asset"].isin(tickers) & df["Date"].notna()].copy()
    if df.empty:
        return None

    if sentiment_col is not None and pd.api.types.is_numeric_dtype(df[sentiment_col]):
        df["sentiment_value"] = df[sentiment_col].astype(float).clip(-1, 1)
    else:
        combined_text = pd.Series([""] * len(df), index=df.index, dtype=object)
        if text_cols:
            combined_text = df[text_cols].fillna("").agg(" ".join, axis=1)
        df["sentiment_value"] = combined_text.map(_score_text_sentiment)

    df["source_bucket"] = df[source_col].map(_source_bucket) if source_col else "news"
    df["attention_weight"] = (
        pd.to_numeric(df[engagement_col], errors="coerce").fillna(1.0).clip(lower=1.0)
        if engagement_col
        else 1.0
    )

    rows = []
    for (date, asset), grp in df.groupby(["Date", "Asset"]):
        news = grp[grp["source_bucket"] == "news"]
        social = grp[grp["source_bucket"] == "social"]
        rows.append(
            {
                "Date": date,
                "Asset": asset,
                "News_Sentiment": float(np.average(news["sentiment_value"], weights=news["attention_weight"])) if len(news) else 0.0,
                "News_Volume": int(len(news)),
                "Social_Sentiment": float(np.average(social["sentiment_value"], weights=social["attention_weight"])) if len(social) else 0.0,
                "Social_Volume": int(len(social)),
                "Sentiment_Disagreement": float(grp["sentiment_value"].std(ddof=0) or 0.0),
                "Attention_Index": float(pd.to_numeric(grp[attention_col], errors="coerce").mean()) if attention_col else float(len(grp)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return None

    out["Attention_Index"] = (
        out.groupby("Asset", group_keys=False)["Attention_Index"].apply(_standardize_attention)
    )
    return out


def _load_raw_sentiment_archives(tickers: list[str]) -> pd.DataFrame | None:
    frames = []
    direct_paths = [
        RAW_SENTIMENT_CRYPTOPANIC_FILE,
        RAW_SENTIMENT_LEGACY_FILE,
    ]
    for path in direct_paths:
        if os.path.exists(path):
            print("[Pillar 2] Loading asset-specific raw sentiment CSV …")
            raw = pd.read_csv(path)
            agg = _aggregate_raw_sentiment(raw, tickers)
            if agg is not None:
                frames.append(agg)

    for raw_dir in [RAW_SENTIMENT_DIR, RAW_SENTIMENT_ARCHIVE_DIR]:
        if not os.path.isdir(raw_dir):
            continue
        for name in os.listdir(raw_dir):
            if not name.lower().endswith(".csv"):
                continue
            if "template" in name.lower():
                continue
            path = os.path.join(raw_dir, name)
            if os.path.abspath(path) == os.path.abspath(RAW_SENTIMENT_CRYPTOPANIC_FILE):
                continue
            raw = pd.read_csv(os.path.join(raw_dir, name))
            agg = _aggregate_raw_sentiment(raw, tickers)
            if agg is not None:
                frames.append(agg)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby(["Date", "Asset"], as_index=False).agg(
        {
            "News_Sentiment": "mean",
            "News_Volume": "sum",
            "Social_Sentiment": "mean",
            "Social_Volume": "sum",
            "Attention_Index": "mean",
            "Sentiment_Disagreement": "mean",
        }
    )
    return merged


def _fetch_cryptopanic_news(
    ticker: str,
    max_pages: int = 30,
    request_timeout: int = 30,
) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch coin-tagged news and convert it into daily features."""
    auth_token = os.environ.get("CRYPTOPANIC_API_KEY", "").strip()
    if not auth_token:
        return None, "missing_api_key"

    meta = get_asset_meta(ticker)
    params = {
        "auth_token": auth_token,
        "kind": "news",
        "public": "true",
        "currencies": meta["cryptopanic_currency"],
    }
    next_url = CRYPTOPANIC_API_URL + "?" + urllib.parse.urlencode(params)
    rows = []

    for _ in range(max_pages):
        try:
            with urllib.request.urlopen(next_url, timeout=request_timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            print(f"[Pillar 2] CryptoPanic API error for {ticker}: HTTP {e.code}")
            return None, f"http_{e.code}"
        except Exception as e:
            print(f"[Pillar 2] CryptoPanic API error for {ticker}: {e}")
            return None, str(e)

        posts = payload.get("results", [])
        if not posts:
            break

        for post in posts:
            published_at = pd.to_datetime(post.get("published_at"), errors="coerce", utc=True)
            if pd.isna(published_at):
                continue
            published_at = published_at.tz_localize(None).normalize()
            if published_at < pd.to_datetime(START_DATE):
                next_url = None
                break
            title = post.get("title", "")
            _append_cryptopanic_corpus_row(
                {
                    "id": post.get("id"),
                    "asset": ticker,
                    "text": title,
                    "published_at": published_at.isoformat(),
                    "source": "cryptopanic",
                }
            )
            rows.append(
                {
                    "Date": published_at,
                    "Asset": ticker,
                    "News_Sentiment": _score_text_sentiment(title),
                    "News_Volume": 1,
                }
            )

        if next_url is None:
            break
        next_url = payload.get("next")
        if not next_url:
            break

    if not rows:
        return None, "empty_response"

    out = pd.DataFrame(rows)
    out = out.groupby(["Date", "Asset"], as_index=False).agg(
        {"News_Sentiment": "mean", "News_Volume": "sum"}
    )
    return out, None


def _fetch_google_trends_attention(
    ticker: str,
    request_timeout: int = 30,
) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch daily asset attention from Google Trends via pytrends."""
    if get_daily_data is None:
        return None, "pytrends_unavailable"

    meta = get_asset_meta(ticker)
    term = meta["trend_terms"][0]
    try:
        trend = get_daily_data(
            word=term,
            start_year=pd.Timestamp(START_DATE).year,
            start_mon=pd.Timestamp(START_DATE).month,
            stop_year=pd.Timestamp(END_DATE).year,
            stop_mon=pd.Timestamp(END_DATE).month,
            geo="",
        )
    except Exception as e:
        print(f"[Pillar 2] Google Trends error for {ticker}: {e}")
        return None, str(e)

    if trend.empty:
        return None, "empty_response"

    col = next((c for c in trend.columns if term in c or c == term), None)
    if col is None:
        numeric_cols = [c for c in trend.columns if pd.api.types.is_numeric_dtype(trend[c])]
        if not numeric_cols:
            return None, "no_numeric_series"
        col = numeric_cols[0]

    out = trend.reset_index().rename(columns={"date": "Date", col: "Attention_Index"})
    out["Date"] = normalize_dates(out["Date"])
    out["Asset"] = ticker
    out["Attention_Index"] = _standardize_attention(out["Attention_Index"].astype(float))
    return out[["Date", "Asset", "Attention_Index"]], None


def _generate_synthetic_sentiment(
    tickers: list[str],
    market_df: pd.DataFrame | None = None,
    fear_greed: pd.Series | None = None,
) -> pd.DataFrame:
    """Fallback sentiment that remains asset-specific when APIs are unavailable."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    frames = []

    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng(SEED + i)
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
        vol_proxy = _standardize_attention(np.log1p(panel["Volume"]))
        move_proxy = panel["Log_Return"].rolling(3, min_periods=1).sum().clip(-0.2, 0.2)

        panel["News_Sentiment"] = np.clip(-2.2 * move_proxy + rng.normal(0, 0.10, len(panel)), -1, 1)
        panel["Social_Sentiment"] = np.clip(-2.8 * move_proxy + rng.normal(0, 0.14, len(panel)), -1, 1)
        panel["News_Volume"] = (15 + 0.20 * vol_proxy + 120 * np.abs(panel["Log_Return"])).round().astype(int)
        panel["Social_Volume"] = (20 + 0.25 * vol_proxy + 180 * np.abs(panel["Log_Return"])).round().astype(int)
        panel["Attention_Index"] = _standardize_attention(
            0.6 * vol_proxy + 40 * np.abs(panel["Log_Return"]) + rng.normal(0, 3, len(panel))
        )
        panel["Sentiment_Disagreement"] = np.clip(
            0.05 + 1.5 * np.abs(panel["Log_Return"]) + rng.normal(0, 0.02, len(panel)),
            0,
            1,
        )
        panel["FinBERT_Polarity"] = np.clip(
            -1.2 * move_proxy + rng.normal(0, 0.08, len(panel)),
            -1,
            1,
        )
        if fear_greed is not None:
            panel["Market_FearGreed"] = panel["Date"].map(fear_greed).ffill().fillna(0.0)
        else:
            panel["Market_FearGreed"] = 0.0

        frames.append(panel[["Date", "Asset"] + SENTIMENT_COLS])

    return pd.concat(frames, ignore_index=True)


def load_sentiment_data(
    tickers: list[str] | None = None,
    use_api: bool = True,
    market_df: pd.DataFrame | None = None,
    return_metadata: bool = False,
    request_timeout: int = 30,
    use_google_trends: bool = ENABLE_GOOGLE_TRENDS,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Load asset-specific sentiment features for each ticker-day.
    Priority:
    1) raw sentiment archives
    2) CryptoPanic + Google Trends + Fear & Greed control
    3) synthetic asset-specific fallback
    """
    tickers = tickers or TICKERS
    if ENABLE_FINBERT_ON_INGEST:
        try:
            from src.sentiment_finbert import build_finbert_daily_panel

            build_finbert_daily_panel(tickers=tickers)
        except ImportError as e:
            print(f"[Pillar 2] FinBERT ingest skipped (missing dependency): {e}")
        except Exception as e:
            print(f"[Pillar 2] FinBERT ingest failed: {e}")

    fear_greed = _fetch_fear_greed_index(request_timeout=request_timeout) if use_api else None
    metadata = {
        "pillar": "sentiment",
        "per_asset": {},
        "uses_marketwide_control": fear_greed is not None,
        "finbert_panel_path": FEATURE_SENTIMENT_FINBERT_FILE,
    }

    raw = _load_raw_sentiment_archives(tickers)
    raw_assets = set(raw["Asset"].unique()) if raw is not None and not raw.empty else set()
    if raw is not None:
        print("[Pillar 2] Aggregated asset-specific sentiment archives.")

    frames = []
    base_dates = pd.date_range(START_DATE, END_DATE, freq="D")

    if use_api and raw_assets != set(tickers):
        print("[Pillar 2] Fetching asset-specific news / attention data …")

    finbert_file_ready = os.path.isfile(FEATURE_SENTIMENT_FINBERT_FILE)

    for ticker in tickers:
        errors = {}
        source_components = []

        if ticker in raw_assets:
            asset = pd.DataFrame({"Date": base_dates, "Asset": ticker})
            asset = asset.merge(raw[raw["Asset"] == ticker], on=["Date", "Asset"], how="left")
            source_type = "raw_archive"
            source_components.append("raw_archive")
        else:
            asset = pd.DataFrame({"Date": base_dates, "Asset": ticker})
            news = None
            trends = None
            if use_api:
                news, news_error = _fetch_cryptopanic_news(
                    ticker,
                    request_timeout=request_timeout,
                )
                if news_error:
                    errors["cryptopanic"] = news_error
                if news is not None:
                    asset = asset.merge(news, on=["Date", "Asset"], how="left")
                    source_components.append("cryptopanic")

                if use_google_trends:
                    trends, trends_error = _fetch_google_trends_attention(
                        ticker,
                        request_timeout=request_timeout,
                    )
                    if trends_error:
                        errors["google_trends"] = trends_error
                    if trends is not None:
                        asset = asset.merge(trends, on=["Date", "Asset"], how="left")
                        source_components.append("google_trends")

            if source_components:
                source_type = "api"
            elif use_api and fear_greed is not None:
                # Use real market-wide sentiment control instead of synthetic fallback
                # when asset-specific APIs are unavailable.
                source_type = "api"
                source_components.append("fear_greed")
            else:
                source_type = "synthetic"
                asset = _generate_synthetic_sentiment(
                    [ticker], market_df=market_df, fear_greed=fear_greed
                )

        if fear_greed is not None and "Market_FearGreed" not in asset.columns:
            asset["Market_FearGreed"] = asset["Date"].map(fear_greed)
        for col in ["Social_Sentiment", "Social_Volume", "Sentiment_Disagreement"]:
            if col not in asset.columns:
                asset[col] = 0.0

        frames.append(asset)
        finbert_components = ["finbert_daily_csv"] if finbert_file_ready else []
        metadata["per_asset"][ticker] = {
            "source_type": source_type,
            "source_components": source_components + finbert_components,
            "is_real": source_type != "synthetic",
            "errors": errors,
            "google_trends_enabled": use_google_trends,
        }

    out = pd.concat(frames, ignore_index=True)
    fb_all = _load_finbert_daily_merge(tickers)
    if fb_all is not None:
        out = out.merge(fb_all, on=["Date", "Asset"], how="left")
    out = _finalize_sentiment_frame(out)
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


def save_sentiment_data(df: pd.DataFrame) -> str:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = normalize_dates(out["Date"])
    for path in [FEATURE_SENTIMENT_FILE, SENTIMENT_DATA_FILE]:
        out.to_csv(path, index=False)
    print(f"[Pillar 2] Saved -> {SENTIMENT_DATA_FILE}  ({len(out)} rows)")
    return SENTIMENT_DATA_FILE


if __name__ == "__main__":
    sentiment_df = load_sentiment_data()
    save_sentiment_data(sentiment_df)
