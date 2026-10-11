"""
FinBERT daily polarity from a newline-delimited JSON corpus.

Corpus lines: {"asset": "BTC-USD", "text": "...", "published_at": "ISO", "source": "..."}
Scores are P(positive) - P(negative) in [-1, 1] aggregated per calendar day (mean).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    END_DATE,
    FEATURE_SENTIMENT_FINBERT_FILE,
    FINBERT_BATCH_SIZE,
    FINBERT_MODEL_NAME,
    RAW_SENTIMENT_CORPUS_DIR,
    START_DATE,
    TICKERS,
)
from src.asset_utils import normalize_ticker
from src.collection_utils import normalize_dates


def _read_corpus_paths() -> list[Path]:
    if not os.path.isdir(RAW_SENTIMENT_CORPUS_DIR):
        return []
    paths = sorted(Path(RAW_SENTIMENT_CORPUS_DIR).glob("*.jsonl"))
    return [p for p in paths if p.is_file()]


def load_corpus_records() -> list[dict]:
    rows = []
    for path in _read_corpus_paths():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _finbert_probs_batch(texts: list[str], model, tokenizer, device) -> np.ndarray:
    import torch

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def _label_order_positive_negative_neutral(model) -> tuple[int, int, int]:
    """Map FinBERT id2label to (positive_idx, negative_idx, neutral_idx)."""
    id2label = getattr(model.config, "id2label", None) or {}
    pos = neg = neu = None
    for k, v in id2label.items():
        lab = str(v).lower()
        idx = int(k)
        if "pos" in lab:
            pos = idx
        elif "neg" in lab:
            neg = idx
        elif "neu" in lab:
            neu = idx
    return (
        pos if pos is not None else 0,
        neg if neg is not None else 1,
        neu if neu is not None else 2,
    )


def score_texts_finbert(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Return shape (n,) scores in [-1, 1]: P(pos) - P(neg).
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise ImportError("Install torch and transformers for FinBERT.") from e

    model_name = model_name or FINBERT_MODEL_NAME
    batch_size = batch_size or FINBERT_BATCH_SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    pos_i, neg_i, neu_i = _label_order_positive_negative_neutral(model)

    scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if not batch:
            continue
        probs = _finbert_probs_batch(batch, model, tokenizer, device)
        # (batch, 3) — order may vary; select columns by label map
        for row in probs:
            p_pos = float(row[pos_i])
            p_neg = float(row[neg_i])
            scores.append(np.clip(p_pos - p_neg, -1.0, 1.0))
    return np.array(scores, dtype=float)


def build_finbert_daily_panel(
    tickers: list[str] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame | None:
    """
    Read JSONL corpus, run FinBERT, write daily FinBERT_Polarity per asset.
    """
    tickers = set(tickers or TICKERS)
    records = load_corpus_records()
    if not records:
        print("[FinBERT] No JSONL corpus under", RAW_SENTIMENT_CORPUS_DIR)
        return None

    parsed = []
    for r in records:
        text = (r.get("text") or r.get("title") or "").strip()
        if not text:
            continue
        asset = normalize_ticker(r.get("asset") or r.get("ticker") or r.get("Asset"))
        if asset is None or asset not in tickers:
            continue
        ts = pd.to_datetime(r.get("published_at") or r.get("timestamp") or r.get("Date"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        ts = ts.tz_localize(None).normalize()
        d = normalize_dates(ts)
        if str(d) < START_DATE or str(d) > END_DATE:
            continue
        parsed.append({"Date": d, "Asset": asset, "text": text})

    if not parsed:
        print("[FinBERT] Corpus had no rows matching TICKERS / date range.")
        return None

    df = pd.DataFrame(parsed)
    texts = df["text"].tolist()
    print(f"[FinBERT] Scoring {len(texts)} texts with {FINBERT_MODEL_NAME} …")
    scores = score_texts_finbert(texts)
    df["score"] = scores

    daily = (
        df.groupby(["Date", "Asset"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "FinBERT_Polarity"})
    )
    daily["FinBERT_Polarity"] = daily["FinBERT_Polarity"].clip(-1.0, 1.0)

    out_path = output_path or FEATURE_SENTIMENT_FINBERT_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"[FinBERT] Daily panel saved -> {out_path} ({len(daily)} rows)")
    return daily


if __name__ == "__main__":
    build_finbert_daily_panel()
