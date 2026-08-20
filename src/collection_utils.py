"""
Shared helpers for the data-collection pipeline.
"""
from __future__ import annotations

import os

import pandas as pd


def load_local_env(project_root: str) -> None:
    """Load simple KEY=VALUE pairs from the project .env."""
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def normalize_dates(series: pd.Series) -> pd.Series:
    """Convert mixed timestamp inputs into UTC-naive daily timestamps."""
    return (
        pd.to_datetime(series, errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
