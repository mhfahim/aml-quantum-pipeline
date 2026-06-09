from pathlib import Path
import json
import re
import os
import polars as pl


TIMESTAMP_CANDIDATES = [
    "timestamp", "time", "datetime", "date", "transaction_time"
]

LABEL_CANDIDATES = [
    "is_laundering", "laundering", "is_fraud", "fraud", "label", "target"
]

TRANSACTION_TYPE_CANDIDATES = [
    "payment_format", "transaction_type", "type", "channel", "payment_type"
]


def clean_col(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    mapping = {col: clean_col(col) for col in df.columns}
    return df.rename(mapping)


def find_col(cols, candidates):
    cols_clean = list(cols)
    for cand in candidates:
        cand = clean_col(cand)
        if cand in cols_clean:
            return cand
    return None


def safe_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name[:150]


def walk_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0

    total = 0
    for root, _, files in os.walk(path):
        for file in files:
            total += (Path(root) / file).stat().st_size
    return total


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
