from pathlib import Path
import json
import polars as pl


DROP_ALWAYS = {
    "_source_file",
}

LABEL_COL = "_label"
TIME_COL = "_ts"


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_parquet_dataset(parquet_dir):
    return pl.scan_parquet(str(Path(parquet_dir) / "**/*.parquet"))


def safe_numeric_feature_candidates(columns):
    """
    Very strict starter features.
    We intentionally exclude IDs, labels, timestamps, and leakage-prone account/entity identifiers.
    """
    block_keywords = [
        "label",
        "laundering",
        "fraud",
        "timestamp",
        "_ts",
        "account",
        "bank",
        "entity",
        "name",
        "id",
        "source",
    ]

    candidates = []

    for col in columns:
        low = col.lower()

        if col in DROP_ALWAYS:
            continue

        if any(k in low for k in block_keywords):
            continue

        candidates.append(col)

    return candidates


def add_temporal_split(lf, train_cutoff, valid_cutoff):
    ts_int = pl.col("_ts").cast(pl.Int64)

    return lf.with_columns(
        pl.when(ts_int <= train_cutoff)
        .then(pl.lit("train"))
        .when(ts_int <= valid_cutoff)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("test"))
        .alias("_split")
    )
