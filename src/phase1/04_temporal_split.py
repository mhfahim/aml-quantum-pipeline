import argparse
from pathlib import Path
from datetime import datetime
import polars as pl

from common import save_json


def micros_to_datetime(value):
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Parquet dataset folder")
    parser.add_argument("--out", required=True, help="Report output folder")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    args = parser.parse_args()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_q = args.train_ratio
    valid_q = args.train_ratio + args.valid_ratio

    lf = pl.scan_parquet(str(parquet_dir / "**/*.parquet"))
    ts_int = pl.col("_ts").cast(pl.Int64)

    quantiles = (
        lf.filter(pl.col("_ts").is_not_null())
        .select([
            ts_int.quantile(train_q).alias("train_cutoff"),
            ts_int.quantile(valid_q).alias("valid_cutoff"),
        ])
        .collect(streaming=True)
    )

    train_cutoff = int(quantiles["train_cutoff"][0])
    valid_cutoff = int(quantiles["valid_cutoff"][0])

    split_lf = lf.with_columns(
        pl.when(ts_int <= train_cutoff)
        .then(pl.lit("train"))
        .when(ts_int <= valid_cutoff)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("test"))
        .alias("_split")
    )

    split_counts = (
        split_lf.group_by(["_split", "_label"])
        .agg(pl.len().alias("n"))
        .sort(["_split", "_label"])
        .collect(streaming=True)
    )

    split_counts.write_csv(out_dir / "temporal_split_counts.csv")

    split_ranges = (
        split_lf.group_by("_split")
        .agg([
            pl.len().alias("n"),
            pl.col("_ts").min().alias("min_timestamp"),
            pl.col("_ts").max().alias("max_timestamp"),
        ])
        .sort("_split")
        .collect(streaming=True)
    )

    split_ranges.write_csv(out_dir / "temporal_split_ranges.csv")

    manifest = {
        "split_type": "temporal",
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": round(1 - args.train_ratio - args.valid_ratio, 4),
        "train_cutoff_raw_microseconds": train_cutoff,
        "valid_cutoff_raw_microseconds": valid_cutoff,
        "train_cutoff_datetime": micros_to_datetime(train_cutoff),
        "valid_cutoff_datetime": micros_to_datetime(valid_cutoff),
        "split_counts": split_counts.to_dicts(),
        "split_ranges": split_ranges.to_dicts(),
        "note": "Data is not physically duplicated. Future scripts should apply these timestamp cutoffs to create train/valid/test views.",
    }

    save_json(manifest, out_dir / "temporal_split_manifest.json")

    print("Temporal split complete.")
    print(split_counts)
    print(split_ranges)


if __name__ == "__main__":
    main()
