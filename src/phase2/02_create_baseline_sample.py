import argparse
from pathlib import Path
import polars as pl

from phase2_common import (
    scan_parquet_dataset,
    read_json,
    save_json,
    add_temporal_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--phase2-reports", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--neg-frac", type=float, default=0.01)
    args = parser.parse_args()

    parquet_dir = Path(args.parquet)
    phase1_reports = Path(args.phase1_reports)
    phase2_reports = Path(args.phase2_reports)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = read_json(phase1_reports / "temporal_split_manifest.json")
    feature_schema = read_json(phase2_reports / "feature_schema.json")

    train_cutoff = split_manifest["train_cutoff_raw_microseconds"]
    valid_cutoff = split_manifest["valid_cutoff_raw_microseconds"]

    safe_features = feature_schema["safe_starter_feature_candidates"]

    if not safe_features:
        raise ValueError("No safe starter features found. Check leakage audit output.")

    selected_cols = ["_ts", "_label"] + safe_features

    lf = scan_parquet_dataset(parquet_dir)
    lf = add_temporal_split(lf, train_cutoff, valid_cutoff)
    lf = lf.select(["_split"] + selected_cols)

    # Keep all positives. Downsample negatives.
    sample_lf = (
        lf
        .filter(
            (pl.col("_label") == 1) |
            ((pl.col("_label") == 0) & (pl.int_range(pl.len()).shuffle(seed=42) < (pl.len() * args.neg_frac)))
        )
    )

    # Polars random sample inside lazy group can be tricky at huge scale.
    # Safer approach: deterministic hash-based negative sampling.
    sample_lf = (
        lf
        .with_columns(
            pl.struct(["_ts"] + safe_features).hash(seed=42).alias("_row_hash")
        )
        .filter(
            (pl.col("_label") == 1) |
            ((pl.col("_label") == 0) & ((pl.col("_row_hash") % 10000) < int(args.neg_frac * 10000)))
        )
        .drop("_row_hash")
    )

    out_file = out_dir / "baseline_sample.parquet"

    sample_lf.sink_parquet(
        out_file,
        compression="zstd"
    )

    sample_scan = pl.scan_parquet(out_file)

    counts = (
        sample_scan
        .group_by(["_split", "_label"])
        .agg(pl.len().alias("n"))
        .sort(["_split", "_label"])
        .collect(engine="streaming")
    )

    counts.write_csv(out_dir / "baseline_sample_counts.csv")

    manifest = {
        "sample_type": "all positives plus deterministic hash-sampled negatives",
        "negative_fraction": args.neg_frac,
        "safe_features": safe_features,
        "output_file": str(out_file),
        "counts": counts.to_dicts(),
    }

    save_json(manifest, out_dir / "baseline_sample_manifest.json")

    print("Baseline sample created.")
    print(counts)


if __name__ == "__main__":
    main()
