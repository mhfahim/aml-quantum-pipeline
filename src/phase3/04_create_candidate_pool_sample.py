import argparse
from pathlib import Path
import time
import joblib
import json
import pyarrow.dataset as ds
import polars as pl

from phase3_common import (
    PHASE3_FEATURES,
    BASE_SCAN_COLUMNS,
    read_split_cutoffs,
    add_temporal_split_df,
    add_phase3_features_df,
    save_json,
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-pct", default="0.10")
    parser.add_argument("--neg-sample-frac", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=250000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_dir = out_dir / "candidate_pool_sample"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)
    model = joblib.load(args.model)
    threshold_data = read_json(args.thresholds)
    thresholds = threshold_data["thresholds"]

    if args.target_pct not in thresholds:
        raise ValueError(f"target pct {args.target_pct} not found in thresholds: {thresholds}")

    threshold = float(thresholds[args.target_pct])

    dataset = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    available_cols = dataset.schema.names
    scan_cols = [c for c in BASE_SCAN_COLUMNS if c in available_cols]

    part_id = 0
    kept_rows = 0
    seen_rows = 0

    for batch in dataset.to_batches(columns=scan_cols, batch_size=args.batch_size):
        if batch.num_rows == 0:
            continue

        seen_rows += batch.num_rows

        df = pl.from_arrow(batch)
        df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
        df = add_phase3_features_df(df)

        X = df.select(PHASE3_FEATURES).to_numpy()
        scores = -model.decision_function(X)

        df = df.with_columns(
            pl.Series("stage1_anomaly_score", scores)
        )

        df = df.with_columns(
            pl.struct(["_ts", "_label", "_source_file", "stage1_anomaly_score"])
            .hash(seed=999)
            .alias("_candidate_hash")
        )

        candidate_df = (
            df
            .filter(pl.col("stage1_anomaly_score") >= threshold)
            .filter(
                (pl.col("_label") == 1) |
                ((pl.col("_label") == 0) & ((pl.col("_candidate_hash") % 1_000_000) < int(args.neg_sample_frac * 1_000_000)))
            )
            .select([
                "_split",
                "_label",
                "_ts",
                "_source_file",
                "stage1_anomaly_score",
                "payment_format",
                "receiving_currency",
                "payment_currency",
            ] + PHASE3_FEATURES)
        )

        if candidate_df.height > 0:
            out_file = candidate_dir / f"candidate_part_{part_id:06d}.parquet"
            candidate_df.write_parquet(out_file, compression="zstd")
            kept_rows += candidate_df.height
            part_id += 1

        if part_id % 25 == 0 and part_id > 0:
            print(f"Parts written: {part_id}, kept rows: {kept_rows:,}, seen rows: {seen_rows:,}")

    candidate_scan = pl.scan_parquet(str(candidate_dir / "*.parquet"))

    counts = (
        candidate_scan
        .group_by(["_split", "_label"])
        .agg(pl.len().alias("n"))
        .sort(["_split", "_label"])
        .collect(engine="streaming")
    )

    counts.write_csv(out_dir / "candidate_pool_sample_counts.csv")

    runtime = time.perf_counter() - start

    manifest = {
        "candidate_pool_type": "sampled candidate pool from Stage 1 Isolation Forest screening",
        "target_candidate_pct": args.target_pct,
        "threshold": threshold,
        "negative_sample_fraction_inside_flagged_pool": args.neg_sample_frac,
        "positive_policy": "all flagged positives retained",
        "features": PHASE3_FEATURES,
        "candidate_dir": str(candidate_dir),
        "parts_written": part_id,
        "kept_rows": kept_rows,
        "seen_rows": seen_rows,
        "counts": counts.to_dicts(),
        "runtime_seconds": runtime,
    }

    save_json(manifest, out_dir / "candidate_pool_sample_manifest.json")

    print("Candidate pool sample created.")
    print(counts)
    print(manifest)


if __name__ == "__main__":
    main()
