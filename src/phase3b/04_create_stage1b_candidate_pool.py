import argparse
from pathlib import Path
import time
import joblib
import json
import shutil
import pyarrow.dataset as ds
import polars as pl
import pandas as pd

from phase3b_common import (
    ALL_FEATURES,
    read_split_cutoffs,
    add_temporal_split_df,
    add_phase3b_features_df,
    existing_scan_columns,
    save_json,
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_auto_target(metrics_csv, min_recall):
    df = pd.read_csv(metrics_csv)
    valid = df[df["split"] == "valid"].copy()
    valid = valid.sort_values("target_candidate_pct")

    passing = valid[valid["fraud_retention"] >= min_recall]

    if len(passing) > 0:
        return str(float(passing.iloc[0]["target_candidate_pct"]))

    best = valid.sort_values("fraud_retention", ascending=False).iloc[0]
    return str(float(best["target_candidate_pct"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--screening-metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-pct", default="auto")
    parser.add_argument("--min-valid-recall", type=float, default=0.80)
    parser.add_argument("--neg-sample-frac", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=200000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_dir = out_dir / "stage1b_candidate_pool_sample"

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    candidate_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    thresholds = read_json(args.thresholds)["thresholds"]

    if args.target_pct == "auto":
        selected_target = choose_auto_target(args.screening_metrics, args.min_valid_recall)
    else:
        selected_target = str(float(args.target_pct))

    if selected_target not in thresholds:
        available = list(thresholds.keys())
        raise ValueError(f"Selected target {selected_target} not found in thresholds. Available: {available}")

    threshold = float(thresholds[selected_target])

    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)

    dataset = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    scan_cols = existing_scan_columns(dataset.schema.names)

    part_id = 0
    kept_rows = 0
    seen_rows = 0

    for batch in dataset.to_batches(columns=scan_cols, batch_size=args.batch_size):
        if batch.num_rows == 0:
            continue

        seen_rows += batch.num_rows

        df = pl.from_arrow(batch)
        df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
        df = add_phase3b_features_df(df)

        pdf = df.select(ALL_FEATURES + ["_split", "_label", "_ts", "_source_file"]).to_pandas()
        scores = model.predict_proba(pdf[ALL_FEATURES])[:, 1]

        df = df.with_columns(pl.Series("stage1b_score", scores))

        df = df.with_columns(
            pl.struct(["_ts", "_label", "_source_file", "stage1b_score"])
            .hash(seed=2026)
            .alias("_candidate_hash")
        )

        candidate_df = (
            df
            .filter(pl.col("stage1b_score") >= threshold)
            .filter(
                (pl.col("_label") == 1) |
                ((pl.col("_label") == 0) & ((pl.col("_candidate_hash") % 1_000_000) < int(args.neg_sample_frac * 1_000_000)))
            )
            .select([
                "_split",
                "_label",
                "_ts",
                "_source_file",
                "stage1b_score",
            ] + ALL_FEATURES)
        )

        if candidate_df.height > 0:
            out_file = candidate_dir / f"candidate_part_{part_id:06d}.parquet"
            candidate_df.write_parquet(out_file, compression="zstd")
            kept_rows += candidate_df.height
            part_id += 1

        if part_id % 50 == 0 and part_id > 0:
            print(f"Parts written: {part_id}, kept rows: {kept_rows:,}, seen rows: {seen_rows:,}")

    candidate_scan = pl.scan_parquet(str(candidate_dir / "*.parquet"))

    counts = (
        candidate_scan
        .group_by(["_split", "_label"])
        .agg(pl.len().alias("n"))
        .sort(["_split", "_label"])
        .collect(engine="streaming")
    )

    counts.write_csv(out_dir / "stage1b_candidate_pool_sample_counts.csv")

    runtime = time.perf_counter() - start

    manifest = {
        "candidate_pool_type": "Phase 3B high-recall supervised screening candidate pool",
        "selected_target_candidate_pct": selected_target,
        "threshold": threshold,
        "selection_policy": "auto selects smallest validation candidate percentage with fraud retention >= min_valid_recall; otherwise best available recall",
        "min_valid_recall": args.min_valid_recall,
        "negative_sample_fraction_inside_flagged_pool": args.neg_sample_frac,
        "positive_policy": "all flagged positives retained",
        "features": ALL_FEATURES,
        "candidate_dir": str(candidate_dir),
        "parts_written": part_id,
        "kept_rows": kept_rows,
        "seen_rows": seen_rows,
        "counts": counts.to_dicts(),
        "runtime_seconds": runtime,
    }

    save_json(manifest, out_dir / "stage1b_candidate_pool_sample_manifest.json")

    print("Stage 3B candidate pool sample created.")
    print(counts)
    print(manifest)


if __name__ == "__main__":
    main()
