import argparse
from pathlib import Path
import time
import joblib
import numpy as np
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-pcts", default="0.05,0.10,0.15")
    parser.add_argument("--score-sample-frac", type=float, default=0.02)
    parser.add_argument("--max-score-rows", type=int, default=1500000)
    parser.add_argument("--batch-size", type=int, default=250000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)
    model = joblib.load(args.model)

    dataset = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    available_cols = dataset.schema.names
    scan_cols = [c for c in BASE_SCAN_COLUMNS if c in available_cols]

    scores = []
    rows_seen = 0
    rows_kept = 0

    for batch in dataset.to_batches(columns=scan_cols, batch_size=args.batch_size):
        if batch.num_rows == 0:
            continue

        df = pl.from_arrow(batch)
        df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
        df = df.filter(pl.col("_split") == "valid")

        if df.height == 0:
            continue

        df = df.with_columns(
            pl.struct(["_ts", "amount_received", "amount_paid", "_source_file"])
            .hash(seed=777)
            .alias("_hash")
        )

        df = df.filter((pl.col("_hash") % 1_000_000) < int(args.score_sample_frac * 1_000_000))

        if df.height == 0:
            continue

        df = add_phase3_features_df(df)
        X = df.select(PHASE3_FEATURES).to_numpy()

        batch_scores = -model.decision_function(X)

        scores.append(batch_scores)
        rows_seen += batch.num_rows
        rows_kept += len(batch_scores)

        if rows_kept >= args.max_score_rows:
            break

    if not scores:
        raise ValueError("No validation scores collected. Increase --score-sample-frac.")

    scores = np.concatenate(scores)

    target_pcts = [float(x.strip()) for x in args.target_pcts.split(",")]

    thresholds = {}
    for pct in target_pcts:
        threshold = float(np.quantile(scores, 1.0 - pct))
        thresholds[str(pct)] = threshold

    runtime = time.perf_counter() - start

    output = {
        "method": "validation score quantile",
        "score_definition": "stage1_anomaly_score = -IsolationForest.decision_function(X); higher means more anomalous",
        "target_candidate_percentages": target_pcts,
        "thresholds": thresholds,
        "validation_score_rows_used": int(len(scores)),
        "rows_seen_before_stop": int(rows_seen),
        "score_sample_frac": args.score_sample_frac,
        "runtime_seconds": runtime,
    }

    save_json(output, out_dir / "stage1_thresholds.json")

    print("Stage 1 thresholds selected.")
    print(output)


if __name__ == "__main__":
    main()
