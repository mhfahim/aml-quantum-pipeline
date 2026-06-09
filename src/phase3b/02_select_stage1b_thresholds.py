import argparse
from pathlib import Path
import time
import joblib
import numpy as np
import pyarrow.dataset as ds
import polars as pl

from sklearn.metrics import roc_auc_score, average_precision_score

from phase3b_common import (
    ALL_FEATURES,
    read_split_cutoffs,
    add_temporal_split_df,
    add_phase3b_features_df,
    existing_scan_columns,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-pcts", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--score-sample-frac", type=float, default=0.005)
    parser.add_argument("--max-score-rows", type=int, default=700000)
    parser.add_argument("--batch-size", type=int, default=200000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)

    dataset = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    scan_cols = existing_scan_columns(dataset.schema.names)

    scores = []
    labels = []

    rows_seen = 0
    rows_scored = 0

    for batch in dataset.to_batches(columns=scan_cols, batch_size=args.batch_size):
        if batch.num_rows == 0:
            continue

        rows_seen += batch.num_rows

        df = pl.from_arrow(batch)
        df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
        df = df.filter(pl.col("_split") == "valid")

        if df.height == 0:
            continue

        df = df.with_columns(
            pl.struct(["_ts", "_label", "amount_received", "amount_paid", "_source_file"])
            .hash(seed=111)
            .alias("_score_hash")
        )

        df = df.filter(
            (pl.col("_score_hash") % 1_000_000) < int(args.score_sample_frac * 1_000_000)
        )

        if df.height == 0:
            continue

        df = add_phase3b_features_df(df)
        pdf = df.select(ALL_FEATURES + ["_label"]).to_pandas()

        batch_scores = model.predict_proba(pdf[ALL_FEATURES])[:, 1]

        scores.append(batch_scores)
        labels.append(pdf["_label"].astype(int).to_numpy())

        rows_scored += len(batch_scores)

        if rows_scored >= args.max_score_rows:
            break

    if not scores:
        raise ValueError("No validation scores collected. Increase --score-sample-frac.")

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    target_pcts = [float(x.strip()) for x in args.target_pcts.split(",")]

    thresholds = {}
    for pct in target_pcts:
        thresholds[str(pct)] = float(np.quantile(scores, 1.0 - pct))

    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else None
    pr_auc = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else None

    runtime = time.perf_counter() - start

    output = {
        "method": "validation predicted-probability quantile",
        "score_definition": "stage1b_score = predicted laundering probability; higher means more suspicious",
        "target_candidate_percentages": target_pcts,
        "thresholds": thresholds,
        "validation_score_rows_used": int(len(scores)),
        "validation_positive_rows_used": int((labels == 1).sum()),
        "validation_negative_rows_used": int((labels == 0).sum()),
        "validation_roc_auc": auc,
        "validation_pr_auc": pr_auc,
        "rows_seen_before_stop": int(rows_seen),
        "score_sample_frac": args.score_sample_frac,
        "runtime_seconds": runtime,
    }

    save_json(output, out_dir / "stage1b_thresholds.json")

    print("Stage 3B thresholds selected.")
    print(output)


if __name__ == "__main__":
    main()
