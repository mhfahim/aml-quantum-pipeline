import argparse
from pathlib import Path
import time
import joblib
import json
import pyarrow.dataset as ds
import polars as pl
import numpy as np

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


def init_stats(threshold_keys):
    stats = {}

    for th in threshold_keys:
        stats[th] = {}

        for split in ["train", "valid", "test"]:
            stats[th][split] = {
                "total_rows": 0,
                "total_legit": 0,
                "total_fraud": 0,
                "candidate_rows": 0,
                "candidate_legit": 0,
                "candidate_fraud": 0,
            }

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=250000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)
    model = joblib.load(args.model)
    threshold_data = read_json(args.thresholds)
    thresholds = threshold_data["thresholds"]

    threshold_keys = list(thresholds.keys())
    stats = init_stats(threshold_keys)

    dataset = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    available_cols = dataset.schema.names
    scan_cols = [c for c in BASE_SCAN_COLUMNS if c in available_cols]

    total_batches = 0
    total_rows_seen = 0

    for batch in dataset.to_batches(columns=scan_cols, batch_size=args.batch_size):
        if batch.num_rows == 0:
            continue

        total_batches += 1
        total_rows_seen += batch.num_rows

        df = pl.from_arrow(batch)
        df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
        df = add_phase3_features_df(df)

        X = df.select(PHASE3_FEATURES).to_numpy()
        scores = -model.decision_function(X)

        split_arr = df["_split"].to_numpy()
        label_arr = df["_label"].to_numpy()

        for th_key, th_value in thresholds.items():
            cand_arr = scores >= float(th_value)

            for split in ["train", "valid", "test"]:
                split_mask = split_arr == split

                if not split_mask.any():
                    continue

                labels_split = label_arr[split_mask]
                cand_split = cand_arr[split_mask]

                total_rows = len(labels_split)
                total_fraud = int((labels_split == 1).sum())
                total_legit = int((labels_split == 0).sum())

                candidate_rows = int(cand_split.sum())
                candidate_fraud = int(((labels_split == 1) & cand_split).sum())
                candidate_legit = int(((labels_split == 0) & cand_split).sum())

                s = stats[th_key][split]
                s["total_rows"] += total_rows
                s["total_legit"] += total_legit
                s["total_fraud"] += total_fraud
                s["candidate_rows"] += candidate_rows
                s["candidate_legit"] += candidate_legit
                s["candidate_fraud"] += candidate_fraud

        if total_batches % 25 == 0:
            print(f"Processed batches: {total_batches}, rows: {total_rows_seen:,}")

    rows = []

    for th_key in threshold_keys:
        for split in ["train", "valid", "test"]:
            s = stats[th_key][split]

            total_rows = s["total_rows"]
            total_fraud = s["total_fraud"]
            total_legit = s["total_legit"]
            candidate_rows = s["candidate_rows"]
            candidate_fraud = s["candidate_fraud"]
            candidate_legit = s["candidate_legit"]

            base_prevalence = total_fraud / total_rows if total_rows else 0
            candidate_prevalence = candidate_fraud / candidate_rows if candidate_rows else 0

            rows.append({
                "target_candidate_pct": float(th_key),
                "threshold": thresholds[th_key],
                "split": split,
                "total_rows": total_rows,
                "total_legit": total_legit,
                "total_fraud": total_fraud,
                "candidate_rows": candidate_rows,
                "candidate_legit": candidate_legit,
                "candidate_fraud": candidate_fraud,
                "actual_candidate_pct": candidate_rows / total_rows if total_rows else 0,
                "fraud_retention": candidate_fraud / total_fraud if total_fraud else 0,
                "legit_pass_rate": candidate_legit / total_legit if total_legit else 0,
                "base_fraud_prevalence": base_prevalence,
                "candidate_fraud_prevalence": candidate_prevalence,
                "enrichment_factor": candidate_prevalence / base_prevalence if base_prevalence > 0 else None,
            })

    metrics_df = pl.DataFrame(rows)
    metrics_df.write_csv(out_dir / "stage1_screening_metrics.csv")

    runtime = time.perf_counter() - start

    save_json(
        {
            "stats": stats,
            "thresholds": thresholds,
            "rows_seen": total_rows_seen,
            "batches_seen": total_batches,
            "runtime_seconds": runtime,
        },
        out_dir / "stage1_screening_metrics.json",
    )

    md = []
    md.append("# Phase 3 Stage 1 Screening Metrics\n")
    md.append("Stage 1 model: Isolation Forest trained on legitimate training transactions.\n")

    for row in rows:
        md.append(
            f"- target={row['target_candidate_pct']} | split={row['split']} | "
            f"actual_candidate_pct={row['actual_candidate_pct']:.6f} | "
            f"fraud_retention={row['fraud_retention']:.6f} | "
            f"candidate_fraud_prevalence={row['candidate_fraud_prevalence']:.6f} | "
            f"enrichment={row['enrichment_factor']}"
        )

    md.append(f"\nRuntime seconds: {runtime:.2f}")

    (out_dir / "stage1_screening_metrics.md").write_text("\n".join(md), encoding="utf-8")

    print(metrics_df)
    print("Stage 1 full evaluation complete.")


if __name__ == "__main__":
    main()
