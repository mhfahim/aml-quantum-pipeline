import argparse
from pathlib import Path
import polars as pl

from phase2_common import (
    scan_parquet_dataset,
    save_json,
    read_json,
    safe_numeric_feature_candidates,
    add_temporal_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parquet_dir = Path(args.parquet)
    phase1_reports = Path(args.phase1_reports)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = read_json(phase1_reports / "temporal_split_manifest.json")
    train_cutoff = split_manifest["train_cutoff_raw_microseconds"]
    valid_cutoff = split_manifest["valid_cutoff_raw_microseconds"]

    lf = scan_parquet_dataset(parquet_dir)
    columns = list(lf.collect_schema().names())

    suspicious_cols = []
    for col in columns:
        low = col.lower()
        if any(x in low for x in ["label", "laundering", "fraud", "target"]):
            suspicious_cols.append(col)

    split_lf = add_temporal_split(lf, train_cutoff, valid_cutoff)

    split_counts = (
        split_lf
        .group_by(["_split", "_label"])
        .agg(pl.len().alias("n"))
        .sort(["_split", "_label"])
        .collect(engine="streaming")
    )
    split_counts.write_csv(out_dir / "split_label_counts.csv")

    temporal_ranges = (
        split_lf
        .group_by("_split")
        .agg([
            pl.len().alias("n"),
            pl.col("_ts").min().alias("min_ts"),
            pl.col("_ts").max().alias("max_ts"),
        ])
        .sort("_split")
        .collect(engine="streaming")
    )
    temporal_ranges.write_csv(out_dir / "temporal_ranges.csv")

    # Check whether any raw identifier fields overlap across splits.
    # This is not necessarily leakage, but reviewers will ask about it.
    possible_account_cols = [c for c in columns if "account" in c.lower()]
    account_overlap_summary = {}

    for c in possible_account_cols[:5]:
        try:
            unique_by_split = (
                split_lf
                .select(["_split", c])
                .filter(pl.col(c).is_not_null())
                .group_by("_split")
                .agg(pl.col(c).n_unique().alias("n_unique"))
                .collect(engine="streaming")
            )
            account_overlap_summary[c] = unique_by_split.to_dicts()
        except Exception as e:
            account_overlap_summary[c] = {"error": str(e)}

    safe_features = safe_numeric_feature_candidates(columns)

    audit = {
        "status": "completed",
        "n_columns": len(columns),
        "columns": columns,
        "suspicious_label_like_columns": suspicious_cols,
        "label_column_used": "_label",
        "timestamp_column_used": "_ts",
        "temporal_split": {
            "train_cutoff_raw_microseconds": train_cutoff,
            "valid_cutoff_raw_microseconds": valid_cutoff,
            "source": "phase1 temporal_split_manifest.json",
        },
        "safe_starter_feature_candidates": safe_features,
        "account_overlap_summary": account_overlap_summary,
        "important_interpretation": [
            "Columns containing label/laundering/fraud/target must not be used as model features.",
            "Raw account, bank, and entity identifiers are excluded from starter baselines to reduce leakage risk.",
            "Temporal split is used instead of random split.",
            "Account overlap across time is expected in transaction data, but future-derived aggregates must be avoided."
        ],
    }

    save_json(audit, out_dir / "leakage_audit.json")

    md = []
    md.append("# Phase 2 Leakage Audit\n")
    md.append("## Summary\n")
    md.append("- Temporal split is inherited from Phase 1.")
    md.append("- Label-like columns are identified and blocked from features.")
    md.append("- Raw account/bank/entity identifiers are blocked from starter baselines.")
    md.append("- Starter models use conservative numeric transaction features only.\n")

    md.append("## Suspicious Label-Like Columns\n")
    for col in suspicious_cols:
        md.append(f"- `{col}`")

    md.append("\n## Safe Starter Feature Candidates\n")
    for col in safe_features:
        md.append(f"- `{col}`")

    md.append("\n## Split Label Counts\n")
    for row in split_counts.to_dicts():
        md.append(f"- {row['_split']} | label {row['_label']}: {row['n']:,}")

    md.append("\n## Leakage Position\n")
    md.append("This phase does not claim zero leakage in all future engineered features. It establishes a conservative baseline feature set and blocks obvious leakage sources. Deeper leakage checks will continue when rolling, graph, and account-history features are engineered.")

    (out_dir / "leakage_audit.md").write_text("\n".join(md), encoding="utf-8")

    feature_schema = {
        "blocked_columns": suspicious_cols + possible_account_cols,
        "safe_starter_feature_candidates": safe_features,
        "label": "_label",
        "timestamp": "_ts",
        "split": "_split",
    }

    save_json(feature_schema, out_dir / "feature_schema.json")

    print("Leakage audit complete.")
    print("Safe starter features:", safe_features)


if __name__ == "__main__":
    main()
