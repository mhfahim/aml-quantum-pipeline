import argparse
import json
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True, help="Report folder")
    args = parser.parse_args()

    report_dir = Path(args.reports)

    inventory = read_json(report_dir / "dataset_inventory.json")
    profile = read_json(report_dir / "phase1_profile.json")
    split = read_json(report_dir / "temporal_split_manifest.json")

    md = []

    md.append("# Phase 1 Data Processing Report\n")

    md.append("## 1. Dataset Inventory\n")
    md.append(f"- CSV files found: **{inventory['n_csv_files']}**")
    md.append(f"- Total raw CSV size: **{inventory['total_csv_size_gb']} GB**\n")

    md.append("## 2. Full Dataset Size\n")
    md.append(f"- Total transactions processed: **{profile['total_rows']:,}**")
    md.append(f"- Number of columns: **{profile['n_columns']}**")
    md.append(f"- Raw size: **{profile['raw_size_gb']} GB**")
    md.append(f"- Parquet size: **{profile['parquet_size_gb']} GB**")
    md.append(f"- Available system RAM during profiling: **{profile['available_ram_gb']} GB**\n")

    md.append("## 3. Laundering Label Distribution\n")
    for row in profile["label_counts"]:
        md.append(f"- Label `{row['_label']}`: **{row['n']:,}** rows ({row['pct']:.6f}%)")
    md.append("")

    md.append("## 4. Time Range\n")
    md.append(f"- Minimum timestamp: **{profile['time_range']['min_timestamp']}**")
    md.append(f"- Maximum timestamp: **{profile['time_range']['max_timestamp']}**\n")

    md.append("## 5. Transaction Type Column\n")
    md.append(f"- Detected transaction type column: **{profile['transaction_type_column']}**\n")

    md.append("## 6. Sender / Receiver / Account Fields\n")
    for key, value in profile["sender_receiver_fields"].items():
        md.append(f"- {key}: `{value}`")
    md.append("")

    md.append("## 7. Duplicate Check\n")
    md.append("```json")
    md.append(json.dumps(profile["duplicate_stats"], indent=2))
    md.append("```\n")

    md.append("## 8. Temporal Split\n")
    md.append(f"- Split type: **{split['split_type']}**")
    md.append(f"- Train cutoff: **{split['train_cutoff_datetime']}**")
    md.append(f"- Validation cutoff: **{split['valid_cutoff_datetime']}**\n")

    md.append("### Split Counts\n")
    for row in split["split_counts"]:
        md.append(f"- {row['_split']} | label {row['_label']}: **{row['n']:,}** rows")

    md.append("\n## 9. Phase 1 Status\n")
    md.append("Phase 1 is complete if these files exist:")
    md.append("- `dataset_inventory.csv`")
    md.append("- `dataset_inventory.json`")
    md.append("- `conversion_log.json`")
    md.append("- `label_counts.csv`")
    md.append("- `missing_values.csv`")
    md.append("- `transaction_type_counts.csv`")
    md.append("- `temporal_split_counts.csv`")
    md.append("- `temporal_split_ranges.csv`")
    md.append("- `temporal_split_manifest.json`")
    md.append("- `PHASE1_DATA_AUDIT.md`")

    output_path = report_dir / "PHASE1_DATA_AUDIT.md"
    output_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
