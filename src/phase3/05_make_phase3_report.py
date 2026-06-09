import argparse
from pathlib import Path
import json


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    report_dir = Path(args.reports)
    data_dir = Path(args.data)

    model_meta = read_json(report_dir / "stage1_model_metadata.json")
    thresholds = read_json(report_dir / "stage1_thresholds.json")
    metrics = read_json(report_dir / "stage1_screening_metrics.json")
    candidate_manifest = read_json(data_dir / "candidate_pool_sample_manifest.json")

    md = []

    md.append("# Phase 3 Report: Stage 1 Classical Anomaly Filtering\n")

    md.append("## 1. Purpose\n")
    md.append(
        "Phase 3 implements the first stage of the two-stage AML pipeline. "
        "A classical anomaly filter screens the full transaction dataset and produces a high-suspicion candidate pool for later quantum and hybrid analysis.\n"
    )

    md.append("## 2. Stage 1 Model\n")
    md.append(f"- Model: **{model_meta['model']}**")
    md.append(f"- Training data: **{model_meta['training_data']}**")
    md.append(f"- Training sample rows: **{model_meta['training_sample_rows']:,}**")
    md.append(f"- Features: `{model_meta['features']}`")
    md.append(f"- Runtime seconds: **{model_meta['runtime_seconds']:.2f}**\n")

    md.append("## 3. Thresholds\n")
    md.append("- Score definition: higher score = more anomalous.")
    for pct, th in thresholds["thresholds"].items():
        md.append(f"- Target candidate percentage `{pct}`: threshold `{th}`")
    md.append("")

    md.append("## 4. Full-Dataset Screening Results\n")
    md.append("See `stage1_screening_metrics.csv` for full split-level results.\n")

    for target_pct, split_stats in metrics["stats"].items():
        md.append(f"### Target candidate percentage: {target_pct}")
        for split, s in split_stats.items():
            total_fraud = s["total_fraud"]
            candidate_fraud = s["candidate_fraud"]
            total_rows = s["total_rows"]
            candidate_rows = s["candidate_rows"]

            fraud_retention = candidate_fraud / total_fraud if total_fraud else 0
            candidate_pct = candidate_rows / total_rows if total_rows else 0

            md.append(
                f"- {split}: candidate rows `{candidate_rows:,}` / `{total_rows:,}` "
                f"({candidate_pct:.6f}), fraud retained `{candidate_fraud:,}` / `{total_fraud:,}` "
                f"({fraud_retention:.6f})"
            )
        md.append("")

    md.append("## 5. Candidate Pool Sample\n")
    md.append(f"- Target threshold used: **{candidate_manifest['target_candidate_pct']}**")
    md.append(f"- Candidate parts written: **{candidate_manifest['parts_written']}**")
    md.append(f"- Sample rows kept: **{candidate_manifest['kept_rows']:,}**")
    md.append(f"- Positive policy: **{candidate_manifest['positive_policy']}**")
    md.append(f"- Negative sample fraction: **{candidate_manifest['negative_sample_fraction_inside_flagged_pool']}**\n")

    md.append("## 6. Interpretation\n")
    md.append(
        "This phase does not claim final classification superiority. It validates the upstream screening layer: "
        "how much of the transaction universe can be reduced while retaining laundering cases. "
        "The resulting candidate pool will be used in Phase 4 for fair reduced-feature classical, quantum, and hybrid comparisons."
    )

    out_path = report_dir / "PHASE3_STAGE1_REPORT.md"
    out_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Phase 3 report written to {out_path}")


if __name__ == "__main__":
    main()
