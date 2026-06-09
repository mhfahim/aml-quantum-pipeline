import argparse
from pathlib import Path
import json
import csv


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    report_dir = Path(args.reports)
    data_dir = Path(args.data)

    model_meta = read_json(report_dir / "stage1b_model_metadata.json")
    thresholds = read_json(report_dir / "stage1b_thresholds.json")
    screening_rows = read_csv_rows(report_dir / "stage1b_screening_metrics.csv")
    candidate_manifest = read_json(data_dir / "stage1b_candidate_pool_sample_manifest.json")

    md = []

    md.append("# Phase 3B Report: Improved High-Recall Stage 1 Screening\n")

    md.append("## 1. Purpose\n")
    md.append(
        "Phase 3B improves the initial Stage 1 Isolation Forest baseline by training a high-recall classical screening model. "
        "The objective is not final classification, but candidate generation: retaining as many laundering cases as possible while reducing the transaction universe before Phase 4 quantum and reduced-feature comparisons.\n"
    )

    md.append("## 2. Model\n")
    md.append(f"- Model name: **{model_meta['model_name']}**")
    md.append(f"- Model type: **{model_meta['model_type']}**")
    md.append(f"- Training rows: **{model_meta['total_training_rows']:,}**")
    md.append(f"- Positive rows: **{model_meta['actual_pos_rows']:,}**")
    md.append(f"- Negative rows: **{model_meta['actual_neg_rows']:,}**")
    md.append(f"- Runtime seconds: **{model_meta['runtime_seconds']:.2f}**")
    md.append(f"- Features: `{model_meta['features']}`\n")

    md.append("## 3. Validation Thresholds\n")
    md.append(f"- Validation ROC-AUC: **{thresholds['validation_roc_auc']}**")
    md.append(f"- Validation PR-AUC: **{thresholds['validation_pr_auc']}**")
    for pct, th in thresholds["thresholds"].items():
        md.append(f"- Target candidate percentage `{pct}`: threshold `{th}`")
    md.append("")

    md.append("## 4. Full-Dataset Screening Results\n")
    for row in screening_rows:
        md.append(
            f"- target={row['target_candidate_pct']} | split={row['split']} | "
            f"actual_candidate_pct={float(row['actual_candidate_pct']):.6f} | "
            f"fraud_retention={float(row['fraud_retention']):.6f} | "
            f"candidate_fraud_prevalence={float(row['candidate_fraud_prevalence']):.6f} | "
            f"enrichment={row['enrichment_factor']}"
        )
    md.append("")

    md.append("## 5. Candidate Pool Sample\n")
    md.append(f"- Selected target candidate percentage: **{candidate_manifest['selected_target_candidate_pct']}**")
    md.append(f"- Candidate parts written: **{candidate_manifest['parts_written']}**")
    md.append(f"- Sample rows kept: **{candidate_manifest['kept_rows']:,}**")
    md.append(f"- Seen rows: **{candidate_manifest['seen_rows']:,}**")
    md.append(f"- Positive policy: **{candidate_manifest['positive_policy']}**")
    md.append(f"- Negative sample fraction: **{candidate_manifest['negative_sample_fraction_inside_flagged_pool']}**\n")

    md.append("## 6. Candidate Pool Counts\n")
    for row in candidate_manifest["counts"]:
        md.append(f"- {row['_split']} | label {row['_label']}: **{row['n']:,}**")
    md.append("")

    md.append("## 7. Interpretation\n")
    md.append(
        "Phase 3B should be used as the stronger Stage 1 candidate generator if it achieves substantially higher fraud retention than the Phase 3 Isolation Forest baseline. "
        "The candidate pool created here is the recommended input for Phase 4 reduced-feature classical, quantum, and hybrid comparisons. "
        "Phase 3 remains useful as an unsupervised baseline, while Phase 3B provides the operationally stronger high-recall screening track."
    )

    out_path = report_dir / "PHASE3B_HIGH_RECALL_REPORT.md"
    out_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Phase 3B report written to {out_path}")


if __name__ == "__main__":
    main()
