import argparse
from pathlib import Path
import pandas as pd
import json


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_json_if_exists(path):
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def best_row(df, metric, higher=True):
    if df.empty or metric not in df.columns:
        return None

    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])

    if d.empty:
        return None

    return d.sort_values(metric, ascending=not higher).iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    args = parser.parse_args()

    reports = Path(args.reports)

    label_summary = read_csv_if_exists(reports / "phase5_label_scarcity_summary.csv")
    fixed_full = read_csv_if_exists(reports / "phase5_fixed_recall_estimated_full_hybrid.csv")
    novel = read_csv_if_exists(reports / "phase5_novel_pattern_results.csv")
    cost = read_csv_if_exists(reports / "phase5_cost_reduction_summary.csv")
    scalability = read_csv_if_exists(reports / "phase5_scalability_runtime_summary.csv")

    label_meta = read_json_if_exists(reports / "phase5_label_scarcity_metadata.json")
    fixed_meta = read_json_if_exists(reports / "phase5_fixed_recall_metadata.json")
    novel_meta = read_json_if_exists(reports / "phase5_novel_pattern_metadata.json")
    cost_meta = read_json_if_exists(reports / "phase5_scalability_cost_metadata.json")

    rows = []

    rows.append("# Phase 5 Report: Conditional Advantage Tests\n")

    rows.append("## 1. Purpose\n")
    rows.append(
        "Phase 5 examines the conditions under which classical, quantum, and hybrid AML detection models perform better. "
        "Unlike Phase 4, which provides the main model comparison, Phase 5 focuses on conditional decision settings: label scarcity, fixed-recall false-positive reduction, novel-pattern generalization, and scalability/cost trade-offs.\n"
    )

    rows.append("## 2. Label Scarcity Experiment\n")
    rows.append(
        "This experiment tests reduced classical models and the quantum kernel model under limited labeled training data. "
        "The purpose is to identify whether quantum-style reduced feature models become more competitive when fewer labeled samples are available.\n"
    )

    if not label_summary.empty:
        rows.append(label_summary.to_markdown(index=False))
        best_pr = best_row(label_summary, "mean_pr_auc")
        best_f1 = best_row(label_summary, "mean_f1")
        if best_pr:
            rows.append(f"\nBest label-scarcity model by mean PR-AUC: **{best_pr['model']}** at train fraction **{best_pr['train_fraction']}**.")
        if best_f1:
            rows.append(f"Best label-scarcity model by mean F1: **{best_f1['model']}** at train fraction **{best_f1['train_fraction']}**.")
    else:
        rows.append("Label scarcity results not found.")

    rows.append("")

    rows.append("## 3. Fixed-Recall False-Positive Trade-Off\n")
    rows.append(
        "This experiment evaluates Stage 2 classical models at fixed recall targets. "
        "For AML systems, this is important because investigators often require high recall while trying to reduce unnecessary alerts.\n"
    )

    if not fixed_full.empty:
        rows.append(fixed_full.to_markdown(index=False))

        for target in sorted(fixed_full["recall_target"].dropna().unique()):
            subset = fixed_full[fixed_full["recall_target"] == target]
            best_fpr = best_row(subset, "fpr", higher=False)
            best_f1 = best_row(subset, "f1")
            if best_fpr:
                rows.append(
                    f"\nAt recall target **{target}**, the lowest estimated full-pipeline FPR is from **{best_fpr['model']}** "
                    f"with FPR={best_fpr['fpr']} and recall={best_fpr['recall']}."
                )
            if best_f1:
                rows.append(
                    f"At recall target **{target}**, the best F1 is from **{best_f1['model']}** "
                    f"with F1={best_f1['f1']}."
                )
    else:
        rows.append("Fixed-recall results not found.")

    rows.append("")

    rows.append("## 4. Novel-Pattern Stress Test\n")
    rows.append(
        "This experiment creates a proxy AML typology using payment format, currency behavior, and amount bands. "
        "One typology is excluded from training and validation, then evaluated separately as a novel-pattern test subset.\n"
    )

    if novel_meta:
        rows.append(f"Selected novel proxy typology: **{novel_meta.get('selected_novel_typology')}**\n")

    if not novel.empty:
        rows.append(novel.to_markdown(index=False))

        novel_only = novel[novel["split"] == "test_novel_typology"]
        best_novel_f1 = best_row(novel_only, "f1")
        best_novel_recall = best_row(novel_only, "recall")

        if best_novel_f1:
            rows.append(f"\nBest novel-pattern model by F1: **{best_novel_f1['model']}** with F1={best_novel_f1['f1']}.")
        if best_novel_recall:
            rows.append(f"Best novel-pattern model by recall: **{best_novel_recall['model']}** with recall={best_novel_recall['recall']}.")
    else:
        rows.append("Novel-pattern results not found.")

    rows.append("")

    rows.append("## 5. Scalability and Cost Trade-Off\n")
    rows.append(
        "This section summarizes practical workload reduction and runtime/cost implications. "
        "Real IBM quantum hardware queue time and execution time are reserved for Phase 6.\n"
    )

    if not cost.empty:
        rows.append(cost.to_markdown(index=False))

    if cost_meta:
        rows.append(
            f"\nPhase 3B Stage 1 retained fraud at **{cost_meta.get('phase3b_test_fraud_retention')}** "
            f"while passing only **{cost_meta.get('phase3b_test_actual_candidate_pct')}** of test transactions to Stage 2. "
            f"This gives an approximate Stage 2 workload reduction factor of **{cost_meta.get('stage2_workload_reduction_factor')}**."
        )

    if not scalability.empty:
        rows.append("\nRuntime summary:\n")
        rows.append(scalability.to_markdown(index=False))

    rows.append("")

    rows.append("## 6. Phase 5 Decision Summary\n")
    rows.append(
        "Phase 5 supports condition-based conclusions rather than a single universal winner. "
        "Classical and hybrid models remain the most practical for deployment because they provide stronger false-positive control and runtime scalability. "
        "Quantum models remain useful as reduced-feature experimental comparators, especially under constrained feature and label settings, but Phase 4 and Phase 5 evidence should be interpreted carefully because current quantum models still have weak precision and high false-positive rates. "
        "The strongest current practical direction remains the Phase 3B high-recall screener followed by a classical Stage 2 model."
    )

    out_path = reports / "PHASE5_CONDITIONAL_ADVANTAGE_REPORT.md"
    out_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
