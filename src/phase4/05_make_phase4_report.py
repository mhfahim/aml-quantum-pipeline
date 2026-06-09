import argparse
from pathlib import Path
import pandas as pd
import json


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_row(df, split="test", metric="pr_auc"):
    if df.empty or metric not in df.columns:
        return None

    d = df[df["split"] == split].copy()

    if d.empty:
        return None

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])

    if d.empty:
        return None

    return d.sort_values(metric, ascending=False).iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    args = parser.parse_args()

    reports = Path(args.reports)

    data_meta = read_json(reports / "phase4_data_metadata.json")

    classical_candidate = read_csv_if_exists(reports / "phase4_classical_candidate_metrics.csv")
    classical_hybrid = read_csv_if_exists(reports / "phase4_classical_hybrid_estimated_full_metrics.csv")
    reduced_classical = read_csv_if_exists(reports / "phase4_reduced_classical_metrics.csv")
    quantum = read_csv_if_exists(reports / "phase4_quantum_statevector_metrics.csv")

    runtime_files = [
        reports / "phase4_classical_candidate_runtime.csv",
        reports / "phase4_reduced_classical_runtime.csv",
        reports / "phase4_quantum_statevector_runtime.csv",
    ]

    runtime_df = pd.concat(
        [read_csv_if_exists(p) for p in runtime_files],
        axis=0,
        ignore_index=True,
    )

    performance_parts = []
    for df in [classical_candidate, classical_hybrid, reduced_classical, quantum]:
        if not df.empty:
            performance_parts.append(df)

    if performance_parts:
        master = pd.concat(performance_parts, axis=0, ignore_index=True)
    else:
        master = pd.DataFrame()

    master_path = reports / "phase4_master_performance_comparison.csv"
    runtime_path = reports / "phase4_master_runtime_comparison.csv"

    master.to_csv(master_path, index=False)
    runtime_df.to_csv(runtime_path, index=False)

    rows = []

    rows.append("# Phase 4 Report: Classical vs Quantum vs Hybrid Model Comparison\n")

    rows.append("## 1. Purpose\n")
    rows.append(
        "Phase 4 performs the main model comparison for the thesis. "
        "The comparison is organized into two segments: predictive performance and runtime/resource cost. "
        "Classical models are evaluated on the Phase 3B candidate pool, reduced classical models are evaluated on the same quantum-compatible features as the quantum models, and quantum models are evaluated using statevector simulation. "
        "Hybrid results combine the Phase 3B high-recall screener with Stage 2 candidate-pool classifiers.\n"
    )

    rows.append("## 2. Data Used\n")
    rows.append(f"- Candidate sample rows: **{data_meta['candidate_rows']:,}**")
    rows.append(f"- Quantum subset rows: **{data_meta['quantum_subset_rows']:,}**")
    rows.append(f"- Number of qubits/features for quantum track: **{data_meta['n_qubits']}**")
    rows.append(f"- Candidate selected target from Phase 3B: **{data_meta['candidate_manifest']['selected_target_candidate_pct']}**")
    rows.append(f"- Negative sample weighting rule: **{data_meta['sample_weight_rule']}**\n")

    rows.append("## 3. Performance Comparison Tracks\n")

    rows.append("### Track A: Candidate-Pool Classical Stage 2 Models\n")
    if not classical_candidate.empty:
        rows.append(classical_candidate[classical_candidate["split"] == "test"].to_markdown(index=False))
    else:
        rows.append("No classical candidate-pool metrics found.")
    rows.append("")

    rows.append("### Track B: Estimated Full Hybrid Pipeline Metrics\n")
    if not classical_hybrid.empty:
        rows.append(classical_hybrid.to_markdown(index=False))
    else:
        rows.append("No estimated full hybrid metrics found.")
    rows.append("")

    rows.append("### Track C: Reduced Classical Models on Quantum-Compatible Features\n")
    if not reduced_classical.empty:
        rows.append(reduced_classical[reduced_classical["split"] == "test"].to_markdown(index=False))
    else:
        rows.append("No reduced classical metrics found.")
    rows.append("")

    rows.append("### Track D: Quantum Statevector Models\n")
    if not quantum.empty:
        rows.append(quantum[quantum["split"] == "test"].to_markdown(index=False))
    else:
        rows.append("No quantum metrics found.")
    rows.append("")

    rows.append("## 4. Runtime and Resource Comparison\n")
    if not runtime_df.empty:
        rows.append(runtime_df.to_markdown(index=False))
    else:
        rows.append("No runtime metrics found.")
    rows.append("")

    rows.append("## 5. Best Model Indicators\n")

    comparisons = [
        ("Best candidate-pool classical by PR-AUC", classical_candidate, "pr_auc"),
        ("Best candidate-pool classical by F1", classical_candidate, "f1"),
        ("Best estimated full hybrid by F1", classical_hybrid, "f1"),
        ("Best reduced classical by PR-AUC", reduced_classical, "pr_auc"),
        ("Best quantum by PR-AUC", quantum, "pr_auc"),
        ("Best quantum by F1", quantum, "f1"),
    ]

    for label, df, metric in comparisons:
        row = best_row(df, split="test", metric=metric)
        if row:
            rows.append(
                f"- **{label}:** {row['model']} "
                f"({metric}={row[metric]}, recall={row.get('recall')}, precision={row.get('precision')})"
            )
        else:
            rows.append(f"- **{label}:** not available")

    rows.append("")
    rows.append("## 6. Interpretation Guidance\n")
    rows.append(
        "The final thesis decision should not be based only on accuracy because the IBM AML dataset is extremely imbalanced. "
        "PR-AUC, recall, F1-score, false positive rate, and runtime/resource cost are more important. "
        "The strongest practical deployment model is likely to be the model that gives high recall and acceptable false positives at the lowest runtime cost. "
        "Quantum models should be interpreted under reduced-feature and simulator constraints in Phase 4; real IBM hardware queue time and execution cost will be evaluated in Phase 6."
    )

    out_path = reports / "PHASE4_MODEL_COMPARISON_REPORT.md"
    out_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {master_path}")
    print(f"Wrote {runtime_path}")


if __name__ == "__main__":
    main()
