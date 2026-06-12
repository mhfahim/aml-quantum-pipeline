import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def clean_model_name(name):
    name = str(name).lower()

    if "kernel" in name or "qksvc" in name:
        return "Isolation Forest + Quantum Kernel SVC IBM Hardware, 300-sample split test"

    if "vqc" in name:
        return "Isolation Forest + VQC IBM Hardware, 300-sample split test"

    return str(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qksvc-reports", required=True)
    parser.add_argument("--vqc-reports", required=True)
    parser.add_argument("--final-reports", required=True)
    args = parser.parse_args()

    qksvc = Path(args.qksvc_reports)
    vqc = Path(args.vqc_reports)
    final = Path(args.final_reports)
    final.mkdir(parents=True, exist_ok=True)

    perf_files = [
        qksvc / "qksvc_hardware300_performance_metrics.csv",
        vqc / "vqc_hardware300_performance_metrics.csv",
    ]

    runtime_files = [
        qksvc / "qksvc_hardware300_runtime_cost.csv",
        vqc / "vqc_hardware300_runtime_cost.csv",
    ]

    score_files = [
        qksvc / "qksvc_hardware300_scores.csv",
        vqc / "vqc_hardware300_scores.csv",
    ]

    perf = pd.concat([pd.read_csv(f) for f in perf_files if f.exists()], ignore_index=True)
    runtime = pd.concat([pd.read_csv(f) for f in runtime_files if f.exists()], ignore_index=True)
    scores = pd.concat([pd.read_csv(f) for f in score_files if f.exists()], ignore_index=True)

    perf["model"] = perf["model"].apply(clean_model_name)
    runtime["model"] = runtime["model"].apply(clean_model_name)
    scores["model"] = scores["model"].apply(clean_model_name)

    for col in [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "fpr", "fnr", "tn", "fp", "fn", "tp", "threshold", "n_rows"
    ]:
        if col not in perf.columns:
            perf[col] = np.nan
        perf[col] = pd.to_numeric(perf[col], errors="coerce")

    perf["confusion_matrix"] = perf.apply(
        lambda r: f"TN={int(round(r['tn']))}, FP={int(round(r['fp']))}, FN={int(round(r['fn']))}, TP={int(round(r['tp']))}",
        axis=1,
    )

    perf_final = perf.rename(columns={"f1": "f1_score"})[
        [
            "model",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "pr_auc",
            "roc_auc",
            "fpr",
            "fnr",
            "tn",
            "fp",
            "fn",
            "tp",
            "confusion_matrix",
            "threshold",
            "n_rows",
        ]
    ].copy()

    for col in [
        "accuracy", "f1_score", "precision", "recall",
        "pr_auc", "roc_auc", "fpr", "fnr", "threshold"
    ]:
        perf_final[col] = pd.to_numeric(perf_final[col], errors="coerce").round(6)

    perf_final = perf_final.sort_values("f1_score", ascending=False).reset_index(drop=True)

    perf_final.to_csv(
        final / "isolation_forest_quantum_hybrid_hardware_300_split_performance_table.csv",
        index=False,
    )

    with open(final / "isolation_forest_quantum_hybrid_hardware_300_split_performance_table.txt", "w", encoding="utf-8") as f:
        f.write(perf_final.to_string(index=False))

    runtime_final = runtime[
        [
            "model",
            "stage1_model",
            "stage2_model",
            "backend_name",
            "job_id",
            "n_qubits",
            "shots",
            "model_circuits",
            "total_submitted_circuits",
            "kernel_train_time_seconds",
            "transpile_time_seconds",
            "wait_for_result_time_seconds",
            "total_turnaround_time_seconds",
            "ibm_quantum_seconds",
            "ibm_usage_seconds",
            "train_rows",
            "valid_rows",
            "test_rows",
            "compute_cost_type",
        ]
    ].copy()

    for col in [
        "kernel_train_time_seconds",
        "transpile_time_seconds",
        "wait_for_result_time_seconds",
        "total_turnaround_time_seconds",
        "ibm_quantum_seconds",
        "ibm_usage_seconds",
    ]:
        runtime_final[col] = pd.to_numeric(runtime_final[col], errors="coerce").round(6)

    runtime_final = runtime_final.sort_values("model").reset_index(drop=True)

    runtime_final.to_csv(
        final / "isolation_forest_quantum_hybrid_hardware_300_split_runtime_cost_table.csv",
        index=False,
    )

    with open(final / "isolation_forest_quantum_hybrid_hardware_300_split_runtime_cost_table.txt", "w", encoding="utf-8") as f:
        f.write(runtime_final.to_string(index=False))

    scores.to_csv(
        final / "isolation_forest_quantum_hybrid_hardware_300_split_scores.csv",
        index=False,
    )

    print("Performance table:")
    print(perf_final)

    print("\nRuntime table:")
    print(runtime_final)

    print("\nSaved scores:")
    print(final / "isolation_forest_quantum_hybrid_hardware_300_split_scores.csv")


if __name__ == "__main__":
    main()
