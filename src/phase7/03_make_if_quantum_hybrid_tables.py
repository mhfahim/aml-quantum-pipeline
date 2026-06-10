import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def clean_model_name(name):
    name = str(name).lower()

    if "kernel" in name:
        return "Isolation Forest + Quantum Kernel SVC"

    if "vqc" in name:
        return "Isolation Forest + VQC Statevector"

    return str(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7-reports", required=True)
    parser.add_argument("--final-reports", required=True)
    args = parser.parse_args()

    phase7 = Path(args.phase7_reports)
    final = Path(args.final_reports)

    final.mkdir(parents=True, exist_ok=True)

    perf = pd.read_csv(phase7 / "phase7_if_quantum_hybrid_performance_metrics.csv")
    runtime = pd.read_csv(phase7 / "phase7_if_quantum_hybrid_runtime_cost.csv")

    perf["model"] = perf["model"].apply(clean_model_name)

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
        perf_final[col] = perf_final[col].round(6)

    perf_final = perf_final.sort_values("f1_score", ascending=False).reset_index(drop=True)

    perf_final.to_csv(
        final / "isolation_forest_quantum_hybrid_performance_table.csv",
        index=False,
    )

    with open(final / "isolation_forest_quantum_hybrid_performance_table.txt", "w", encoding="utf-8") as f:
        f.write(perf_final.to_string(index=False))

    runtime["model"] = runtime["model"].apply(clean_model_name)

    runtime_final = runtime[
        [
            "model",
            "stage1_model",
            "stage2_model",
            "n_qubits",
            "n_circuits_or_state_evaluations",
            "shots",
            "feature_state_time_seconds",
            "kernel_matrix_time_seconds",
            "training_time_seconds",
            "validation_inference_time_seconds",
            "test_inference_time_seconds",
            "total_runtime_seconds",
            "train_rows",
            "valid_rows",
            "test_rows",
            "compute_cost_type",
        ]
    ].copy()

    for col in [
        "feature_state_time_seconds",
        "kernel_matrix_time_seconds",
        "training_time_seconds",
        "validation_inference_time_seconds",
        "test_inference_time_seconds",
        "total_runtime_seconds",
    ]:
        runtime_final[col] = pd.to_numeric(runtime_final[col], errors="coerce").round(6)

    runtime_final = runtime_final.sort_values("total_runtime_seconds", ascending=True).reset_index(drop=True)

    runtime_final.to_csv(
        final / "isolation_forest_quantum_hybrid_runtime_cost_table.csv",
        index=False,
    )

    with open(final / "isolation_forest_quantum_hybrid_runtime_cost_table.txt", "w", encoding="utf-8") as f:
        f.write(runtime_final.to_string(index=False))

    print("Performance table:")
    print(perf_final)

    print("\nRuntime/cost table:")
    print(runtime_final)


if __name__ == "__main__":
    main()
