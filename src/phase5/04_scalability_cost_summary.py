import argparse
from pathlib import Path
import pandas as pd

from phase5_common import read_json, save_json, write_rows_csv


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-reports", required=True)
    parser.add_argument("--phase5-reports", required=True)
    parser.add_argument("--phase3b-screening", required=True)
    parser.add_argument("--phase4-metadata", required=True)
    parser.add_argument("--out-reports", required=True)
    args = parser.parse_args()

    phase4_reports = Path(args.phase4_reports)
    phase5_reports = Path(args.phase5_reports)
    out_reports = Path(args.out_reports)
    out_reports.mkdir(parents=True, exist_ok=True)

    meta = read_json(args.phase4_metadata)
    target_pct = float(meta["candidate_manifest"]["selected_target_candidate_pct"])

    screening = pd.read_csv(args.phase3b_screening)
    stage1_test = screening[
        (screening["split"] == "test") &
        (screening["target_candidate_pct"].astype(float).round(6) == round(target_pct, 6))
    ].iloc[0].to_dict()

    total_rows = float(stage1_test["total_rows"])
    candidate_rows = float(stage1_test["candidate_rows"])
    actual_candidate_pct = float(stage1_test["actual_candidate_pct"])
    fraud_retention = float(stage1_test["fraud_retention"])

    reduction_factor = total_rows / candidate_rows if candidate_rows > 0 else None
    removed_pct = 1.0 - actual_candidate_pct

    phase4_runtime = read_csv_if_exists(phase4_reports / "phase4_master_runtime_comparison.csv")
    phase4_perf = read_csv_if_exists(phase4_reports / "phase4_master_performance_comparison.csv")
    fixed_recall = read_csv_if_exists(phase5_reports / "phase5_fixed_recall_estimated_full_hybrid.csv")
    quantum_resources = read_csv_if_exists(phase4_reports / "phase4_quantum_resource_estimates.csv")

    scalability_rows = []

    if not phase4_perf.empty:
        test_perf = phase4_perf[phase4_perf["split"] == "test"].copy()

        for _, row in test_perf.iterrows():
            infer_rps = row.get("inference_rows_per_second", None)
            infer_time = row.get("inference_time_seconds", None)

            scalability_rows.append({
                "model": row["model"],
                "track": row["track"],
                "test_recall": row.get("recall"),
                "test_precision": row.get("precision"),
                "test_f1": row.get("f1"),
                "test_fpr": row.get("fpr"),
                "inference_rows_per_second": infer_rps,
                "inference_time_seconds": infer_time,
                "estimated_seconds_per_1m_candidate_rows": (1_000_000 / infer_rps) if pd.notna(infer_rps) and infer_rps not in [0, None] else None,
            })

    cost_rows = [
        {
            "component": "phase3b_stage1_screening",
            "test_total_rows": total_rows,
            "test_candidate_rows": candidate_rows,
            "actual_candidate_pct": actual_candidate_pct,
            "transactions_removed_before_stage2_pct": removed_pct,
            "fraud_retention": fraud_retention,
            "stage2_workload_reduction_factor": reduction_factor,
            "interpretation": "Stage 1 reduces the transaction universe before expensive Stage 2 classical/quantum analysis.",
        }
    ]

    if not quantum_resources.empty:
        for _, row in quantum_resources.iterrows():
            cost_rows.append({
                "component": row["model"],
                "n_qubits": row.get("n_qubits"),
                "statevector_dimension": row.get("statevector_dimension"),
                "estimated_circuit_depth": row.get("estimated_circuit_depth", row.get("estimated_feature_map_depth")),
                "shots": row.get("shots"),
                "hardware_queue_time_seconds": row.get("hardware_queue_time_seconds"),
                "hardware_execution_time_seconds": row.get("hardware_execution_time_seconds"),
                "interpretation": "Hardware queue and execution cost will be measured in Phase 6.",
            })

    write_rows_csv(scalability_rows, out_reports / "phase5_scalability_runtime_summary.csv")
    write_rows_csv(cost_rows, out_reports / "phase5_cost_reduction_summary.csv")

    save_json(
        {
            "experiment": "scalability_cost_summary",
            "phase3b_test_actual_candidate_pct": actual_candidate_pct,
            "phase3b_test_fraud_retention": fraud_retention,
            "stage2_workload_reduction_factor": reduction_factor,
            "transactions_removed_before_stage2_pct": removed_pct,
            "phase6_note": "IBM queue time, hardware execution time, and direct hardware-cost evidence will be added in Phase 6.",
        },
        out_reports / "phase5_scalability_cost_metadata.json",
    )

    print("Scalability and cost summary complete.")
    print(pd.DataFrame(cost_rows))


if __name__ == "__main__":
    main()
