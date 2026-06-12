import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--hardware-final", required=True)
    parser.add_argument("--out-final", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root)
    hardware_final = Path(args.hardware_final)
    out_final = Path(args.out_final)
    out_final.mkdir(parents=True, exist_ok=True)

    rows = []

    classical_path = root / "reports" / "final" / "main_classical_models_runtime_cost_table.csv"
    hardware_path = hardware_final / "isolation_forest_quantum_hybrid_hardware_300_split_runtime_cost_table.csv"

    if classical_path.exists():
        classical = pd.read_csv(classical_path)

        for _, r in classical.iterrows():
            rows.append({
                "model_group": "Classical",
                "model": r.get("model"),
                "evaluation_type": "Classical inference/runtime",
                "test_rows": r.get("test_rows"),
                "training_time_seconds": r.get("training_time_seconds"),
                "test_inference_time_seconds": r.get("test_inference_time_seconds"),
                "total_runtime_seconds": r.get("total_runtime_seconds"),
                "n_qubits": np.nan,
                "shots": np.nan,
                "model_circuits": np.nan,
                "total_submitted_circuits": np.nan,
                "transpile_time_seconds": np.nan,
                "wait_for_result_time_seconds": np.nan,
                "total_turnaround_time_seconds": np.nan,
                "ibm_quantum_seconds": np.nan,
                "interpretation": "Classical runtime. No IBM QPU queue or quantum execution cost."
            })

    hardware = pd.read_csv(hardware_path)

    for _, r in hardware.iterrows():
        rows.append({
            "model_group": "IBM hardware, 300-sample feasibility",
            "model": r.get("model"),
            "evaluation_type": "Real IBM Quantum hardware runtime",
            "test_rows": r.get("test_rows"),
            "training_time_seconds": r.get("kernel_train_time_seconds"),
            "test_inference_time_seconds": np.nan,
            "total_runtime_seconds": np.nan,
            "n_qubits": r.get("n_qubits"),
            "shots": r.get("shots"),
            "model_circuits": r.get("model_circuits"),
            "total_submitted_circuits": r.get("total_submitted_circuits"),
            "transpile_time_seconds": r.get("transpile_time_seconds"),
            "wait_for_result_time_seconds": r.get("wait_for_result_time_seconds"),
            "total_turnaround_time_seconds": r.get("total_turnaround_time_seconds"),
            "ibm_quantum_seconds": r.get("ibm_quantum_seconds"),
            "interpretation": "Includes IBM hardware transpilation, queue/wait/result time, circuits, shots, and quantum usage."
        })

    out = pd.DataFrame(rows)

    for col in [
        "training_time_seconds",
        "test_inference_time_seconds",
        "total_runtime_seconds",
        "transpile_time_seconds",
        "wait_for_result_time_seconds",
        "total_turnaround_time_seconds",
        "ibm_quantum_seconds",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(6)

    out.to_csv(out_final / "hardware_300_split_runtime_vs_classical_table.csv", index=False)

    with open(out_final / "hardware_300_split_runtime_vs_classical_table.txt", "w", encoding="utf-8") as f:
        f.write(out.to_string(index=False))

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
