from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = ROOT / "reports" / "supervisor_fixes"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

def read_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    print("[MISSING]", path)
    return pd.DataFrame()

def read_json(path):
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print("[MISSING]", path)
    return {}

# Classical runtime
classical = read_csv(ROOT / "reports/final/main_classical_models_runtime_cost_table.csv")

if not classical.empty:
    for _, r in classical.iterrows():
        rows.append({
            "model_group": "Classical",
            "model": r.get("model"),
            "evaluation_type": "Classical CPU/GPU runtime",
            "sample_size_or_test_rows": r.get("test_rows"),
            "training_time_seconds": r.get("training_time_seconds"),
            "test_inference_time_seconds": r.get("test_inference_time_seconds"),
            "total_runtime_seconds": r.get("total_runtime_seconds"),
            "n_qubits": np.nan,
            "shots": np.nan,
            "circuits": np.nan,
            "transpile_time_seconds": np.nan,
            "wait_for_result_time_seconds": np.nan,
            "total_turnaround_time_seconds": np.nan,
            "ibm_quantum_seconds": np.nan,
            "cost_interpretation": "Classical runtime only; no IBM QPU cost."
        })

# Quantum simulation runtime
quantum = read_csv(ROOT / "reports/final/main_quantum_models_runtime_cost_table.csv")

if not quantum.empty:
    for _, r in quantum.iterrows():
        rows.append({
            "model_group": "Quantum simulation / hardware",
            "model": r.get("model"),
            "evaluation_type": r.get("evaluation_type"),
            "sample_size_or_test_rows": r.get("test_rows"),
            "training_time_seconds": r.get("training_time_seconds"),
            "test_inference_time_seconds": r.get("test_inference_time_seconds"),
            "total_runtime_seconds": r.get("total_runtime_seconds"),
            "n_qubits": r.get("n_qubits"),
            "shots": r.get("shots"),
            "circuits": r.get("n_circuits"),
            "transpile_time_seconds": r.get("transpile_time_seconds"),
            "wait_for_result_time_seconds": r.get("wait_for_result_time_seconds"),
            "total_turnaround_time_seconds": r.get("total_turnaround_time_seconds"),
            "ibm_quantum_seconds": r.get("ibm_quantum_seconds"),
            "cost_interpretation": r.get("compute_cost_type")
        })

# Hybrid quantum simulation runtime
hybrid_sim = read_csv(ROOT / "reports/final/isolation_forest_quantum_hybrid_runtime_cost_table.csv")

if not hybrid_sim.empty:
    for _, r in hybrid_sim.iterrows():
        rows.append({
            "model_group": "Isolation Forest + quantum simulation",
            "model": r.get("model"),
            "evaluation_type": "Statevector simulation on Isolation Forest-selected subset",
            "sample_size_or_test_rows": r.get("test_rows"),
            "training_time_seconds": r.get("training_time_seconds"),
            "test_inference_time_seconds": r.get("test_inference_time_seconds"),
            "total_runtime_seconds": r.get("total_runtime_seconds"),
            "n_qubits": r.get("n_qubits"),
            "shots": r.get("shots"),
            "circuits": r.get("n_circuits_or_state_evaluations"),
            "transpile_time_seconds": np.nan,
            "wait_for_result_time_seconds": np.nan,
            "total_turnaround_time_seconds": np.nan,
            "ibm_quantum_seconds": np.nan,
            "cost_interpretation": r.get("compute_cost_type")
        })

# Hybrid quantum hardware runtime
hybrid_hw = read_csv(ROOT / "reports/final/isolation_forest_quantum_hybrid_hardware_runtime_cost_table.csv")

if not hybrid_hw.empty:
    for _, r in hybrid_hw.iterrows():
        rows.append({
            "model_group": "Isolation Forest + quantum hardware",
            "model": r.get("model"),
            "evaluation_type": "Real IBM Quantum hardware feasibility test",
            "sample_size_or_test_rows": r.get("test_rows"),
            "training_time_seconds": r.get("kernel_train_time_seconds"),
            "test_inference_time_seconds": np.nan,
            "total_runtime_seconds": np.nan,
            "n_qubits": r.get("n_qubits"),
            "shots": r.get("shots"),
            "circuits": r.get("model_circuits"),
            "transpile_time_seconds": r.get("transpile_time_seconds"),
            "wait_for_result_time_seconds": r.get("wait_for_result_time_seconds"),
            "total_turnaround_time_seconds": r.get("total_turnaround_time_seconds"),
            "ibm_quantum_seconds": r.get("ibm_quantum_seconds"),
            "cost_interpretation": r.get("compute_cost_type")
        })

df = pd.DataFrame(rows)

for col in [
    "training_time_seconds",
    "test_inference_time_seconds",
    "total_runtime_seconds",
    "transpile_time_seconds",
    "wait_for_result_time_seconds",
    "total_turnaround_time_seconds",
    "ibm_quantum_seconds",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

df.to_csv(OUT / "final_runtime_comparison_table.csv", index=False)

with open(OUT / "final_runtime_comparison_table.txt", "w", encoding="utf-8") as f:
    f.write(df.to_string(index=False))

print(df.to_string(index=False))
print("\nCreated reports/supervisor_fixes/final_runtime_comparison_table.csv")
