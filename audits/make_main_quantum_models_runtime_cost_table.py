from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

runtime_files = [
    ROOT / "reports" / "phase4" / "phase4_master_runtime_comparison.csv",
]

resource_files = [
    ROOT / "reports" / "phase4" / "phase4_quantum_resource_estimates.csv",
]

phase6_meta_path = ROOT / "reports" / "phase6" / "phase6_hardware_job_metadata.json"
phase6_circuit_path = ROOT / "reports" / "phase6" / "phase6_hardware_circuit_inventory.csv"
phase6_transpiled_path = ROOT / "reports" / "phase6" / "phase6_transpiled_circuit_inventory.csv"

parts = []

# -------------------------
# Phase 4 quantum simulation runtime
# -------------------------
for path in runtime_files:
    if path.exists():
        df = pd.read_csv(path)
        df["source_file"] = str(path)

        if "model" in df.columns:
            quantum_mask = df["model"].astype(str).str.contains(
                "quantum|vqc|qsvc|statevector|kernel",
                case=False,
                na=False
            )

            df = df[quantum_mask].copy()

            if not df.empty:
                parts.append(df)
    else:
        print("[SKIP] Missing:", path)

if parts:
    sim_df = pd.concat(parts, ignore_index=True)
else:
    sim_df = pd.DataFrame()

# -------------------------
# Standardize model names
# -------------------------
def standard_quantum_name(name):
    name = str(name).lower()

    if "hardware" in name or "ibm" in name:
        return "VQC IBM Hardware"

    if "kernel" in name or "qsvc" in name:
        return "Quantum Kernel SVC"

    if "vqc" in name or "variational" in name:
        return "VQC Statevector"

    if "statevector" in name:
        return "Quantum Statevector Model"

    return str(name)

# -------------------------
# Normalize simulation runtime rows
# -------------------------
sim_rows = []

if not sim_df.empty:
    numeric_cols = [
        "training_time_seconds",
        "validation_inference_time_seconds",
        "valid_inference_time_seconds",
        "test_inference_time_seconds",
        "inference_time_seconds",
        "feature_state_time_seconds",
        "kernel_matrix_time_seconds",
        "train_rows",
        "training_rows",
        "valid_rows",
        "test_rows",
        "available_ram_gb",
    ]

    for col in numeric_cols:
        if col not in sim_df.columns:
            sim_df[col] = np.nan
        sim_df[col] = pd.to_numeric(sim_df[col], errors="coerce")

    sim_df["main_model"] = sim_df["model"].apply(standard_quantum_name)

    sim_df["validation_inference_time_seconds_clean"] = sim_df["validation_inference_time_seconds"].fillna(
        sim_df["valid_inference_time_seconds"]
    )

    sim_df["train_rows_clean"] = sim_df["train_rows"].fillna(sim_df["training_rows"])

    sim_df["total_runtime_seconds"] = sim_df[
        [
            "training_time_seconds",
            "validation_inference_time_seconds_clean",
            "test_inference_time_seconds",
            "inference_time_seconds",
            "feature_state_time_seconds",
            "kernel_matrix_time_seconds",
        ]
    ].sum(axis=1, skipna=True)

    sim_df.loc[sim_df["total_runtime_seconds"] == 0, "total_runtime_seconds"] = np.nan

    # Keep fastest/least-cost row per quantum simulation model
    sim_df = sim_df.sort_values(
        by=["main_model", "total_runtime_seconds"],
        ascending=[True, True],
        na_position="last"
    )

    best_sim = sim_df.groupby("main_model", as_index=False).first()

    for _, r in best_sim.iterrows():
        sim_rows.append({
            "model": r["main_model"],
            "evaluation_type": "Statevector simulation",
            "n_qubits": np.nan,
            "n_circuits": np.nan,
            "shots": np.nan,
            "training_time_seconds": r.get("training_time_seconds", np.nan),
            "validation_inference_time_seconds": r.get("validation_inference_time_seconds_clean", np.nan),
            "test_inference_time_seconds": r.get("test_inference_time_seconds", np.nan),
            "other_inference_time_seconds": r.get("inference_time_seconds", np.nan),
            "feature_state_time_seconds": r.get("feature_state_time_seconds", np.nan),
            "kernel_matrix_time_seconds": r.get("kernel_matrix_time_seconds", np.nan),
            "total_runtime_seconds": r.get("total_runtime_seconds", np.nan),
            "transpile_time_seconds": np.nan,
            "wait_for_result_time_seconds": np.nan,
            "total_turnaround_time_seconds": np.nan,
            "ibm_quantum_seconds": np.nan,
            "ibm_usage_seconds": np.nan,
            "ibm_bss_seconds": np.nan,
            "avg_untranspiled_depth": np.nan,
            "avg_transpiled_depth": np.nan,
            "avg_untranspiled_size": np.nan,
            "avg_transpiled_size": np.nan,
            "train_rows": r.get("train_rows_clean", np.nan),
            "test_rows": r.get("test_rows", np.nan),
            "available_ram_gb": r.get("available_ram_gb", np.nan),
            "compute_cost_type": "Quantum simulation cost on classical hardware; no IBM QPU cost",
            "source_file": r.get("source_file", ""),
        })

# -------------------------
# Add Phase 4 quantum resource estimates, if available
# -------------------------
resource_df = pd.DataFrame()

for path in resource_files:
    if path.exists():
        temp = pd.read_csv(path)
        temp["source_file"] = str(path)
        resource_df = pd.concat([resource_df, temp], ignore_index=True)
    else:
        print("[SKIP] Missing:", path)

if not resource_df.empty and sim_rows:
    if "model" in resource_df.columns:
        resource_df["main_model"] = resource_df["model"].apply(standard_quantum_name)

        for row in sim_rows:
            match = resource_df[resource_df["main_model"] == row["model"]]
            if not match.empty:
                m = match.iloc[0]

                for possible_col, target_col in [
                    ("n_qubits", "n_qubits"),
                    ("qubits", "n_qubits"),
                    ("circuit_depth", "avg_untranspiled_depth"),
                    ("depth", "avg_untranspiled_depth"),
                    ("circuit_size", "avg_untranspiled_size"),
                    ("size", "avg_untranspiled_size"),
                    ("n_circuits", "n_circuits"),
                    ("shots", "shots"),
                ]:
                    if possible_col in m.index and pd.notna(m[possible_col]):
                        row[target_col] = m[possible_col]

# -------------------------
# Phase 6 IBM hardware runtime/cost row
# -------------------------
hardware_rows = []

if phase6_meta_path.exists():
    with open(phase6_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    job_metrics = meta.get("job_payload", {}).get("metrics", {})
    usage = job_metrics.get("usage", {})
    bss = job_metrics.get("bss", {})

    avg_un_depth = np.nan
    avg_un_size = np.nan
    avg_tr_depth = np.nan
    avg_tr_size = np.nan

    if phase6_circuit_path.exists():
        cdf = pd.read_csv(phase6_circuit_path)

        if "depth_untranspiled" in cdf.columns:
            avg_un_depth = pd.to_numeric(cdf["depth_untranspiled"], errors="coerce").mean()

        if "size_untranspiled" in cdf.columns:
            avg_un_size = pd.to_numeric(cdf["size_untranspiled"], errors="coerce").mean()

    if phase6_transpiled_path.exists():
        tdf = pd.read_csv(phase6_transpiled_path)

        if "depth_transpiled" in tdf.columns:
            avg_tr_depth = pd.to_numeric(tdf["depth_transpiled"], errors="coerce").mean()

        if "size_transpiled" in tdf.columns:
            avg_tr_size = pd.to_numeric(tdf["size_transpiled"], errors="coerce").mean()

    hardware_rows.append({
        "model": "VQC IBM Hardware",
        "evaluation_type": "Real IBM Quantum hardware",
        "backend_name": meta.get("backend_name"),
        "job_id": meta.get("job_id"),
        "n_qubits": meta.get("n_qubits"),
        "n_circuits": meta.get("n_circuits"),
        "shots": meta.get("shots"),
        "training_time_seconds": np.nan,
        "validation_inference_time_seconds": np.nan,
        "test_inference_time_seconds": np.nan,
        "other_inference_time_seconds": np.nan,
        "feature_state_time_seconds": np.nan,
        "kernel_matrix_time_seconds": np.nan,
        "total_runtime_seconds": np.nan,
        "transpile_time_seconds": meta.get("transpile_time_seconds"),
        "wait_for_result_time_seconds": meta.get("wait_for_result_time_seconds"),
        "total_turnaround_time_seconds": meta.get("total_turnaround_time_seconds"),
        "ibm_quantum_seconds": usage.get("quantum_seconds"),
        "ibm_usage_seconds": usage.get("seconds"),
        "ibm_bss_seconds": bss.get("seconds"),
        "avg_untranspiled_depth": avg_un_depth,
        "avg_transpiled_depth": avg_tr_depth,
        "avg_untranspiled_size": avg_un_size,
        "avg_transpiled_size": avg_tr_size,
        "train_rows": np.nan,
        "test_rows": 80,
        "available_ram_gb": np.nan,
        "compute_cost_type": "Real IBM QPU cost measured by quantum seconds and turnaround time",
        "source_file": str(phase6_meta_path),
    })
else:
    print("[SKIP] Missing:", phase6_meta_path)

# -------------------------
# Final table
# -------------------------
final = pd.DataFrame(sim_rows + hardware_rows)

wanted_cols = [
    "model",
    "evaluation_type",
    "backend_name",
    "job_id",
    "n_qubits",
    "n_circuits",
    "shots",
    "training_time_seconds",
    "validation_inference_time_seconds",
    "test_inference_time_seconds",
    "other_inference_time_seconds",
    "feature_state_time_seconds",
    "kernel_matrix_time_seconds",
    "total_runtime_seconds",
    "transpile_time_seconds",
    "wait_for_result_time_seconds",
    "total_turnaround_time_seconds",
    "ibm_quantum_seconds",
    "ibm_usage_seconds",
    "ibm_bss_seconds",
    "avg_untranspiled_depth",
    "avg_transpiled_depth",
    "avg_untranspiled_size",
    "avg_transpiled_size",
    "train_rows",
    "test_rows",
    "available_ram_gb",
    "compute_cost_type",
    "source_file",
]

for col in wanted_cols:
    if col not in final.columns:
        final[col] = np.nan

final = final[wanted_cols].copy()

round_cols = [
    "training_time_seconds",
    "validation_inference_time_seconds",
    "test_inference_time_seconds",
    "other_inference_time_seconds",
    "feature_state_time_seconds",
    "kernel_matrix_time_seconds",
    "total_runtime_seconds",
    "transpile_time_seconds",
    "wait_for_result_time_seconds",
    "total_turnaround_time_seconds",
    "ibm_quantum_seconds",
    "ibm_usage_seconds",
    "ibm_bss_seconds",
    "avg_untranspiled_depth",
    "avg_transpiled_depth",
    "avg_untranspiled_size",
    "avg_transpiled_size",
    "available_ram_gb",
]

for col in round_cols:
    final[col] = pd.to_numeric(final[col], errors="coerce").round(6)

final = final.sort_values(
    by=["evaluation_type", "model"],
    ascending=True
).reset_index(drop=True)

csv_path = OUT / "main_quantum_models_runtime_cost_table.csv"
txt_path = OUT / "main_quantum_models_runtime_cost_table.txt"

final.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(final.to_string(index=False))

print("Final quantum runtime/cost table created.")
print("CSV:", csv_path)
print("TXT:", txt_path)
print()
print(final.to_string(index=False))
