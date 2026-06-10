from pathlib import Path
import pandas as pd
import json

REPORTS = Path("/kaggle/working/aml_phase6/reports")

required_core = [
    "phase6_subset_counts.csv",
    "phase6_subset_metadata.json",
    "phase6_statevector_reference_metrics.csv",
    "phase6_statevector_training_metadata.json",
    "phase6_hardware_circuit_inventory.csv",
    "PHASE6_IBM_HARDWARE_VALIDATION_REPORT.md",
]

required_hardware = [
    "phase6_transpiled_circuit_inventory.csv",
    "phase6_hardware_counts.csv",
    "phase6_hardware_scores.csv",
    "phase6_hardware_metrics.csv",
    "phase6_hardware_job_metadata.json",
    "phase6_simulator_vs_hardware_scores.csv",
    "phase6_simulator_hardware_metric_comparison.csv",
    "phase6_simulator_vs_hardware_summary.json",
]

print("\nPHASE 6 OUTPUT CHECK")
print("====================")

core_ok = True

for f in required_core:
    p = REPORTS / f
    if p.exists():
        print("[OK]", f)
    else:
        print("[MISSING]", f)
        core_ok = False

hardware_ok = True

print("\nHardware-specific files:")

for f in required_hardware:
    p = REPORTS / f
    if p.exists():
        print("[OK]", f)
    else:
        print("[MISSING]", f)
        hardware_ok = False

if core_ok:
    print("\n[OK] Phase 6 core simulator and circuit-preparation validation is complete.")

if hardware_ok:
    hw = pd.read_csv(REPORTS / "phase6_hardware_metrics.csv")
    comp = pd.read_csv(REPORTS / "phase6_simulator_hardware_metric_comparison.csv")

    with open(REPORTS / "phase6_hardware_job_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\nHardware job summary:")
    print("Backend:", meta.get("backend_name"))
    print("Job ID:", meta.get("job_id"))
    print("Shots:", meta.get("shots"))
    print("Circuits:", meta.get("n_circuits"))
    print("Total turnaround seconds:", meta.get("total_turnaround_time_seconds"))

    metrics = meta.get("job_payload", {}).get("metrics", {})
    usage = metrics.get("usage", {})

    print("IBM quantum seconds:", usage.get("quantum_seconds"))
    print("IBM status:", usage.get("status"))

    print("\nHardware metrics:")
    print(hw.to_string(index=False))

    print("\nSimulator vs hardware metrics:")
    print(comp.to_string(index=False))

    print("\n[OK] Phase 6 IBM hardware validation is complete and reviewable.")
else:
    print("\n[WARNING] Hardware files are incomplete. Real IBM hardware validation is not fully complete.")
