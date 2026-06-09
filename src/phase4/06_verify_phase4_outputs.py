from pathlib import Path
import pandas as pd

REPORTS = Path("/kaggle/working/aml_phase4/reports")

required = [
    "phase4_data_metadata.json",
    "phase4_candidate_sample_counts.csv",
    "phase4_quantum_subset_counts.csv",
    "phase4_classical_candidate_metrics.csv",
    "phase4_classical_hybrid_estimated_full_metrics.csv",
    "phase4_classical_candidate_runtime.csv",
    "phase4_reduced_classical_metrics.csv",
    "phase4_reduced_classical_runtime.csv",
    "phase4_quantum_statevector_metrics.csv",
    "phase4_quantum_statevector_runtime.csv",
    "phase4_quantum_resource_estimates.csv",
    "phase4_master_performance_comparison.csv",
    "phase4_master_runtime_comparison.csv",
    "PHASE4_MODEL_COMPARISON_REPORT.md",
]

print("\nPHASE 4 OUTPUT CHECK")
print("====================")

all_ok = True
for f in required:
    p = REPORTS / f
    if p.exists():
        print("[OK]", f)
    else:
        print("[MISSING]", f)
        all_ok = False

if all_ok:
    perf = pd.read_csv(REPORTS / "phase4_master_performance_comparison.csv")
    runtime = pd.read_csv(REPORTS / "phase4_master_runtime_comparison.csv")

    print("\nPerformance rows:", len(perf))
    print("Runtime rows:", len(runtime))

    print("\nModels compared:")
    print(perf["model"].drop_duplicates().to_string(index=False))

    print("\nTracks compared:")
    print(perf["track"].drop_duplicates().to_string(index=False))

    print("\nTest performance summary:")
    cols = ["track", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr"]
    test = perf[perf["split"] == "test"].copy()
    print(test[cols].to_string(index=False))

    print("\n[OK] Phase 4 is complete and reviewable.")
else:
    print("\n[PROBLEM] Some Phase 4 files are missing.")
