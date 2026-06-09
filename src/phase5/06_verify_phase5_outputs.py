from pathlib import Path
import pandas as pd

REPORTS = Path("/kaggle/working/aml_phase5/reports")

required = [
    "phase5_label_scarcity_results.csv",
    "phase5_label_scarcity_summary.csv",
    "phase5_label_scarcity_metadata.json",
    "phase5_fixed_recall_candidate_pool.csv",
    "phase5_fixed_recall_estimated_full_hybrid.csv",
    "phase5_fixed_recall_runtime.csv",
    "phase5_fixed_recall_metadata.json",
    "phase5_novel_typology_candidates.csv",
    "phase5_novel_pattern_results.csv",
    "phase5_novel_pattern_runtime.csv",
    "phase5_novel_pattern_metadata.json",
    "phase5_scalability_runtime_summary.csv",
    "phase5_cost_reduction_summary.csv",
    "phase5_scalability_cost_metadata.json",
    "PHASE5_CONDITIONAL_ADVANTAGE_REPORT.md",
]

print("\nPHASE 5 OUTPUT CHECK")
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
    label = pd.read_csv(REPORTS / "phase5_label_scarcity_summary.csv")
    fixed = pd.read_csv(REPORTS / "phase5_fixed_recall_estimated_full_hybrid.csv")
    novel = pd.read_csv(REPORTS / "phase5_novel_pattern_results.csv")
    cost = pd.read_csv(REPORTS / "phase5_cost_reduction_summary.csv")

    print("\nLabel scarcity rows:", len(label))
    print("Fixed-recall rows:", len(fixed))
    print("Novel-pattern rows:", len(novel))
    print("Cost summary rows:", len(cost))

    print("\nBest fixed-recall full hybrid rows by recall target:")
    for target in sorted(fixed["recall_target"].dropna().unique()):
        sub = fixed[fixed["recall_target"] == target].copy()
        sub["f1"] = pd.to_numeric(sub["f1"], errors="coerce")
        best = sub.sort_values("f1", ascending=False).iloc[0]
        print(
            "target=", target,
            "| best=", best["model"],
            "| f1=", best["f1"],
            "| recall=", best["recall"],
            "| fpr=", best["fpr"]
        )

    print("\nNovel-pattern test rows:")
    print(novel[["split", "model", "precision", "recall", "f1", "fpr", "fnr"]].to_string(index=False))

    print("\n[OK] Phase 5 is complete and reviewable.")
else:
    print("\n[PROBLEM] Some Phase 5 files are missing.")
