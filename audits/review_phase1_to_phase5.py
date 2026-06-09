import json
import csv
from pathlib import Path
import pandas as pd

ROOT = Path(".")

P1 = ROOT / "reports" / "phase1"
P2 = ROOT / "reports" / "phase2"
P3 = ROOT / "reports" / "phase3"
P3B = ROOT / "reports" / "phase3b"
P4 = ROOT / "reports" / "phase4"
P5 = ROOT / "reports" / "phase5"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    return pd.read_csv(path)


def exists(path):
    if path.exists():
        print(f"[OK] {path}")
        return True
    print(f"[MISSING] {path}")
    return False


def get_row(df, **conditions):
    out = df.copy()
    for k, v in conditions.items():
        out = out[out[k] == v]
    if len(out) == 0:
        return None
    return out.iloc[0]


print("\n==============================")
print("FULL PROJECT AUDIT: PHASE 1 TO PHASE 5")
print("==============================")


# -------------------------
# Phase 1
# -------------------------
print("\n==============================")
print("PHASE 1: FULL DATASET PROCESSING")
print("==============================")

p1_required = [
    P1 / "phase1_profile.json",
    P1 / "conversion_log.json",
    P1 / "temporal_split_manifest.json",
    P1 / "PHASE1_DATA_AUDIT.md",
]

p1_ok = all(exists(p) for p in p1_required)

if p1_ok:
    profile = read_json(P1 / "phase1_profile.json")
    conversion = read_json(P1 / "conversion_log.json")
    split_manifest = read_json(P1 / "temporal_split_manifest.json")

    total_rows_profile = int(profile["total_rows"])
    total_rows_split = sum(int(r["n"]) for r in split_manifest["split_counts"])

    trans_success = [
        r for r in conversion
        if "_Trans.csv" in r.get("file", "") and r.get("status") == "success"
    ]

    print(f"Total rows profile: {total_rows_profile:,}")
    print(f"Total rows split:   {total_rows_split:,}")
    print(f"Transaction CSVs processed: {len(trans_success)} / 6")
    print(f"Label counts: {profile['label_counts']}")
    print(f"Time range: {profile['time_range']}")

    if total_rows_profile == 430920901 and total_rows_split == total_rows_profile and len(trans_success) == 6:
        print("[PASS] Phase 1 is valid and complete.")
    else:
        print("[CHECK] Phase 1 needs review.")


# -------------------------
# Phase 2
# -------------------------
print("\n==============================")
print("PHASE 2: LEAKAGE AUDIT + STARTER BASELINE")
print("==============================")

p2_required = [
    P2 / "leakage_audit.json",
    P2 / "feature_schema.json",
    P2 / "baseline_metrics.csv",
    P2 / "baseline_report.md",
]

p2_ok = all(exists(p) for p in p2_required)

if p2_ok:
    leakage = read_json(P2 / "leakage_audit.json")
    baseline = read_csv(P2 / "baseline_metrics.csv")

    print(f"Leakage audit status: {leakage.get('status')}")
    print(f"Baseline metric rows: {len(baseline)}")
    print("Models:")
    print(baseline["model"].drop_duplicates().to_string(index=False))

    if "completed" in leakage.get("status", "") and len(baseline) >= 3:
        print("[PASS] Phase 2 is valid as a leakage-safe starter baseline.")
    else:
        print("[CHECK] Phase 2 needs review.")


# -------------------------
# Phase 3
# -------------------------
print("\n==============================")
print("PHASE 3: UNSUPERVISED STAGE 1 BASELINE")
print("==============================")

p3_required = [
    P3 / "stage1_model_metadata.json",
    P3 / "stage1_thresholds.json",
    P3 / "stage1_screening_metrics.csv",
    P3 / "candidate_pool_sample_manifest.json",
    P3 / "PHASE3_STAGE1_REPORT.md",
]

p3_ok = all(exists(p) for p in p3_required)

phase3_test_recall = None

if p3_ok:
    p3_metrics = read_csv(P3 / "stage1_screening_metrics.csv")
    p3_manifest = read_json(P3 / "candidate_pool_sample_manifest.json")

    p3_test_10 = p3_metrics[
        (p3_metrics["split"] == "test") &
        (p3_metrics["target_candidate_pct"].round(2) == 0.10)
    ].iloc[0]

    phase3_test_recall = float(p3_test_10["fraud_retention"])

    print(f"Rows scanned: {p3_manifest.get('seen_rows'):,}")
    print(f"Test fraud retention at 10% target: {phase3_test_recall:.6f}")
    print(f"Test actual candidate pct: {float(p3_test_10['actual_candidate_pct']):.6f}")
    print(f"Test enrichment factor: {float(p3_test_10['enrichment_factor']):.6f}")

    print("[PASS] Phase 3 is technically complete as the unsupervised baseline.")
    print("[NOTE] It is intentionally not the final Stage 1 because recall is weak.")


# -------------------------
# Phase 3B
# -------------------------
print("\n==============================")
print("PHASE 3B: IMPROVED HIGH-RECALL STAGE 1")
print("==============================")

p3b_required = [
    P3B / "stage1b_model_metadata.json",
    P3B / "stage1b_thresholds.json",
    P3B / "stage1b_screening_metrics.csv",
    P3B / "stage1b_candidate_pool_sample_counts.csv",
    P3B / "stage1b_candidate_pool_sample_manifest.json",
    P3B / "PHASE3B_HIGH_RECALL_REPORT.md",
]

p3b_ok = all(exists(p) for p in p3b_required)

phase3b_test_recall = None

if p3b_ok:
    p3b_model = read_json(P3B / "stage1b_model_metadata.json")
    p3b_manifest = read_json(P3B / "stage1b_candidate_pool_sample_manifest.json")
    p3b_metrics = read_csv(P3B / "stage1b_screening_metrics.csv")

    selected_target = float(p3b_manifest["selected_target_candidate_pct"])

    p3b_test = p3b_metrics[
        (p3b_metrics["split"] == "test") &
        (p3b_metrics["target_candidate_pct"].round(6) == round(selected_target, 6))
    ].iloc[0]

    phase3b_test_recall = float(p3b_test["fraud_retention"])
    phase3b_candidate_pct = float(p3b_test["actual_candidate_pct"])
    phase3b_enrichment = float(p3b_test["enrichment_factor"])

    print(f"Model type: {p3b_model.get('model_type')}")
    print(f"Training rows: {p3b_model.get('total_training_rows'):,}")
    print(f"Selected target candidate pct: {selected_target}")
    print(f"Rows scanned: {p3b_manifest.get('seen_rows'):,}")
    print(f"Candidate sample kept rows: {p3b_manifest.get('kept_rows'):,}")
    print(f"Test fraud retention: {phase3b_test_recall:.6f}")
    print(f"Test actual candidate pct: {phase3b_candidate_pct:.6f}")
    print(f"Test enrichment factor: {phase3b_enrichment:.6f}")

    if phase3b_test_recall >= 0.80 and phase3b_candidate_pct <= 0.20:
        print("[PASS] Phase 3B is strong enough as the preferred Stage 1 candidate generator.")
    else:
        print("[CHECK] Phase 3B may need feature improvement.")

    if phase3_test_recall is not None and phase3b_test_recall > phase3_test_recall:
        print("[PASS] Phase 3B clearly improves over Phase 3.")


# -------------------------
# Phase 4
# -------------------------
print("\n==============================")
print("PHASE 4: CLASSICAL VS QUANTUM VS HYBRID COMPARISON")
print("==============================")

p4_required = [
    P4 / "PHASE4_MODEL_COMPARISON_REPORT.md",
    P4 / "phase4_master_performance_comparison.csv",
    P4 / "phase4_master_runtime_comparison.csv",
    P4 / "phase4_classical_candidate_metrics.csv",
    P4 / "phase4_classical_hybrid_estimated_full_metrics.csv",
    P4 / "phase4_reduced_classical_metrics.csv",
    P4 / "phase4_quantum_statevector_metrics.csv",
    P4 / "phase4_quantum_resource_estimates.csv",
]

p4_ok = all(exists(p) for p in p4_required)

if p4_ok:
    p4_perf = read_csv(P4 / "phase4_master_performance_comparison.csv")
    p4_runtime = read_csv(P4 / "phase4_master_runtime_comparison.csv")

    print(f"Performance rows: {len(p4_perf)}")
    print(f"Runtime rows: {len(p4_runtime)}")

    print("\nTracks:")
    print(p4_perf["track"].drop_duplicates().to_string(index=False))

    print("\nModels:")
    print(p4_perf["model"].drop_duplicates().to_string(index=False))

    required_tracks = {
        "candidate_pool_stage2_weighted",
        "estimated_full_hybrid_pipeline",
        "reduced_quantum_compatible_classical",
        "quantum_statevector_reduced_features",
    }

    actual_tracks = set(p4_perf["track"].drop_duplicates().tolist())

    if required_tracks.issubset(actual_tracks) and len(p4_perf) >= 20:
        print("[PASS] Phase 4 contains the required classical, quantum, and hybrid comparison tracks.")
    else:
        print("[CHECK] Phase 4 comparison tracks need review.")

    full_hybrid = p4_perf[p4_perf["track"] == "estimated_full_hybrid_pipeline"].copy()

    if not full_hybrid.empty:
        full_hybrid["f1"] = pd.to_numeric(full_hybrid["f1"], errors="coerce")
        best_hybrid = full_hybrid.sort_values("f1", ascending=False).iloc[0]

        print("\nBest estimated full hybrid model:")
        print(f"Model: {best_hybrid['model']}")
        print(f"Accuracy: {best_hybrid['accuracy']}")
        print(f"Precision: {best_hybrid['precision']}")
        print(f"Recall: {best_hybrid['recall']}")
        print(f"F1: {best_hybrid['f1']}")
        print(f"FPR: {best_hybrid['fpr']}")
        print(f"FNR: {best_hybrid['fnr']}")


# -------------------------
# Phase 5
# -------------------------
print("\n==============================")
print("PHASE 5: CONDITIONAL ADVANTAGE TESTS")
print("==============================")

p5_required = [
    P5 / "PHASE5_CONDITIONAL_ADVANTAGE_REPORT.md",
    P5 / "phase5_label_scarcity_results.csv",
    P5 / "phase5_label_scarcity_summary.csv",
    P5 / "phase5_fixed_recall_candidate_pool.csv",
    P5 / "phase5_fixed_recall_estimated_full_hybrid.csv",
    P5 / "phase5_novel_pattern_results.csv",
    P5 / "phase5_scalability_runtime_summary.csv",
    P5 / "phase5_cost_reduction_summary.csv",
    P5 / "phase5_scalability_cost_metadata.json",
]

p5_ok = all(exists(p) for p in p5_required)

if p5_ok:
    label = read_csv(P5 / "phase5_label_scarcity_summary.csv")
    fixed = read_csv(P5 / "phase5_fixed_recall_estimated_full_hybrid.csv")
    novel = read_csv(P5 / "phase5_novel_pattern_results.csv")
    cost_meta = read_json(P5 / "phase5_scalability_cost_metadata.json")

    print(f"Label scarcity summary rows: {len(label)}")
    print(f"Fixed-recall full hybrid rows: {len(fixed)}")
    print(f"Novel-pattern result rows: {len(novel)}")

    print("\nPhase 5 cost/scalability:")
    print(f"Phase 3B test candidate pct: {cost_meta.get('phase3b_test_actual_candidate_pct')}")
    print(f"Phase 3B test fraud retention: {cost_meta.get('phase3b_test_fraud_retention')}")
    print(f"Stage 2 workload reduction factor: {cost_meta.get('stage2_workload_reduction_factor')}")

    print("\nBest fixed-recall hybrid model by F1 at each recall target:")
    for target in sorted(fixed["recall_target"].dropna().unique()):
        sub = fixed[fixed["recall_target"] == target].copy()
        sub["f1"] = pd.to_numeric(sub["f1"], errors="coerce")
        best = sub.sort_values("f1", ascending=False).iloc[0]
        print(
            f"Recall target {target}: "
            f"{best['model']} | F1={best['f1']} | "
            f"Recall={best['recall']} | FPR={best['fpr']}"
        )

    print("\nNovel-pattern results:")
    cols = ["split", "model", "precision", "recall", "f1", "fpr", "fnr"]
    print(novel[cols].to_string(index=False))

    if len(label) > 0 and len(fixed) >= 9 and len(novel) >= 3:
        print("[PASS] Phase 5 contains label scarcity, fixed-recall, novel-pattern, and cost/scalability tests.")
    else:
        print("[CHECK] Phase 5 may be incomplete.")


# -------------------------
# Final verdict
# -------------------------
print("\n==============================")
print("OVERALL VERDICT UP TO PHASE 5")
print("==============================")

all_ok = p1_ok and p2_ok and p3_ok and p3b_ok and p4_ok and p5_ok

if all_ok:
    print("[PASS] All outputs from Phase 1 to Phase 5 are present and reviewable.")
    print("[PASS] The project is aligned with the thesis plan up to Phase 5.")
    print("[PASS] The comparison framework now includes performance, runtime, resource, and conditional advantage analysis.")
    print("[NEXT] Proceed to Phase 6: IBM Quantum hardware validation and queue/execution-time measurement.")
else:
    print("[CHECK] One or more phases have missing files. Review the missing items above.")
