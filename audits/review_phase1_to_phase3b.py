import json
import csv
from pathlib import Path

ROOT = Path(".")
P1 = ROOT / "reports" / "phase1"
P2 = ROOT / "reports" / "phase2"
P3 = ROOT / "reports" / "phase3"
P3B = ROOT / "reports" / "phase3b"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_exists(path):
    if path.exists():
        print(f"[OK] EXISTS: {path}")
        return True
    else:
        print(f"[MISSING] {path}")
        return False


def get_metric(rows, split, target_pct):
    for row in rows:
        try:
            if row["split"] == split and round(float(row["target_candidate_pct"]), 2) == round(float(target_pct), 2):
                return row
        except Exception:
            pass
    return None


print("\n==============================")
print("PHASE 1 CHECK")
print("==============================")

phase1_required = [
    "phase1_profile.json",
    "conversion_log.json",
    "temporal_split_manifest.json",
    "PHASE1_DATA_AUDIT.md",
]

p1_ok = all(check_exists(P1 / f) for f in phase1_required)

if p1_ok:
    profile = read_json(P1 / "phase1_profile.json")
    conversion = read_json(P1 / "conversion_log.json")
    split = read_json(P1 / "temporal_split_manifest.json")

    trans_success = [
        row for row in conversion
        if "_Trans.csv" in row.get("file", "") and row.get("status") == "success"
    ]

    total_rows_profile = int(profile["total_rows"])
    total_rows_split = sum(int(row["n"]) for row in split["split_counts"])

    print(f"Total rows profile: {total_rows_profile:,}")
    print(f"Total rows split: {total_rows_split:,}")
    print(f"Successful transaction files: {len(trans_success)} / 6")
    print(f"Label counts: {profile['label_counts']}")
    print(f"Time range: {profile['time_range']}")

    if total_rows_profile == total_rows_split and len(trans_success) == 6:
        print("[OK] Phase 1 is valid.")
    else:
        print("[PROBLEM] Phase 1 has row/file mismatch.")


print("\n==============================")
print("PHASE 2 CHECK")
print("==============================")

phase2_required = [
    "leakage_audit.json",
    "feature_schema.json",
    "baseline_metrics.csv",
    "baseline_report.md",
]

p2_ok = all(check_exists(P2 / f) for f in phase2_required)

if p2_ok:
    leakage = read_json(P2 / "leakage_audit.json")
    metrics = read_csv_rows(P2 / "baseline_metrics.csv")

    print(f"Leakage audit status: {leakage.get('status')}")
    print(f"Baseline metric rows: {len(metrics)}")

    if "completed" in leakage.get("status", "") and len(metrics) > 0:
        print("[OK] Phase 2 is valid as starter leakage-safe baseline.")
    else:
        print("[PROBLEM] Phase 2 needs review.")


print("\n==============================")
print("PHASE 3 CHECK")
print("==============================")

phase3_required = [
    "stage1_model_metadata.json",
    "stage1_thresholds.json",
    "stage1_screening_metrics.csv",
    "candidate_pool_sample_manifest.json",
    "PHASE3_STAGE1_REPORT.md",
]

p3_ok = all(check_exists(P3 / f) for f in phase3_required)

phase3_test_10 = None

if p3_ok:
    screening3 = read_csv_rows(P3 / "stage1_screening_metrics.csv")
    manifest3 = read_json(P3 / "candidate_pool_sample_manifest.json")

    phase3_test_10 = get_metric(screening3, "test", 0.10)

    print(f"Phase 3 rows scanned: {manifest3.get('seen_rows')}")

    if phase3_test_10:
        print("Phase 3 test @ 10%:")
        print(f"  actual_candidate_pct = {phase3_test_10['actual_candidate_pct']}")
        print(f"  fraud_retention = {phase3_test_10['fraud_retention']}")
        print(f"  enrichment_factor = {phase3_test_10['enrichment_factor']}")

    print("[OK] Phase 3 is technically complete as unsupervised baseline.")


print("\n==============================")
print("PHASE 3B CHECK")
print("==============================")

phase3b_required = [
    "stage1b_model_metadata.json",
    "stage1b_thresholds.json",
    "stage1b_screening_metrics.csv",
    "stage1b_screening_metrics.json",
    "stage1b_candidate_pool_sample_counts.csv",
    "stage1b_candidate_pool_sample_manifest.json",
    "PHASE3B_HIGH_RECALL_REPORT.md",
]

p3b_ok = all(check_exists(P3B / f) for f in phase3b_required)

best_valid = None
best_test_matching = None

if p3b_ok:
    model3b = read_json(P3B / "stage1b_model_metadata.json")
    thresholds3b = read_json(P3B / "stage1b_thresholds.json")
    screening3b = read_csv_rows(P3B / "stage1b_screening_metrics.csv")
    manifest3b = read_json(P3B / "stage1b_candidate_pool_sample_manifest.json")

    print(f"Phase 3B model type: {model3b.get('model_type')}")
    print(f"Phase 3B training rows: {model3b.get('total_training_rows')}")
    print(f"Validation ROC-AUC: {thresholds3b.get('validation_roc_auc')}")
    print(f"Validation PR-AUC: {thresholds3b.get('validation_pr_auc')}")
    print(f"Selected candidate target: {manifest3b.get('selected_target_candidate_pct')}")
    print(f"Rows scanned for candidate pool: {manifest3b.get('seen_rows')}")
    print(f"Candidate pool kept rows: {manifest3b.get('kept_rows')}")

    valid_rows = [r for r in screening3b if r["split"] == "valid"]
    valid_rows_sorted = sorted(valid_rows, key=lambda r: float(r["target_candidate_pct"]))

    passing = [r for r in valid_rows_sorted if float(r["fraud_retention"]) >= 0.80]

    if passing:
        best_valid = passing[0]
    elif valid_rows_sorted:
        best_valid = max(valid_rows_sorted, key=lambda r: float(r["fraud_retention"]))

    if best_valid:
        target = float(best_valid["target_candidate_pct"])
        best_test_matching = get_metric(screening3b, "test", target)

        print("\nBest/selected Phase 3B validation threshold:")
        print(f"  target_candidate_pct = {best_valid['target_candidate_pct']}")
        print(f"  valid_actual_candidate_pct = {best_valid['actual_candidate_pct']}")
        print(f"  valid_fraud_retention = {best_valid['fraud_retention']}")

        if best_test_matching:
            print("\nMatching Phase 3B test result:")
            print(f"  test_actual_candidate_pct = {best_test_matching['actual_candidate_pct']}")
            print(f"  test_fraud_retention = {best_test_matching['fraud_retention']}")
            print(f"  test_enrichment_factor = {best_test_matching['enrichment_factor']}")

    if manifest3b.get("seen_rows") == read_json(P1 / "phase1_profile.json")["total_rows"]:
        print("[OK] Phase 3B candidate-pool creation scanned full dataset.")
    else:
        print("[WARNING] Phase 3B seen_rows does not match Phase 1 total rows.")


print("\n==============================")
print("PHASE 3 VS PHASE 3B DECISION")
print("==============================")

if phase3_test_10 and best_test_matching:
    p3_recall = float(phase3_test_10["fraud_retention"])
    p3b_recall = float(best_test_matching["fraud_retention"])

    print(f"Phase 3 test fraud retention @ 10%: {p3_recall:.6f}")
    print(f"Phase 3B selected test fraud retention: {p3b_recall:.6f}")

    if p3b_recall > p3_recall:
        print("[OK] Phase 3B improves fraud retention over Phase 3.")
    else:
        print("[PROBLEM] Phase 3B does not improve fraud retention over Phase 3.")

    if p3b_recall >= 0.80:
        print("[OK] Phase 3B is strong enough as high-recall Stage 1.")
    elif p3b_recall >= 0.50:
        print("[WARNING] Phase 3B is moderate. It may still need richer behavioral features.")
    else:
        print("[PROBLEM] Phase 3B is still weak for high-recall AML screening.")

print("\n==============================")
print("OVERALL VERDICT")
print("==============================")

if p1_ok and p2_ok and p3_ok and p3b_ok:
    print("[OK] Phases 1, 2, 3, and 3B are present and reviewable.")
    print("[NEXT] If Phase 3B recall is strong, proceed to Phase 4.")
    print("[NEXT] If Phase 3B recall is weak/moderate, improve Stage 1 features before serious quantum comparison.")
else:
    print("[PROBLEM] One or more phase outputs are missing.")
