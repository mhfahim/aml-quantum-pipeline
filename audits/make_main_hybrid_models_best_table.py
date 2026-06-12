from pathlib import Path
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

files = [
    ROOT / "reports" / "phase4" / "phase4_master_performance_comparison.csv",
    ROOT / "reports" / "phase4" / "phase4_classical_hybrid_estimated_full_metrics.csv",
    ROOT / "reports" / "phase5" / "phase5_fixed_recall_estimated_full_hybrid.csv",
]

parts = []

for path in files:
    if path.exists():
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        parts.append(df)
    else:
        print("[SKIP] Missing:", path)

if not parts:
    raise FileNotFoundError("No hybrid performance files found.")

df = pd.concat(parts, ignore_index=True)

if "model" not in df.columns:
    raise ValueError("No model column found.")

# Keep only hybrid rows
hybrid_terms = "phase3b|screener|hybrid|estimated_full"
df = df[
    df["model"].astype(str).str.contains(hybrid_terms, case=False, na=False) |
    df.get("track", pd.Series([""] * len(df))).astype(str).str.contains(hybrid_terms, case=False, na=False)
].copy()

# Remove pure quantum rows if any appear accidentally
exclude_terms = "quantum|vqc|qsvc|statevector|hardware"
df = df[
    ~df["model"].astype(str).str.contains(exclude_terms, case=False, na=False)
].copy()

# Convert numeric columns
numeric_cols = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "fpr", "fnr", "tn", "fp", "fn", "tp", "threshold", "n_rows"
]

for col in numeric_cols:
    if col not in df.columns:
        df[col] = None
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Standardize hybrid model names
def standard_hybrid_name(name):
    name_lower = str(name).lower()

    if "random_forest" in name_lower or "random forest" in name_lower:
        return "Phase3B Screener + Random Forest"

    if "hist_gradient" in name_lower or "histgradient" in name_lower:
        return "Phase3B Screener + HistGradientBoosting"

    if "logistic" in name_lower:
        return "Phase3B Screener + Logistic Regression"

    if "xgboost" in name_lower or "xgb" in name_lower:
        return "Phase3B Screener + XGBoost"

    return str(name)

df["main_hybrid_model"] = df["model"].apply(standard_hybrid_name)

# Remove rows without F1
df = df.dropna(subset=["f1"], how="all").copy()

# Best-performance rule:
# 1. Highest F1-score
# 2. Highest PR-AUC
# 3. Highest ROC-AUC
df = df.sort_values(
    by=["main_hybrid_model", "f1", "pr_auc", "roc_auc"],
    ascending=[True, False, False, False]
)

best = df.groupby("main_hybrid_model", as_index=False).first()

# Create confusion matrix column
best["confusion_matrix"] = best.apply(
    lambda r: f"TN={int(r['tn'])}, FP={int(r['fp'])}, FN={int(r['fn'])}, TP={int(r['tp'])}",
    axis=1
)

final = best.rename(columns={
    "main_hybrid_model": "model",
    "f1": "f1_score",
})

final_cols = [
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

final = final[final_cols].copy()

# Round for thesis readability
round_cols = [
    "accuracy", "f1_score", "precision", "recall",
    "pr_auc", "roc_auc", "fpr", "fnr", "threshold"
]

for col in round_cols:
    final[col] = final[col].round(6)

# Sort by best F1
final = final.sort_values("f1_score", ascending=False).reset_index(drop=True)

csv_path = OUT / "main_hybrid_models_best_performance_table.csv"
txt_path = OUT / "main_hybrid_models_best_performance_table.txt"

final.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(final.to_string(index=False))

print("Final main hybrid models table created.")
print("CSV:", csv_path)
print("TXT:", txt_path)
print()
print(final.to_string(index=False))
