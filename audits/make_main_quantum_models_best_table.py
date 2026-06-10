from pathlib import Path
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

files = [
    ROOT / "reports" / "phase4" / "phase4_master_performance_comparison.csv",
    ROOT / "reports" / "phase4" / "phase4_quantum_statevector_metrics.csv",
    ROOT / "reports" / "phase6" / "phase6_hardware_metrics.csv",
    ROOT / "reports" / "phase6" / "phase6_simulator_hardware_metric_comparison.csv",
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
    raise FileNotFoundError("No quantum performance files found.")

df = pd.concat(parts, ignore_index=True)

if "model" not in df.columns:
    raise ValueError("No model column found in the performance files.")

# Keep only quantum rows
quantum_terms = "quantum|vqc|qsvc|statevector|hardware"
df = df[
    df["model"].astype(str).str.contains(quantum_terms, case=False, na=False)
].copy()

# Remove classical or hybrid rows if they appear accidentally
exclude_terms = "logistic|random_forest|random forest|hist_gradient|histgradient|mlp|svm_rbf|phase3b_screener_plus"
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

# Standardize quantum model names
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

df["main_quantum_model"] = df["model"].apply(standard_quantum_name)

# Remove rows without F1
df = df.dropna(subset=["f1"], how="all").copy()

# Best-performance rule:
# 1. Highest F1-score
# 2. Highest PR-AUC
# 3. Highest ROC-AUC
df = df.sort_values(
    by=["main_quantum_model", "f1", "pr_auc", "roc_auc"],
    ascending=[True, False, False, False]
)

best = df.groupby("main_quantum_model", as_index=False).first()

# Create confusion matrix column
best["confusion_matrix"] = best.apply(
    lambda r: f"TN={int(r['tn'])}, FP={int(r['fp'])}, FN={int(r['fn'])}, TP={int(r['tp'])}",
    axis=1
)

final = best.rename(columns={
    "main_quantum_model": "model",
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

final = final.sort_values("f1_score", ascending=False).reset_index(drop=True)

csv_path = OUT / "main_quantum_models_best_performance_table.csv"
txt_path = OUT / "main_quantum_models_best_performance_table.txt"

final.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(final.to_string(index=False))

print("Final main quantum models table created.")
print("CSV:", csv_path)
print("TXT:", txt_path)
print()
print(final.to_string(index=False))
