from pathlib import Path
import pandas as pd

ROOT = Path(".")
INFILE = ROOT / "reports" / "final" / "classical_model_performance_table.csv"
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

if not INFILE.exists():
    raise FileNotFoundError(f"Missing input file: {INFILE}")

df = pd.read_csv(INFILE)

# Convert numeric columns
numeric_cols = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "fpr", "fnr", "tn", "fp", "fn", "tp", "threshold", "n_rows"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize model names into main model families
def standard_model_name(name):
    name = str(name).lower()

    if "xgboost" in name or "xgb" in name:
        return "XGBoost"
    if "hist_gradient" in name or "histgradient" in name:
        return "HistGradientBoosting"
    if "random_forest" in name or "random forest" in name:
        return "Random Forest"
    if "logistic" in name:
        return "Logistic Regression"
    if "svm" in name or "svc" in name:
        return "SVM"
    if "mlp" in name or "neural" in name:
        return "MLP"
    if "decision_tree" in name or "decision tree" in name:
        return "Decision Tree"
    if "naive" in name:
        return "Naive Bayes"

    return str(name)

df["main_model"] = df["model"].apply(standard_model_name)

# Keep only main classical model families
main_models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "HistGradientBoosting",
    "SVM",
    "MLP",
    "Decision Tree",
    "Naive Bayes",
]

df = df[df["main_model"].isin(main_models)].copy()

# Remove rows where key metrics are missing
df = df.dropna(subset=["f1"], how="all").copy()

# Best-performance rule:
# 1. Highest F1-score
# 2. Highest PR-AUC
# 3. Highest ROC-AUC
df = df.sort_values(
    by=["main_model", "f1", "pr_auc", "roc_auc"],
    ascending=[True, False, False, False]
)

best = df.groupby("main_model", as_index=False).first()

# Create confusion matrix column
best["confusion_matrix"] = best.apply(
    lambda r: f"TN={int(r['tn'])}, FP={int(r['fp'])}, FN={int(r['fn'])}, TP={int(r['tp'])}",
    axis=1
)

# Final thesis-style columns
final = best.rename(columns={
    "main_model": "model",
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

# Sort best models by F1-score
final = final.sort_values("f1_score", ascending=False).reset_index(drop=True)

csv_path = OUT / "main_classical_models_best_performance_table.csv"
txt_path = OUT / "main_classical_models_best_performance_table.txt"

final.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(final.to_string(index=False))

print("Final main classical models table created.")
print("CSV:", csv_path)
print("TXT:", txt_path)
print()
print(final.to_string(index=False))
