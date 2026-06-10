from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

files = [
    ROOT / "reports" / "phase4" / "phase4_master_runtime_comparison.csv",
    ROOT / "reports" / "phase5" / "phase5_fixed_recall_runtime.csv",
    ROOT / "reports" / "phase5" / "phase5_novel_pattern_runtime.csv",
    ROOT / "reports" / "phase5" / "phase5_scalability_runtime_summary.csv",
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
    raise FileNotFoundError("No runtime report files found.")

df = pd.concat(parts, ignore_index=True)

if "model" not in df.columns:
    raise ValueError("No model column found in runtime files.")

# Remove quantum and hybrid rows
exclude_terms = "quantum|vqc|qsvc|statevector|hardware|phase3b_screener_plus|hybrid"
df = df[
    ~df["model"].astype(str).str.contains(exclude_terms, case=False, na=False)
].copy()

# Standardize model names into main classical families
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

# Runtime-related possible columns
possible_numeric_cols = [
    "training_time_seconds",
    "validation_inference_time_seconds",
    "valid_inference_time_seconds",
    "test_inference_time_seconds",
    "inference_time_seconds",
    "inference_rows_per_second",
    "train_rows",
    "training_rows",
    "valid_rows",
    "test_rows",
    "available_ram_gb",
]

for col in possible_numeric_cols:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Combine validation inference columns
df["validation_inference_time_seconds_clean"] = df["validation_inference_time_seconds"]
df["validation_inference_time_seconds_clean"] = df["validation_inference_time_seconds_clean"].fillna(
    df["valid_inference_time_seconds"]
)

# Combine train row columns
df["train_rows_clean"] = df["train_rows"]
df["train_rows_clean"] = df["train_rows_clean"].fillna(df["training_rows"])

# Estimate total runtime
runtime_components = [
    "training_time_seconds",
    "validation_inference_time_seconds_clean",
    "test_inference_time_seconds",
    "inference_time_seconds",
]

df["total_runtime_seconds"] = df[runtime_components].sum(axis=1, skipna=True)

# If total is 0 because all components are missing, set NaN
df.loc[df["total_runtime_seconds"] == 0, "total_runtime_seconds"] = np.nan

# Keep the fastest available runtime row per model
# For runtime/cost table, lower total runtime = lower computational cost.
df = df.sort_values(
    by=["main_model", "total_runtime_seconds"],
    ascending=[True, True],
    na_position="last"
)

best = df.groupby("main_model", as_index=False).first()

final = pd.DataFrame({
    "model": best["main_model"],
    "training_time_seconds": best["training_time_seconds"],
    "validation_inference_time_seconds": best["validation_inference_time_seconds_clean"],
    "test_inference_time_seconds": best["test_inference_time_seconds"],
    "other_inference_time_seconds": best["inference_time_seconds"],
    "total_runtime_seconds": best["total_runtime_seconds"],
    "inference_rows_per_second": best["inference_rows_per_second"],
    "train_rows": best["train_rows_clean"],
    "test_rows": best["test_rows"],
    "available_ram_gb": best["available_ram_gb"],
    "compute_cost_type": "Classical CPU/GPU runtime cost only; no IBM quantum hardware cost",
    "source_file": best["source_file"],
})

round_cols = [
    "training_time_seconds",
    "validation_inference_time_seconds",
    "test_inference_time_seconds",
    "other_inference_time_seconds",
    "total_runtime_seconds",
    "inference_rows_per_second",
    "available_ram_gb",
]

for col in round_cols:
    final[col] = pd.to_numeric(final[col], errors="coerce").round(6)

for col in ["train_rows", "test_rows"]:
    final[col] = pd.to_numeric(final[col], errors="coerce")

final = final.sort_values("total_runtime_seconds", ascending=True, na_position="last").reset_index(drop=True)

csv_path = OUT / "main_classical_models_runtime_cost_table.csv"
txt_path = OUT / "main_classical_models_runtime_cost_table.txt"

final.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(final.to_string(index=False))

print("Final classical runtime/cost table created.")
print("CSV:", csv_path)
print("TXT:", txt_path)
print()
print(final.to_string(index=False))
