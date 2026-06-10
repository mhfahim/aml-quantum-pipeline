from pathlib import Path
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)

phase2_path = ROOT / "reports" / "phase2" / "baseline_metrics.csv"
phase4_path = ROOT / "reports" / "phase4" / "phase4_master_performance_comparison.csv"
phase5_fixed_path = ROOT / "reports" / "phase5" / "phase5_fixed_recall_candidate_pool.csv"
phase5_novel_path = ROOT / "reports" / "phase5" / "phase5_novel_pattern_results.csv"

parts = []

def add_table(path, phase, experiment):
    if not path.exists():
        print(f"[SKIP] Missing: {path}")
        return

    df = pd.read_csv(path)
    df["phase"] = phase
    df["experiment"] = experiment

    if "track" not in df.columns:
        df["track"] = experiment

    if "split" not in df.columns:
        df["split"] = "test"

    parts.append(df)

add_table(phase2_path, "Phase 2", "starter_classical_baseline")
add_table(phase4_path, "Phase 4", "main_classical_model_comparison")
add_table(phase5_fixed_path, "Phase 5", "fixed_recall_classical_candidate_pool")
add_table(phase5_novel_path, "Phase 5", "novel_pattern_classical_test")

if not parts:
    raise FileNotFoundError("No performance files found. Check reports folders.")

df = pd.concat(parts, ignore_index=True)

# Keep only classical models, exclude quantum and hybrid rows
exclude_terms = "quantum|vqc|qsvc|statevector|hardware|phase3b_screener_plus"
df = df[
    ~df["model"].astype(str).str.contains(exclude_terms, case=False, na=False)
].copy()

# Keep test-related rows only
df = df[
    df["split"].astype(str).str.contains("test", case=False, na=False)
].copy()

wanted_cols = [
    "phase",
    "experiment",
    "track",
    "model",
    "split",
    "recall_target",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp",
    "n_rows",
]

for col in wanted_cols:
    if col not in df.columns:
        df[col] = None

table = df[wanted_cols].copy()

numeric_cols = [
    "recall_target",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp",
    "n_rows",
]

for col in numeric_cols:
    table[col] = pd.to_numeric(table[col], errors="coerce")

round_cols = [
    "recall_target",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
]

for col in round_cols:
    table[col] = table[col].round(6)

table = table.sort_values(
    by=["phase", "experiment", "track", "model"],
    ascending=True
).reset_index(drop=True)

csv_path = OUT / "classical_model_performance_table.csv"
txt_path = OUT / "classical_model_performance_table.txt"

table.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(table.to_string(index=False))

print("\nClassical model performance table created successfully.")
print("CSV:", csv_path)
print("TXT:", txt_path)
print("\nPreview:\n")
print(table.to_string(index=False))
