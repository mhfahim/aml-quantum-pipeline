from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "reports" / "supervisor_fixes"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = 42
rng = np.random.default_rng(SEED)

input_files = [
    ROOT / "reports/final/main_classical_models_best_performance_table.csv",
    ROOT / "reports/final/main_quantum_models_best_performance_table.csv",
    ROOT / "reports/final/isolation_forest_quantum_hybrid_performance_table.csv",
    ROOT / "reports/final/isolation_forest_quantum_hybrid_hardware_performance_table.csv",
]

def metric_from_counts(tn, fp, fn, tp):
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return accuracy, precision, recall, f1, fpr, fnr

def bootstrap_ci_from_confusion(tn, fp, fn, tp, n_boot=N_BOOT):
    counts = np.array([tn, fp, fn, tp], dtype=float)
    total = counts.sum()

    if total <= 0:
        return {}

    probs = counts / total
    samples = rng.multinomial(int(round(total)), probs, size=n_boot)

    vals = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "fpr": [],
        "fnr": [],
    }

    for s in samples:
        b_tn, b_fp, b_fn, b_tp = s
        accuracy, precision, recall, f1, fpr, fnr = metric_from_counts(b_tn, b_fp, b_fn, b_tp)

        vals["accuracy"].append(accuracy)
        vals["precision"].append(precision)
        vals["recall"].append(recall)
        vals["f1_score"].append(f1)
        vals["fpr"].append(fpr)
        vals["fnr"].append(fnr)

    out = {}

    for k, arr in vals.items():
        arr = np.asarray(arr, dtype=float)
        out[f"{k}_ci_lower"] = float(np.nanpercentile(arr, 2.5))
        out[f"{k}_ci_upper"] = float(np.nanpercentile(arr, 97.5))
        out[f"{k}_bootstrap_mean"] = float(np.nanmean(arr))
        out[f"{k}_bootstrap_std"] = float(np.nanstd(arr))

    return out

rows = []

for path in input_files:
    if not path.exists():
        print("[SKIP missing]", path)
        continue

    df = pd.read_csv(path)

    required = ["model", "tn", "fp", "fn", "tp"]

    if not all(c in df.columns for c in required):
        print("[SKIP no confusion columns]", path)
        continue

    for _, r in df.iterrows():
        tn = float(r.get("tn", 0))
        fp = float(r.get("fp", 0))
        fn = float(r.get("fn", 0))
        tp = float(r.get("tp", 0))

        ci = bootstrap_ci_from_confusion(tn, fp, fn, tp)

        row = {
            "source_file": str(path),
            "model": r.get("model"),
            "n_rows_reported": r.get("n_rows", np.nan),
            "confusion_total": tn + fp + fn + tp,
            "accuracy_point": r.get("accuracy", np.nan),
            "precision_point": r.get("precision", np.nan),
            "recall_point": r.get("recall", np.nan),
            "f1_score_point": r.get("f1_score", r.get("f1", np.nan)),
            "pr_auc_point": r.get("pr_auc", np.nan),
            "roc_auc_point": r.get("roc_auc", np.nan),
            "fpr_point": r.get("fpr", np.nan),
            "fnr_point": r.get("fnr", np.nan),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "ci_method": "Bootstrap from confusion-matrix counts for threshold metrics. PR-AUC and ROC-AUC CI require score-level prediction files.",
        }

        row.update(ci)
        rows.append(row)

out = pd.DataFrame(rows)

for col in out.columns:
    if any(x in col for x in ["point", "lower", "upper", "mean", "std", "total"]):
        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)

out.to_csv(OUT / "confidence_interval_table.csv", index=False)

with open(OUT / "confidence_interval_table.txt", "w", encoding="utf-8") as f:
    f.write(out.to_string(index=False))

print(out.to_string(index=False))
print("\nCreated reports/supervisor_fixes/confidence_interval_table.csv")
