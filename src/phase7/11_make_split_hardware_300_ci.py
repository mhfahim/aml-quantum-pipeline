import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def metric_dict(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_rows": int(len(y_true)),
    }

    if len(np.unique(y_true)) >= 2:
        out["pr_auc"] = average_precision_score(y_true, scores)
        out["roc_auc"] = roc_auc_score(y_true, scores)
    else:
        out["pr_auc"] = np.nan
        out["roc_auc"] = np.nan

    return out


def bootstrap_ci(y_true, scores, threshold, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        sb = scores[idx]

        if len(np.unique(yb)) < 2:
            continue

        rows.append(metric_dict(yb, sb, threshold))

    boot = pd.DataFrame(rows)
    out = {}

    for metric in [
        "accuracy", "precision", "recall", "f1_score",
        "pr_auc", "roc_auc", "fpr", "fnr"
    ]:
        arr = boot[metric].dropna().to_numpy()

        if len(arr) == 0:
            out[f"{metric}_ci_lower"] = np.nan
            out[f"{metric}_ci_upper"] = np.nan
            out[f"{metric}_bootstrap_mean"] = np.nan
            out[f"{metric}_bootstrap_std"] = np.nan
        else:
            out[f"{metric}_ci_lower"] = np.percentile(arr, 2.5)
            out[f"{metric}_ci_upper"] = np.percentile(arr, 97.5)
            out[f"{metric}_bootstrap_mean"] = np.mean(arr)
            out[f"{metric}_bootstrap_std"] = np.std(arr)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-reports", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    final = Path(args.final_reports)
    scores = pd.read_csv(final / "isolation_forest_quantum_hybrid_hardware_300_split_scores.csv")

    rows = []

    for model, g in scores.groupby("model"):
        y_true = g["label"].astype(int).to_numpy()
        s = g["score"].astype(float).to_numpy()
        threshold = float(g["threshold"].iloc[0])

        point = metric_dict(y_true, s, threshold)
        ci = bootstrap_ci(y_true, s, threshold, args.bootstrap, args.random_state)

        row = {
            "model": model,
            "model_family": "IBM hardware feasibility, split 300-sample test",
            "n_bootstrap": args.bootstrap,
            "ci_basis": "Score-level bootstrap on 300-sample IBM hardware feasibility test.",
            "important_note": "300 samples improve reliability compared with the 10-sample run, but hardware results remain feasibility/resource evidence, not full-scale superiority evidence.",
        }

        row.update(point)
        row.update(ci)
        rows.append(row)

    out = pd.DataFrame(rows)

    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].round(6)

    out.to_csv(final / "hardware_300_split_score_level_bootstrap_ci_table.csv", index=False)

    with open(final / "hardware_300_split_score_level_bootstrap_ci_table.txt", "w", encoding="utf-8") as f:
        f.write(out.to_string(index=False))

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
