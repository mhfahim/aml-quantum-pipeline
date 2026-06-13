import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_predictions(df):
    model_col = find_col(df, ["model", "model_name", "classifier"])
    y_col = find_col(df, ["y_true", "label", "target", "true_label", "actual"])
    score_col = find_col(df, ["y_score", "score", "probability", "prob", "pred_score"])
    pred_col = find_col(df, ["y_pred", "pred", "prediction", "pred_label"])
    split_col = find_col(df, ["split", "dataset_split"])
    row_col = find_col(df, ["row_id", "sample_id", "index", "id"])

    if model_col is None:
        raise ValueError("Could not find model column.")
    if y_col is None:
        raise ValueError("Could not find true label column.")

    out = pd.DataFrame()
    out["model"] = df[model_col].astype(str)
    out["y_true"] = df[y_col].astype(int)

    if score_col is not None:
        out["y_score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        out["y_score"] = np.nan

    if pred_col is not None:
        out["y_pred"] = df[pred_col].astype(int)
    else:
        threshold_col = find_col(df, ["threshold"])
        if score_col is None:
            raise ValueError("No prediction or score column found.")
        if threshold_col is not None:
            out["threshold"] = pd.to_numeric(df[threshold_col], errors="coerce")
            out["y_pred"] = (out["y_score"] >= out["threshold"]).astype(int)
        else:
            out["threshold"] = 0.5
            out["y_pred"] = (out["y_score"] >= 0.5).astype(int)

    if split_col is not None:
        out["split"] = df[split_col].astype(str)
        test_mask = out["split"].str.lower().str.contains("test|hardware", regex=True)
        if test_mask.any():
            out = out[test_mask].copy()
    else:
        out["split"] = "test"

    if row_col is not None:
        out["row_id"] = df.loc[out.index, row_col].astype(str).values
    else:
        out["row_id"] = out.groupby("model").cumcount().astype(str)

    out = out.dropna(subset=["y_true", "y_pred"]).copy()
    out["y_true"] = out["y_true"].astype(int)
    out["y_pred"] = out["y_pred"].astype(int)

    return out[["row_id", "split", "model", "y_true", "y_score", "y_pred"]]


def metric_row(model, y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    fdr = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    predicted_positive_rate = (tp + fp) / len(y_true)

    if len(np.unique(y_true)) >= 2 and not np.all(pd.isna(y_score)):
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "model": model,
        "n_rows": len(y_true),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fdr": fdr,
        "fnr": fnr,
        "predicted_positive_rate": predicted_positive_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    }


def compute_metrics(pred):
    rows = []
    for model, g in pred.groupby("model"):
        rows.append(
            metric_row(
                model,
                g["y_true"].to_numpy(),
                g["y_pred"].to_numpy(),
                g["y_score"].to_numpy(),
            )
        )

    out = pd.DataFrame(rows)
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].round(6)
    return out.sort_values("f1_score", ascending=False).reset_index(drop=True)


def mcnemar_pairwise(pred):
    rows = []
    models = sorted(pred["model"].unique())

    for a, b in itertools.combinations(models, 2):
        pa = pred[pred["model"] == a][["row_id", "y_true", "y_pred"]].rename(
            columns={"y_pred": "pred_a"}
        )
        pb = pred[pred["model"] == b][["row_id", "y_true", "y_pred"]].rename(
            columns={"y_pred": "pred_b"}
        )

        merged = pa.merge(pb, on=["row_id", "y_true"], how="inner")
        if len(merged) == 0:
            continue

        correct_a = merged["pred_a"].to_numpy() == merged["y_true"].to_numpy()
        correct_b = merged["pred_b"].to_numpy() == merged["y_true"].to_numpy()

        b_count = int(np.sum(correct_a & ~correct_b))
        c_count = int(np.sum(~correct_a & correct_b))
        discordant = b_count + c_count

        if discordant == 0:
            p_value = 1.0
        else:
            p_value = binomtest(min(b_count, c_count), n=discordant, p=0.5, alternative="two-sided").pvalue

        acc_a = accuracy_score(merged["y_true"], merged["pred_a"])
        acc_b = accuracy_score(merged["y_true"], merged["pred_b"])

        rows.append({
            "model_a": a,
            "model_b": b,
            "n_paired_rows": len(merged),
            "a_correct_b_wrong": b_count,
            "a_wrong_b_correct": c_count,
            "discordant_pairs": discordant,
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "accuracy_difference_a_minus_b": acc_a - acc_b,
            "mcnemar_exact_p_value": p_value,
            "significant_at_0_05": p_value < 0.05,
            "interpretation": "Significant difference in paired error pattern" if p_value < 0.05 else "No statistically significant paired error difference",
        })

    out = pd.DataFrame(rows)
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].round(6)
    return out


def metric_value(y_true, y_pred, y_score, metric):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if metric == "f1_score":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "fpr":
        return fp / (fp + tn) if (fp + tn) > 0 else np.nan
    if metric == "fdr":
        return fp / (tp + fp) if (tp + fp) > 0 else np.nan
    if metric == "roc_auc":
        if len(np.unique(y_true)) < 2 or np.all(pd.isna(y_score)):
            return np.nan
        return roc_auc_score(y_true, y_score)
    if metric == "pr_auc":
        if len(np.unique(y_true)) < 2 or np.all(pd.isna(y_score)):
            return np.nan
        return average_precision_score(y_true, y_score)

    raise ValueError(metric)


def paired_bootstrap_tests(pred, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    models = sorted(pred["model"].unique())
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "fpr", "fdr"]

    for a, b in itertools.combinations(models, 2):
        pa = pred[pred["model"] == a][["row_id", "y_true", "y_pred", "y_score"]].rename(
            columns={"y_pred": "pred_a", "y_score": "score_a"}
        )
        pb = pred[pred["model"] == b][["row_id", "y_true", "y_pred", "y_score"]].rename(
            columns={"y_pred": "pred_b", "y_score": "score_b"}
        )

        merged = pa.merge(pb, on=["row_id", "y_true"], how="inner")
        if len(merged) == 0:
            continue

        n = len(merged)

        for metric in metrics:
            y = merged["y_true"].to_numpy()
            pred_a = merged["pred_a"].to_numpy()
            pred_b = merged["pred_b"].to_numpy()
            score_a = merged["score_a"].to_numpy()
            score_b = merged["score_b"].to_numpy()

            point_a = metric_value(y, pred_a, score_a, metric)
            point_b = metric_value(y, pred_b, score_b, metric)
            point_diff = point_a - point_b

            boot_diffs = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                yb = y[idx]

                if metric in ["roc_auc", "pr_auc"] and len(np.unique(yb)) < 2:
                    continue

                va = metric_value(yb, pred_a[idx], score_a[idx], metric)
                vb = metric_value(yb, pred_b[idx], score_b[idx], metric)

                if not np.isnan(va) and not np.isnan(vb):
                    boot_diffs.append(va - vb)

            boot_diffs = np.array(boot_diffs)

            if len(boot_diffs) == 0:
                ci_low, ci_high, p_value = np.nan, np.nan, np.nan
            else:
                ci_low = np.percentile(boot_diffs, 2.5)
                ci_high = np.percentile(boot_diffs, 97.5)
                p_left = np.mean(boot_diffs <= 0)
                p_right = np.mean(boot_diffs >= 0)
                p_value = 2 * min(p_left, p_right)
                p_value = min(float(p_value), 1.0)

            rows.append({
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "n_paired_rows": n,
                "metric_a": point_a,
                "metric_b": point_b,
                "difference_a_minus_b": point_diff,
                "bootstrap_ci_lower": ci_low,
                "bootstrap_ci_upper": ci_high,
                "bootstrap_p_value_two_sided": p_value,
                "significant_at_0_05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                "interpretation": "Significant metric difference" if (not np.isnan(p_value) and p_value < 0.05) else "No statistically significant metric difference",
            })

    out = pd.DataFrame(rows)
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].round(6)
    return out


def diagnostic_table(metrics):
    out = metrics.copy()

    def explain(row):
        if row["accuracy"] < 0.5 and row["fpr"] > 0.5:
            return "Low accuracy is mainly associated with very high false-positive behavior."
        if row["fpr"] > 0.3:
            return "FPR is high; many normal transactions are wrongly flagged as suspicious."
        if row["fdr"] > 0.5:
            return "FDR is high; many predicted suspicious alerts are false discoveries."
        return "No extreme low-accuracy/high-FPR pattern."

    out["diagnostic_interpretation"] = out.apply(explain, axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.predictions)
    pred = normalize_predictions(raw)
    pred.to_csv(out_dir / "normalized_predictions_for_significance.csv", index=False)

    metrics = compute_metrics(pred)
    metrics.to_csv(out_dir / "metrics_with_fpr_fdr_table.csv", index=False)

    mcnemar = mcnemar_pairwise(pred)
    mcnemar.to_csv(out_dir / "mcnemar_pairwise_significance_tests.csv", index=False)

    boot = paired_bootstrap_tests(pred, n_bootstrap=args.bootstrap, seed=args.random_state)
    boot.to_csv(out_dir / "paired_bootstrap_metric_significance_tests.csv", index=False)

    diag = diagnostic_table(metrics)
    diag.to_csv(out_dir / "low_accuracy_high_fpr_fdr_diagnostic_table.csv", index=False)

    with open(out_dir / "README_statistical_tests.txt", "w", encoding="utf-8") as f:
        f.write(
            "Statistical significance revision outputs.\\n"
            "McNemar exact test compares paired classification disagreement.\\n"
            "Paired bootstrap compares metric differences on the same test samples.\\n"
            "FPR = FP/(FP+TN). FDR = FP/(TP+FP) = 1 - Precision.\\n"
            "Tests are valid only within the same sample/feature/evaluation scope.\\n"
        )

    print("Saved to:", out_dir)
    print("\\nMetrics with FPR/FDR:")
    print(metrics.to_string(index=False))
    print("\\nMcNemar tests:")
    print(mcnemar.head(20).to_string(index=False))
    print("\\nBootstrap tests:")
    print(boot.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
