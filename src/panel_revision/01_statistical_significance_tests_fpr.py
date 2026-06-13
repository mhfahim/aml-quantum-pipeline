import argparse
from pathlib import Path
import itertools
import warnings

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

warnings.filterwarnings("ignore")


def find_col(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(f"Could not find required column from candidates: {candidates}. Existing columns: {list(df.columns)}")

    return None


def normalize_prediction_file(path, scope):
    path = Path(path)
    df = pd.read_csv(path)

    model_col = find_col(df, ["model", "Model"])
    label_col = find_col(df, ["label", "y_true", "true_label", "target", "actual"])
    score_col = find_col(df, ["score", "y_score", "probability", "positive_probability", "pred_score"], required=False)
    pred_col = find_col(df, ["pred", "prediction", "y_pred", "predicted_label"], required=False)
    threshold_col = find_col(df, ["threshold", "decision_threshold"], required=False)
    split_col = find_col(df, ["split", "Split"], required=False)

    if split_col is not None:
        split_values = df[split_col].astype(str).str.lower()
        test_mask = (
            split_values.str.contains("test")
            | split_values.str.contains("hardware")
            | split_values.str.contains("subset")
        )
        if test_mask.sum() > 0:
            df = df[test_mask].copy()

    out_rows = []

    for model, g in df.groupby(model_col):
        g = g.copy().reset_index(drop=True)

        y_true = g[label_col].astype(int).to_numpy()

        if score_col is not None:
            scores = pd.to_numeric(g[score_col], errors="coerce").to_numpy()
        elif pred_col is not None:
            scores = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        else:
            raise ValueError(f"No score or prediction column found in {path}")

        if pred_col is not None:
            pred = pd.to_numeric(g[pred_col], errors="coerce").fillna(0).astype(int).to_numpy()
        else:
            if threshold_col is not None:
                threshold = pd.to_numeric(g[threshold_col], errors="coerce").dropna()
                threshold_value = float(threshold.iloc[0]) if len(threshold) else 0.5
            else:
                threshold_value = 0.5
            pred = (scores >= threshold_value).astype(int)

        if threshold_col is not None:
            threshold = pd.to_numeric(g[threshold_col], errors="coerce").dropna()
            threshold_value = float(threshold.iloc[0]) if len(threshold) else np.nan
        else:
            threshold_value = np.nan

        sample_id_col = find_col(g, ["sample_id", "row_id", "index", "test_index"], required=False)
        if sample_id_col is not None:
            sample_ids = g[sample_id_col].astype(str).to_numpy()
        else:
            sample_ids = np.arange(len(g)).astype(str)

        for i in range(len(g)):
            out_rows.append({
                "scope": scope,
                "source_file": str(path),
                "model": str(model),
                "sample_id": sample_ids[i],
                "label": int(y_true[i]),
                "score": float(scores[i]),
                "pred": int(pred[i]),
                "threshold": threshold_value,
            })

    return pd.DataFrame(out_rows)


def safe_metrics(y_true, y_score, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "n_rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
        "actual_positive_rate": float(np.mean(y_true == 1)),
    }

    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    return out


def compute_model_metrics(preds):
    rows = []

    for (scope, model), g in preds.groupby(["scope", "model"]):
        y_true = g["label"].astype(int).to_numpy()
        y_score = g["score"].astype(float).to_numpy()
        y_pred = g["pred"].astype(int).to_numpy()

        row = {
            "scope": scope,
            "model": model,
        }
        row.update(safe_metrics(y_true, y_score, y_pred))
        rows.append(row)

    out = pd.DataFrame(rows)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(6)

    return out.sort_values(["scope", "model"]).reset_index(drop=True)


def mcnemar_tests(preds):
    rows = []

    for scope, scope_df in preds.groupby("scope"):
        pivot_pred = scope_df.pivot_table(index="sample_id", columns="model", values="pred", aggfunc="first")
        pivot_true = scope_df.pivot_table(index="sample_id", columns="model", values="label", aggfunc="first")

        models = list(pivot_pred.columns)

        for a, b in itertools.combinations(models, 2):
            common_idx = pivot_pred[[a, b]].dropna().index
            if len(common_idx) == 0:
                continue

            y_true = pivot_true.loc[common_idx, a].astype(int).to_numpy()
            pred_a = pivot_pred.loc[common_idx, a].astype(int).to_numpy()
            pred_b = pivot_pred.loc[common_idx, b].astype(int).to_numpy()

            correct_a = pred_a == y_true
            correct_b = pred_b == y_true

            b_count = int(np.sum(correct_a & ~correct_b))
            c_count = int(np.sum(~correct_a & correct_b))
            discordant = b_count + c_count

            if discordant == 0:
                p_value = 1.0
            else:
                p_value = float(binomtest(min(b_count, c_count), n=discordant, p=0.5, alternative="two-sided").pvalue)

            rows.append({
                "scope": scope,
                "model_a": a,
                "model_b": b,
                "n_common_samples": int(len(common_idx)),
                "model_a_correct_model_b_wrong": b_count,
                "model_a_wrong_model_b_correct": c_count,
                "discordant_pairs": discordant,
                "mcnemar_exact_p_value": p_value,
                "significant_at_0_05": bool(p_value < 0.05),
                "interpretation": "Significant difference in paired error pattern" if p_value < 0.05 else "No significant paired error-pattern difference at 0.05",
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(6)

    return out


def metric_value(metric, y_true, y_score, y_pred):
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if metric == "f1_score":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "fpr":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else np.nan
    if metric == "roc_auc":
        return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else np.nan
    if metric == "pr_auc":
        return average_precision_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else np.nan

    raise ValueError(metric)


def paired_bootstrap_tests(preds, n_bootstrap=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    rows = []
    metrics = ["accuracy", "precision", "recall", "f1_score", "fpr", "roc_auc", "pr_auc"]

    for scope, scope_df in preds.groupby("scope"):
        pivot_pred = scope_df.pivot_table(index="sample_id", columns="model", values="pred", aggfunc="first")
        pivot_score = scope_df.pivot_table(index="sample_id", columns="model", values="score", aggfunc="first")
        pivot_true = scope_df.pivot_table(index="sample_id", columns="model", values="label", aggfunc="first")

        models = list(pivot_pred.columns)

        for a, b in itertools.combinations(models, 2):
            common_idx = pivot_pred[[a, b]].dropna().index
            if len(common_idx) < 10:
                continue

            y_true = pivot_true.loc[common_idx, a].astype(int).to_numpy()
            pred_a = pivot_pred.loc[common_idx, a].astype(int).to_numpy()
            pred_b = pivot_pred.loc[common_idx, b].astype(int).to_numpy()
            score_a = pivot_score.loc[common_idx, a].astype(float).to_numpy()
            score_b = pivot_score.loc[common_idx, b].astype(float).to_numpy()

            n = len(common_idx)

            for metric in metrics:
                point_a = metric_value(metric, y_true, score_a, pred_a)
                point_b = metric_value(metric, y_true, score_b, pred_b)
                point_diff = point_a - point_b

                boot_diffs = []

                for _ in range(n_bootstrap):
                    idx = rng.integers(0, n, size=n)
                    yt = y_true[idx]

                    if metric in ["roc_auc", "pr_auc"] and len(np.unique(yt)) < 2:
                        continue

                    va = metric_value(metric, yt, score_a[idx], pred_a[idx])
                    vb = metric_value(metric, yt, score_b[idx], pred_b[idx])

                    if not np.isnan(va) and not np.isnan(vb):
                        boot_diffs.append(va - vb)

                if len(boot_diffs) == 0:
                    continue

                boot_diffs = np.array(boot_diffs)
                ci_low = np.percentile(boot_diffs, 2.5)
                ci_high = np.percentile(boot_diffs, 97.5)

                p_left = np.mean(boot_diffs <= 0)
                p_right = np.mean(boot_diffs >= 0)
                p_value = min(1.0, 2 * min(p_left, p_right))

                rows.append({
                    "scope": scope,
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "n_common_samples": int(n),
                    "model_a_value": point_a,
                    "model_b_value": point_b,
                    "difference_a_minus_b": point_diff,
                    "bootstrap_ci_lower": ci_low,
                    "bootstrap_ci_upper": ci_high,
                    "paired_bootstrap_p_value": p_value,
                    "significant_at_0_05": bool(p_value < 0.05),
                    "interpretation": "Significant paired metric difference" if p_value < 0.05 else "No significant paired metric difference at 0.05",
                })

    out = pd.DataFrame(rows)

    if not out.empty:
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].round(6)

    return out


def low_accuracy_high_fpr_analysis(metrics):
    rows = []

    for _, r in metrics.iterrows():
        fpr = float(r["fpr"])
        acc = float(r["accuracy"])

        if acc < 0.5 and fpr >= 0.5:
            flag = "Low accuracy with high FPR"
            explanation = "Model is flagging many normal samples as suspicious, creating many false positives."
        elif fpr >= 0.5:
            flag = "High FPR"
            explanation = "Model has high false-positive behavior even if other metrics may look acceptable."
        elif acc < 0.5:
            flag = "Low accuracy"
            explanation = "Model has weak threshold-level correctness under this evaluation setting."
        else:
            flag = "Not extreme"
            explanation = "Accuracy and FPR are not both extreme under this scope."

        rows.append({
            "scope": r["scope"],
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"],
            "fpr": r["fpr"],
            "fnr": r["fnr"],
            "tn": r["tn"],
            "fp": r["fp"],
            "fn": r["fn"],
            "tp": r["tp"],
            "predicted_positive_rate": r["predicted_positive_rate"],
            "actual_positive_rate": r["actual_positive_rate"],
            "issue_flag": flag,
            "explanation": explanation,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="reports/panel_revision")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.repo_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = []

    reduced_pred = root / "reports" / "supervisor_fixes" / "score_level_predictions_reduced_subset.csv"
    if reduced_pred.exists():
        sources.append((reduced_pred, "same_reduced_four_feature_subset"))

    hardware_pred = root / "reports" / "final_hardware_300_split" / "isolation_forest_quantum_hybrid_hardware_300_split_scores.csv"
    if hardware_pred.exists():
        sources.append((hardware_pred, "ibm_hardware_300_sample_feasibility"))

    sota_pred = root / "reports" / "panel_revision" / "sota_transformer_predictions_reduced_subset.csv"
    if sota_pred.exists():
        sources.append((sota_pred, "same_reduced_four_feature_subset"))

    if not sources:
        raise FileNotFoundError(
            "No prediction-level files found. Expected at least reports/supervisor_fixes/score_level_predictions_reduced_subset.csv"
        )

    all_preds = []

    for path, scope in sources:
        print(f"Loading {path} as scope={scope}")
        all_preds.append(normalize_prediction_file(path, scope))

    preds = pd.concat(all_preds, ignore_index=True)

    preds.to_csv(out_dir / "panel_revision_all_normalized_predictions.csv", index=False)

    metrics = compute_model_metrics(preds)
    metrics.to_csv(out_dir / "panel_revision_model_metrics_with_fpr.csv", index=False)

    mcnemar = mcnemar_tests(preds)
    mcnemar.to_csv(out_dir / "panel_revision_mcnemar_tests.csv", index=False)

    bootstrap = paired_bootstrap_tests(
        preds,
        n_bootstrap=args.bootstrap,
        random_state=args.random_state,
    )
    bootstrap.to_csv(out_dir / "panel_revision_paired_bootstrap_tests.csv", index=False)

    error_analysis = low_accuracy_high_fpr_analysis(metrics)
    error_analysis.to_csv(out_dir / "panel_revision_low_accuracy_high_fpr_analysis.csv", index=False)

    with open(out_dir / "panel_revision_statistical_test_summary.txt", "w", encoding="utf-8") as f:
        f.write("Panel Revision Statistical Significance Summary\n")
        f.write("================================================\n\n")
        f.write("Tests included:\n")
        f.write("1. McNemar exact test for paired threshold-level prediction disagreement.\n")
        f.write("2. Paired bootstrap tests for Accuracy, Precision, Recall, F1-score, FPR, ROC-AUC, and PR-AUC.\n")
        f.write("3. Low Accuracy / High FPR diagnostic table based on confusion matrix results.\n\n")
        f.write("Important note: Pairwise tests are performed only inside the same evaluation scope.\n")
        f.write("No cross-scope comparison is made between full-scale, reduced-subset, and hardware settings.\n")

    print("\nSaved:")
    print(out_dir / "panel_revision_all_normalized_predictions.csv")
    print(out_dir / "panel_revision_model_metrics_with_fpr.csv")
    print(out_dir / "panel_revision_mcnemar_tests.csv")
    print(out_dir / "panel_revision_paired_bootstrap_tests.csv")
    print(out_dir / "panel_revision_low_accuracy_high_fpr_analysis.csv")
    print(out_dir / "panel_revision_statistical_test_summary.txt")

    print("\nModel metrics:")
    print(metrics.to_string(index=False))

    print("\nLow accuracy / high FPR analysis:")
    print(error_analysis[["scope", "model", "accuracy", "fpr", "issue_flag", "explanation"]].to_string(index=False))


if __name__ == "__main__":
    main()
