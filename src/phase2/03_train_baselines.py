import argparse
from pathlib import Path
import json
import numpy as np
import polars as pl

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_model(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    model.fit(X_train, y_train)

    rows = []

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[:, 1]
        else:
            score = pred

        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        rows.append({
            "model": name,
            "split": split_name,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else None,
            "pr_auc": average_precision_score(y, score) if len(np.unique(y)) > 1 else None,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sample_path = Path(args.sample)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    features = manifest["safe_features"]

    df = pl.read_parquet(sample_path)

    # Keep only rows with split labels.
    df = df.filter(pl.col("_split").is_in(["train", "valid", "test"]))

    train = df.filter(pl.col("_split") == "train")
    valid = df.filter(pl.col("_split") == "valid")
    test = df.filter(pl.col("_split") == "test")

    X_train = train.select(features).to_numpy()
    y_train = train.select("_label").to_numpy().ravel()

    X_valid = valid.select(features).to_numpy()
    y_valid = valid.select("_label").to_numpy().ravel()

    X_test = test.select(features).to_numpy()
    y_test = test.select("_label").to_numpy().ravel()

    print("Train shape:", X_train.shape)
    print("Valid shape:", X_valid.shape)
    print("Test shape:", X_test.shape)

    models = {
        "logistic_regression_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=42,
            )),
        ]),
    }

    all_rows = []

    for name, model in models.items():
        print(f"Training {name}...")
        rows = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
        )
        all_rows.extend(rows)

    metrics_df = pl.DataFrame(all_rows)
    metrics_df.write_csv(out_dir / "baseline_metrics.csv")

    md = []
    md.append("# Phase 2 Classical Baseline Report\n")
    md.append("## Important Note\n")
    md.append("These are starter leakage-safe baselines trained on a sampled dataset: all positives plus a deterministic sample of negatives. They are not the final full-scale production baselines.\n")

    md.append("## Features Used\n")
    for f in features:
        md.append(f"- `{f}`")

    md.append("\n## Metrics\n")
    for row in all_rows:
        md.append(
            f"- {row['model']} | {row['split']} | "
            f"F1={row['f1']:.4f}, ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
            f"Precision={row['precision']:.4f}, Recall={row['recall']:.4f}"
        )

    (out_dir / "baseline_report.md").write_text("\n".join(md), encoding="utf-8")

    print(metrics_df)
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
