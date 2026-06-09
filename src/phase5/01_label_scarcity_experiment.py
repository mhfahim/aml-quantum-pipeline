import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import psutil

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from phase5_common import (
    binary_metrics_from_scores,
    choose_threshold_for_recall,
    feature_map_states,
    quantum_kernel,
    write_rows_csv,
    save_json,
)


def get_split(data, split_name):
    mask = data["split"] == split_name
    return (
        data["X"][mask],
        data["y"][mask].astype(int),
        data["sample_weight"][mask].astype(float),
    )


def stratified_fraction_indices(y, fraction, seed):
    rng = np.random.default_rng(seed)
    idx_all = []

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        n = max(2, int(len(idx) * fraction))
        n = min(n, len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        idx_all.append(chosen)

    out = np.concatenate(idx_all)
    rng.shuffle(out)
    return out


def run_standard_model(model_name, model, X_train, y_train, w_train, X_valid, y_valid, w_valid, X_test, y_test, w_test, recall_target):
    start_train = time.perf_counter()

    if model_name == "reduced_mlp":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=w_train)

    train_time = time.perf_counter() - start_train

    valid_scores = model.predict_proba(X_valid)[:, 1]

    threshold, policy = choose_threshold_for_recall(
        y_valid,
        valid_scores,
        sample_weight=w_valid,
        recall_target=recall_target,
    )

    start_test = time.perf_counter()
    test_scores = model.predict_proba(X_test)[:, 1]
    test_time = time.perf_counter() - start_test

    row = binary_metrics_from_scores(
        y_test,
        test_scores,
        threshold=threshold,
        sample_weight=w_test,
        model_name=model_name,
        split="test",
        track="phase5_label_scarcity",
    )

    row["training_time_seconds"] = train_time
    row["test_inference_time_seconds"] = test_time
    row["threshold_policy"] = policy
    row["recall_target"] = recall_target
    return row


def run_quantum_kernel(X_train, y_train, w_train, X_valid, y_valid, w_valid, X_test, y_test, w_test, recall_target):
    start_feature = time.perf_counter()
    train_states = feature_map_states(X_train)
    valid_states = feature_map_states(X_valid)
    test_states = feature_map_states(X_test)
    feature_time = time.perf_counter() - start_feature

    start_kernel = time.perf_counter()
    K_train = quantum_kernel(train_states, train_states)
    K_valid = quantum_kernel(valid_states, train_states)
    K_test = quantum_kernel(test_states, train_states)
    kernel_time = time.perf_counter() - start_kernel

    model = SVC(kernel="precomputed", class_weight="balanced", random_state=42)

    start_train = time.perf_counter()
    model.fit(K_train, y_train, sample_weight=w_train)
    train_time = time.perf_counter() - start_train

    valid_scores = model.decision_function(K_valid)

    threshold, policy = choose_threshold_for_recall(
        y_valid,
        valid_scores,
        sample_weight=w_valid,
        recall_target=recall_target,
    )

    start_test = time.perf_counter()
    test_scores = model.decision_function(K_test)
    test_time = time.perf_counter() - start_test

    row = binary_metrics_from_scores(
        y_test,
        test_scores,
        threshold=threshold,
        sample_weight=w_test,
        model_name="quantum_kernel_svc_statevector",
        split="test",
        track="phase5_label_scarcity",
    )

    row["training_time_seconds"] = train_time
    row["feature_state_time_seconds"] = feature_time
    row["kernel_matrix_time_seconds"] = kernel_time
    row["test_inference_time_seconds"] = test_time
    row["threshold_policy"] = policy
    row["recall_target"] = recall_target
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum-dataset", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--fractions", default="0.10,0.25,0.50,1.00")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--recall-target", type=float, default=0.80)
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_reports.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.quantum_dataset, allow_pickle=True)

    data = {
        "X": raw["X"],
        "y": raw["y"],
        "sample_weight": raw["sample_weight"],
        "split": raw["split"].astype(str),
    }

    X_train_full, y_train_full, w_train_full = get_split(data, "train")
    X_valid, y_valid, w_valid = get_split(data, "valid")
    X_test, y_test, w_test = get_split(data, "test")

    fractions = [float(x.strip()) for x in args.fractions.split(",")]

    rows = []

    for frac in fractions:
        for repeat in range(args.repeats):
            seed = 1000 + repeat
            idx = stratified_fraction_indices(y_train_full, frac, seed)

            X_train = X_train_full[idx]
            y_train = y_train_full[idx]
            w_train = w_train_full[idx]

            models = {
                "reduced_logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1),
                "reduced_svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
                "reduced_mlp": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=250, random_state=seed),
            }

            for model_name, model in models.items():
                print(f"Fraction={frac}, repeat={repeat}, model={model_name}")

                row = run_standard_model(
                    model_name,
                    model,
                    X_train,
                    y_train,
                    w_train,
                    X_valid,
                    y_valid,
                    w_valid,
                    X_test,
                    y_test,
                    w_test,
                    args.recall_target,
                )

                row["train_fraction"] = frac
                row["repeat"] = repeat
                row["train_rows_used"] = int(len(idx))
                rows.append(row)

            print(f"Fraction={frac}, repeat={repeat}, model=quantum_kernel_svc_statevector")

            row = run_quantum_kernel(
                X_train,
                y_train,
                w_train,
                X_valid,
                y_valid,
                w_valid,
                X_test,
                y_test,
                w_test,
                args.recall_target,
            )

            row["train_fraction"] = frac
            row["repeat"] = repeat
            row["train_rows_used"] = int(len(idx))
            rows.append(row)

    out_csv = out_reports / "phase5_label_scarcity_results.csv"
    write_rows_csv(rows, out_csv)

    summary = (
        pd.DataFrame(rows)
        .groupby(["model", "train_fraction"])
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_training_time=("training_time_seconds", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(out_reports / "phase5_label_scarcity_summary.csv", index=False)

    save_json(
        {
            "experiment": "label_scarcity",
            "fractions": fractions,
            "repeats": args.repeats,
            "recall_target": args.recall_target,
            "models": sorted(pd.DataFrame(rows)["model"].unique().tolist()),
            "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
        },
        out_reports / "phase5_label_scarcity_metadata.json",
    )

    print("Label scarcity experiment complete.")
    print(summary)


if __name__ == "__main__":
    main()
