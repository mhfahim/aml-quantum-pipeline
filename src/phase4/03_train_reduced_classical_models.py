import argparse
from pathlib import Path
import time
import psutil
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from phase4_common import (
    binary_metrics_from_scores,
    choose_threshold_for_recall,
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


def build_models():
    return {
        "reduced_logistic_regression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        ),
        "reduced_svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "reduced_mlp": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=250,
            random_state=42,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum-dataset", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--out-models", required=True)
    parser.add_argument("--recall-target", type=float, default=0.80)
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_models = Path(args.out_models)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.quantum_dataset, allow_pickle=True)
    data = {
        "X": raw["X"],
        "y": raw["y"],
        "sample_weight": raw["sample_weight"],
        "split": raw["split"].astype(str),
    }

    X_train, y_train, w_train = get_split(data, "train")
    X_valid, y_valid, w_valid = get_split(data, "valid")
    X_test, y_test, w_test = get_split(data, "test")

    models = build_models()

    metric_rows = []
    runtime_rows = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}")

        start_train = time.perf_counter()

        if model_name == "reduced_mlp":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)

        train_time = time.perf_counter() - start_train

        start_valid = time.perf_counter()
        valid_scores = model.predict_proba(X_valid)[:, 1]
        valid_inference_time = time.perf_counter() - start_valid

        threshold, threshold_policy = choose_threshold_for_recall(
            y_valid,
            valid_scores,
            sample_weight=w_valid,
            recall_target=args.recall_target,
        )

        for split_name, X, y, w in [
            ("train", X_train, y_train, w_train),
            ("valid", X_valid, y_valid, w_valid),
            ("test", X_test, y_test, w_test),
        ]:
            start_pred = time.perf_counter()
            scores = model.predict_proba(X)[:, 1]
            inference_time = time.perf_counter() - start_pred

            row = binary_metrics_from_scores(
                y,
                scores,
                threshold=threshold,
                sample_weight=w,
                model_name=model_name,
                split=split_name,
                track="reduced_quantum_compatible_classical",
            )
            row["threshold_policy"] = threshold_policy
            row["recall_target"] = args.recall_target
            row["inference_time_seconds"] = inference_time
            row["inference_rows_per_second"] = len(y) / inference_time if inference_time > 0 else None
            metric_rows.append(row)

        model_path = out_models / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        runtime_rows.append({
            "track": "reduced_quantum_compatible_classical",
            "model": model_name,
            "training_time_seconds": train_time,
            "validation_inference_time_seconds": valid_inference_time,
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "model_path": str(model_path),
            "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
        })

    write_rows_csv(metric_rows, out_reports / "phase4_reduced_classical_metrics.csv")
    write_rows_csv(runtime_rows, out_reports / "phase4_reduced_classical_runtime.csv")

    save_json(
        {
            "models": list(models.keys()),
            "recall_target": args.recall_target,
            "note": "These models use the exact same reduced PCA/angle-scaled feature space as the quantum models for fair comparison.",
        },
        out_reports / "phase4_reduced_classical_metadata.json",
    )

    print("Reduced classical training complete.")
    print(pd.DataFrame(metric_rows))


if __name__ == "__main__":
    main()
