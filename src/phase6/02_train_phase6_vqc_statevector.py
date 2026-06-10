import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import psutil

from phase6_common import (
    save_json,
    write_rows_csv,
    feature_map_states,
    vqc_scores_from_params,
    weighted_bce,
    choose_threshold_for_recall,
    binary_metrics_from_scores,
)


def get_split(data, split_name):
    mask = data["split"] == split_name
    return (
        data["X"][mask],
        data["y"][mask].astype(int),
        data["sample_weight"][mask].astype(float),
    )


def train_vqc(states_train, y_train, w_train, n_qubits, layers, maxiter, random_state):
    rng = np.random.default_rng(random_state)
    init = rng.normal(0, 0.1, size=(layers * n_qubits + 2,))
    init[-2] = 2.0
    init[-1] = 0.0

    def objective(params):
        p = vqc_scores_from_params(states_train, params, n_qubits, layers)
        return weighted_bce(y_train, p, w_train)

    result = minimize(
        objective,
        init,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.25, "disp": False},
    )

    return result.x, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-dataset", required=True)
    parser.add_argument("--out-models", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--recall-target", type=float, default=0.80)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_models = Path(args.out_models)
    out_reports = Path(args.out_reports)

    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.phase6_dataset, allow_pickle=True)

    data = {
        "X": raw["X"],
        "y": raw["y"],
        "sample_weight": raw["sample_weight"],
        "split": raw["split"].astype(str),
    }

    X_train, y_train, w_train = get_split(data, "train")
    X_valid, y_valid, w_valid = get_split(data, "valid")
    X_test, y_test, w_test = get_split(data, "test")

    n_qubits = X_train.shape[1]

    start_feature = time.perf_counter()
    train_states = feature_map_states(X_train)
    valid_states = feature_map_states(X_valid)
    test_states = feature_map_states(X_test)
    feature_time = time.perf_counter() - start_feature

    start_train = time.perf_counter()
    params, result = train_vqc(
        train_states,
        y_train,
        w_train,
        n_qubits=n_qubits,
        layers=args.layers,
        maxiter=args.maxiter,
        random_state=args.random_state,
    )
    train_time = time.perf_counter() - start_train

    valid_scores = vqc_scores_from_params(valid_states, params, n_qubits, args.layers)

    threshold, threshold_policy = choose_threshold_for_recall(
        y_valid,
        valid_scores,
        sample_weight=w_valid,
        recall_target=args.recall_target,
    )

    rows = []

    for split_name, states, y, w in [
        ("train", train_states, y_train, w_train),
        ("valid", valid_states, y_valid, w_valid),
        ("test", test_states, y_test, w_test),
    ]:
        start_pred = time.perf_counter()
        scores = vqc_scores_from_params(states, params, n_qubits, args.layers)
        infer_time = time.perf_counter() - start_pred

        row = binary_metrics_from_scores(
            y,
            scores,
            threshold=threshold,
            sample_weight=w,
            model_name="phase6_vqc_statevector",
            split=split_name,
            track="phase6_statevector_reference",
        )

        row["inference_time_seconds"] = infer_time
        row["threshold_policy"] = threshold_policy
        row["recall_target"] = args.recall_target
        rows.append(row)

    write_rows_csv(rows, out_reports / "phase6_statevector_reference_metrics.csv")

    param_payload = {
        "params": params.tolist(),
        "n_qubits": int(n_qubits),
        "layers": int(args.layers),
        "scale": float(params[-2]),
        "bias": float(params[-1]),
        "threshold": float(threshold),
        "threshold_policy": threshold_policy,
        "recall_target": args.recall_target,
    }

    save_json(param_payload, out_models / "phase6_vqc_params.json")

    metadata = {
        "model": "phase6_vqc_statevector",
        "n_qubits": int(n_qubits),
        "layers": int(args.layers),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "test_rows": int(len(y_test)),
        "feature_state_time_seconds": feature_time,
        "training_time_seconds": train_time,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_final_loss": float(result.fun),
        "trainable_parameters": int(len(params)),
        "estimated_statevector_dimension": int(2 ** n_qubits),
        "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
    }

    save_json(metadata, out_reports / "phase6_statevector_training_metadata.json")

    print("Phase 6 VQC statevector training complete.")
    print(pd.DataFrame(rows))
    print(metadata)


if __name__ == "__main__":
    main()
