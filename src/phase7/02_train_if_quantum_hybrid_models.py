import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def feature_states(X):
    states = []

    for row in X:
        state = np.array([1.0 + 0j])

        for x in row:
            q = np.array([np.cos(x / 2), np.sin(x / 2)], dtype=complex)
            state = np.kron(state, q)

        states.append(state)

    return np.asarray(states, dtype=complex)


def quantum_kernel(A, B):
    return np.abs(A @ B.conj().T) ** 2


def select_best_f1_threshold(y_true, scores, sample_weight=None):
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))

    best_threshold = thresholds[0]
    best_f1 = -1

    for th in thresholds:
        pred = (scores >= th).astype(int)

        f1 = f1_score(
            y_true,
            pred,
            sample_weight=sample_weight,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    return float(best_threshold)


def metrics_row(track, model, split, y_true, scores, threshold, sample_weight=None):
    pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
        sample_weight=sample_weight,
    )

    tn, fp, fn, tp = cm.ravel()

    try:
        roc = roc_auc_score(y_true, scores, sample_weight=sample_weight)
    except Exception:
        roc = np.nan

    try:
        pr = average_precision_score(y_true, scores, sample_weight=sample_weight)
    except Exception:
        pr = np.nan

    return {
        "track": track,
        "model": model,
        "split": split,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred, sample_weight=sample_weight),
        "precision": precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0),
        "recall": recall_score(y_true, pred, sample_weight=sample_weight, zero_division=0),
        "f1": f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_rows": int(len(y_true)),
    }


def ry(theta):
    return np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )


def rz(theta):
    return np.array(
        [
            [np.exp(-0.5j * theta), 0],
            [0, np.exp(0.5j * theta)],
        ],
        dtype=complex,
    )


def apply_gate_batch(states, gate, qubit, n_qubits):
    reshaped = states.reshape((states.shape[0],) + (2,) * n_qubits)
    axis = 1 + qubit

    moved = np.moveaxis(reshaped, axis, -1)
    moved = moved @ gate.T
    restored = np.moveaxis(moved, -1, axis)

    return restored.reshape(states.shape)


def apply_cnot_batch(states, control, target, n_qubits):
    dim = 2 ** n_qubits
    out = np.zeros_like(states)

    control_bit = n_qubits - 1 - control
    target_bit = n_qubits - 1 - target

    for i in range(dim):
        if (i >> control_bit) & 1:
            j = i ^ (1 << target_bit)
        else:
            j = i

        out[:, j] += states[:, i]

    return out


def vqc_scores(params, input_states, n_qubits, layers):
    states = input_states.copy()
    pos = 0

    for _ in range(layers):
        for q in range(n_qubits):
            states = apply_gate_batch(states, ry(params[pos]), q, n_qubits)
            pos += 1
            states = apply_gate_batch(states, rz(params[pos]), q, n_qubits)
            pos += 1

        for q in range(n_qubits - 1):
            states = apply_cnot_batch(states, q, q + 1, n_qubits)

    dim = 2 ** n_qubits
    measure_bit = n_qubits - 1

    idx = [i for i in range(dim) if (i >> measure_bit) & 1]
    probs = np.sum(np.abs(states[:, idx]) ** 2, axis=1)

    return np.clip(probs.real, 1e-7, 1 - 1e-7)


def train_vqc(train_states, y_train, w_train, n_qubits, layers, maxiter, seed):
    rng = np.random.default_rng(seed)
    n_params = layers * n_qubits * 2
    init = rng.normal(0, 0.1, size=n_params)

    def loss_fn(params):
        scores = vqc_scores(params, train_states, n_qubits, layers)

        loss = -(
            y_train * np.log(scores) +
            (1 - y_train) * np.log(1 - scores)
        )

        return float(np.average(loss, weights=w_train))

    result = minimize(
        loss_fn,
        init,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.25},
    )

    return result.x, float(result.fun), bool(result.success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--out-models", required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_models = Path(args.out_models)

    out_reports.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.dataset, allow_pickle=True)

    X = raw["X"]
    y = raw["y"].astype(int)
    split = raw["split"].astype(str)
    sample_weight = raw["sample_weight"].astype(float)

    n_qubits = X.shape[1]

    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"

    X_train, y_train, w_train = X[train_mask], y[train_mask], sample_weight[train_mask]
    X_valid, y_valid, w_valid = X[valid_mask], y[valid_mask], sample_weight[valid_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], sample_weight[test_mask]

    runtime_rows = []
    metric_rows = []

    t0 = time.perf_counter()
    states_train = feature_states(X_train)
    states_valid = feature_states(X_valid)
    states_test = feature_states(X_test)
    feature_state_time = time.perf_counter() - t0

    # Model 1: Isolation Forest + Quantum Kernel SVC
    model_name = "isolation_forest_plus_quantum_kernel_svc_statevector"

    t0 = time.perf_counter()
    K_train = quantum_kernel(states_train, states_train)
    K_valid = quantum_kernel(states_valid, states_train)
    K_test = quantum_kernel(states_test, states_train)
    kernel_time = time.perf_counter() - t0

    clf = SVC(kernel="precomputed", class_weight="balanced")

    t0 = time.perf_counter()
    clf.fit(K_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    valid_scores = clf.decision_function(K_valid)
    valid_infer_time = time.perf_counter() - t0

    threshold = select_best_f1_threshold(y_valid, valid_scores, w_valid)

    t0 = time.perf_counter()
    test_scores = clf.decision_function(K_test)
    test_infer_time = time.perf_counter() - t0

    metric_rows.append(
        metrics_row(
            "isolation_forest_plus_quantum",
            model_name,
            "test",
            y_test,
            test_scores,
            threshold,
            w_test,
        )
    )

    runtime_rows.append({
        "track": "isolation_forest_plus_quantum",
        "model": model_name,
        "stage1_model": "Isolation Forest",
        "stage2_model": "Quantum Kernel SVC Statevector",
        "n_qubits": n_qubits,
        "n_circuits_or_state_evaluations": int(len(X_train) + len(X_valid) + len(X_test)),
        "shots": np.nan,
        "feature_state_time_seconds": feature_state_time,
        "kernel_matrix_time_seconds": kernel_time,
        "training_time_seconds": train_time,
        "validation_inference_time_seconds": valid_infer_time,
        "test_inference_time_seconds": test_infer_time,
        "total_runtime_seconds": feature_state_time + kernel_time + train_time + valid_infer_time + test_infer_time,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "compute_cost_type": "Isolation Forest candidate-pool + quantum statevector simulation on classical hardware",
    })

    # Model 2: Isolation Forest + VQC
    model_name = "isolation_forest_plus_vqc_statevector"

    t0 = time.perf_counter()
    params, final_loss, success = train_vqc(
        states_train,
        y_train,
        w_train,
        n_qubits=n_qubits,
        layers=args.layers,
        maxiter=args.maxiter,
        seed=args.random_state,
    )
    train_time = time.perf_counter() - t0

    np.savez_compressed(
        out_models / "phase7_if_plus_vqc_params.npz",
        params=params,
        n_qubits=n_qubits,
        layers=args.layers,
    )

    t0 = time.perf_counter()
    valid_scores = vqc_scores(params, states_valid, n_qubits, args.layers)
    valid_infer_time = time.perf_counter() - t0

    threshold = select_best_f1_threshold(y_valid, valid_scores, w_valid)

    t0 = time.perf_counter()
    test_scores = vqc_scores(params, states_test, n_qubits, args.layers)
    test_infer_time = time.perf_counter() - t0

    metric_rows.append(
        metrics_row(
            "isolation_forest_plus_quantum",
            model_name,
            "test",
            y_test,
            test_scores,
            threshold,
            w_test,
        )
    )

    runtime_rows.append({
        "track": "isolation_forest_plus_quantum",
        "model": model_name,
        "stage1_model": "Isolation Forest",
        "stage2_model": "VQC Statevector",
        "n_qubits": n_qubits,
        "n_circuits_or_state_evaluations": int(len(X_train) + len(X_valid) + len(X_test)),
        "shots": np.nan,
        "feature_state_time_seconds": feature_state_time,
        "kernel_matrix_time_seconds": np.nan,
        "training_time_seconds": train_time,
        "validation_inference_time_seconds": valid_infer_time,
        "test_inference_time_seconds": test_infer_time,
        "total_runtime_seconds": feature_state_time + train_time + valid_infer_time + test_infer_time,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "vqc_layers": args.layers,
        "vqc_final_loss": final_loss,
        "vqc_optimizer_success": success,
        "compute_cost_type": "Isolation Forest candidate-pool + VQC statevector simulation on classical hardware",
    })

    pd.DataFrame(metric_rows).to_csv(
        out_reports / "phase7_if_quantum_hybrid_performance_metrics.csv",
        index=False,
    )

    pd.DataFrame(runtime_rows).to_csv(
        out_reports / "phase7_if_quantum_hybrid_runtime_cost.csv",
        index=False,
    )

    save_json(
        {
            "dataset": args.dataset,
            "n_qubits": n_qubits,
            "stage1_model": "Isolation Forest",
            "stage2_models": [
                "Quantum Kernel SVC Statevector",
                "VQC Statevector",
            ],
            "layers": args.layers,
            "maxiter": args.maxiter,
            "note": "Hybrid quantum models are evaluated on an Isolation Forest-ranked candidate subset prepared in Phase 7.",
        },
        out_reports / "phase7_if_quantum_hybrid_metadata.json",
    )

    print("Saved performance metrics:")
    print(pd.DataFrame(metric_rows))

    print("\nSaved runtime/cost metrics:")
    print(pd.DataFrame(runtime_rows))


if __name__ == "__main__":
    main()
