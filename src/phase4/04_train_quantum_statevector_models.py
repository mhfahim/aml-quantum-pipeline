import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import psutil

from sklearn.svm import SVC
from scipy.optimize import minimize

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


def feature_map_states(X):
    X = np.asarray(X, dtype=float)
    n_samples, n_qubits = X.shape
    dim = 2 ** n_qubits

    states = np.zeros((n_samples, dim), dtype=np.complex128)

    for s in range(n_samples):
        x = X[s]

        for basis in range(dim):
            amp = 1.0 + 0j
            z_values = []

            for q in range(n_qubits):
                bit = (basis >> q) & 1

                if bit == 0:
                    amp *= np.cos(x[q] / 2.0)
                    z_values.append(1.0)
                else:
                    amp *= np.sin(x[q] / 2.0)
                    z_values.append(-1.0)

            phase = 0.0
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    phase += 0.25 * x[i] * x[j] * z_values[i] * z_values[j]

            states[s, basis] = amp * np.exp(1j * phase)

        norm = np.linalg.norm(states[s])
        if norm > 0:
            states[s] /= norm

    return states


def quantum_kernel(A, B):
    return np.abs(A @ B.conj().T) ** 2


def apply_ry(state, theta, qubit, n_qubits):
    state = state.copy()
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)

    dim = len(state)
    step = 2 ** qubit

    for i in range(0, dim, 2 * step):
        for j in range(step):
            idx0 = i + j
            idx1 = idx0 + step

            a0 = state[idx0]
            a1 = state[idx1]

            state[idx0] = c * a0 - s * a1
            state[idx1] = s * a0 + c * a1

    return state


def apply_cnot(state, control, target, n_qubits):
    state = state.copy()
    dim = len(state)

    for i in range(dim):
        c_bit = (i >> control) & 1
        t_bit = (i >> target) & 1

        if c_bit == 1 and t_bit == 0:
            j = i | (1 << target)
            state[i], state[j] = state[j], state[i]

    return state


def expectation_z0(state):
    exp = 0.0
    for i, amp in enumerate(state):
        bit = i & 1
        z = 1.0 if bit == 0 else -1.0
        exp += z * (np.abs(amp) ** 2)
    return float(exp)


def vqc_scores_from_params(states, params, n_qubits, layers):
    circuit_params = params[: layers * n_qubits]
    scale = params[-2]
    bias = params[-1]

    scores = []

    for base_state in states:
        state = base_state.copy()
        k = 0

        for _ in range(layers):
            for q in range(n_qubits):
                state = apply_ry(state, circuit_params[k], q, n_qubits)
                k += 1

            for q in range(n_qubits - 1):
                state = apply_cnot(state, q, q + 1, n_qubits)

        exp_z = expectation_z0(state)
        logit = scale * exp_z + bias
        prob = 1.0 / (1.0 + np.exp(-logit))
        scores.append(prob)

    return np.asarray(scores)


def weighted_bce(y, p, w):
    eps = 1e-8
    p = np.clip(p, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(loss, weights=w))


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
    parser.add_argument("--quantum-dataset", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--recall-target", type=float, default=0.80)
    parser.add_argument("--vqc-train-per-class", type=int, default=300)
    parser.add_argument("--vqc-layers", type=int, default=2)
    parser.add_argument("--vqc-maxiter", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
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

    X_train, y_train, w_train = get_split(data, "train")
    X_valid, y_valid, w_valid = get_split(data, "valid")
    X_test, y_test, w_test = get_split(data, "test")

    n_qubits = X_train.shape[1]

    metric_rows = []
    runtime_rows = []
    resource_rows = []

    print("\nRunning quantum kernel SVC")

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

    qsvc = SVC(
        kernel="precomputed",
        class_weight="balanced",
        probability=False,
        random_state=args.random_state,
    )

    start_train = time.perf_counter()
    qsvc.fit(K_train, y_train, sample_weight=w_train)
    train_time = time.perf_counter() - start_train

    valid_scores = qsvc.decision_function(K_valid)

    threshold, threshold_policy = choose_threshold_for_recall(
        y_valid,
        valid_scores,
        sample_weight=w_valid,
        recall_target=args.recall_target,
    )

    for split_name, K, y, w in [
        ("train", K_train, y_train, w_train),
        ("valid", K_valid, y_valid, w_valid),
        ("test", K_test, y_test, w_test),
    ]:
        start_pred = time.perf_counter()
        scores = qsvc.decision_function(K)
        infer_time = time.perf_counter() - start_pred

        row = binary_metrics_from_scores(
            y,
            scores,
            threshold=threshold,
            sample_weight=w,
            model_name="quantum_kernel_svc_statevector",
            split=split_name,
            track="quantum_statevector_reduced_features",
        )
        row["threshold_policy"] = threshold_policy
        row["recall_target"] = args.recall_target
        row["inference_time_seconds"] = infer_time
        row["inference_rows_per_second"] = len(y) / infer_time if infer_time > 0 else None
        metric_rows.append(row)

    runtime_rows.append({
        "track": "quantum_statevector_reduced_features",
        "model": "quantum_kernel_svc_statevector",
        "feature_state_time_seconds": feature_time,
        "kernel_matrix_time_seconds": kernel_time,
        "training_time_seconds": train_time,
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "test_rows": int(len(y_test)),
        "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
    })

    resource_rows.append({
        "model": "quantum_kernel_svc_statevector",
        "n_qubits": int(n_qubits),
        "statevector_dimension": int(2 ** n_qubits),
        "feature_map": "Ry angle encoding + ZZ-style phase feature map",
        "estimated_feature_map_depth": int(2 + n_qubits - 1),
        "shots": "statevector_exact",
        "hardware_queue_time_seconds": "Phase 6 only",
        "hardware_execution_time_seconds": "Phase 6 only",
    })

    print("\nRunning VQC statevector")

    rng = np.random.default_rng(args.random_state)
    train_idx_pos = np.where(y_train == 1)[0]
    train_idx_neg = np.where(y_train == 0)[0]

    train_idx_pos = rng.choice(train_idx_pos, size=min(args.vqc_train_per_class, len(train_idx_pos)), replace=False)
    train_idx_neg = rng.choice(train_idx_neg, size=min(args.vqc_train_per_class, len(train_idx_neg)), replace=False)

    vqc_idx = np.concatenate([train_idx_pos, train_idx_neg])
    rng.shuffle(vqc_idx)

    states_vqc_train = train_states[vqc_idx]
    y_vqc_train = y_train[vqc_idx]
    w_vqc_train = w_train[vqc_idx]

    start_train = time.perf_counter()
    params, result = train_vqc(
        states_vqc_train,
        y_vqc_train,
        w_vqc_train,
        n_qubits=n_qubits,
        layers=args.vqc_layers,
        maxiter=args.vqc_maxiter,
        random_state=args.random_state,
    )
    vqc_train_time = time.perf_counter() - start_train

    valid_scores = vqc_scores_from_params(valid_states, params, n_qubits, args.vqc_layers)

    threshold, threshold_policy = choose_threshold_for_recall(
        y_valid,
        valid_scores,
        sample_weight=w_valid,
        recall_target=args.recall_target,
    )

    for split_name, states, y, w in [
        ("train", train_states, y_train, w_train),
        ("valid", valid_states, y_valid, w_valid),
        ("test", test_states, y_test, w_test),
    ]:
        start_pred = time.perf_counter()
        scores = vqc_scores_from_params(states, params, n_qubits, args.vqc_layers)
        infer_time = time.perf_counter() - start_pred

        row = binary_metrics_from_scores(
            y,
            scores,
            threshold=threshold,
            sample_weight=w,
            model_name="variational_quantum_classifier_statevector",
            split=split_name,
            track="quantum_statevector_reduced_features",
        )
        row["threshold_policy"] = threshold_policy
        row["recall_target"] = args.recall_target
        row["inference_time_seconds"] = infer_time
        row["inference_rows_per_second"] = len(y) / infer_time if infer_time > 0 else None
        metric_rows.append(row)

    runtime_rows.append({
        "track": "quantum_statevector_reduced_features",
        "model": "variational_quantum_classifier_statevector",
        "feature_state_time_seconds": feature_time,
        "training_time_seconds": vqc_train_time,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_final_loss": float(result.fun),
        "vqc_train_rows_used": int(len(vqc_idx)),
        "train_rows_scored": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "test_rows": int(len(y_test)),
        "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
    })

    resource_rows.append({
        "model": "variational_quantum_classifier_statevector",
        "n_qubits": int(n_qubits),
        "statevector_dimension": int(2 ** n_qubits),
        "feature_map": "Ry angle encoding + ZZ-style phase feature map",
        "ansatz": f"Ry layers={args.vqc_layers} with CNOT chain",
        "trainable_parameters": int(args.vqc_layers * n_qubits + 2),
        "estimated_circuit_depth": int((2 + n_qubits - 1) + args.vqc_layers * (n_qubits + n_qubits - 1)),
        "shots": "statevector_exact",
        "hardware_queue_time_seconds": "Phase 6 only",
        "hardware_execution_time_seconds": "Phase 6 only",
    })

    write_rows_csv(metric_rows, out_reports / "phase4_quantum_statevector_metrics.csv")
    write_rows_csv(runtime_rows, out_reports / "phase4_quantum_statevector_runtime.csv")
    write_rows_csv(resource_rows, out_reports / "phase4_quantum_resource_estimates.csv")

    save_json(
        {
            "n_qubits": int(n_qubits),
            "recall_target": args.recall_target,
            "models": [
                "quantum_kernel_svc_statevector",
                "variational_quantum_classifier_statevector",
            ],
            "note": "These are statevector simulator quantum models on the reduced quantum-compatible feature set. IBM hardware queue/execution metrics are reserved for Phase 6.",
        },
        out_reports / "phase4_quantum_metadata.json",
    )

    print("Quantum statevector models complete.")
    print(pd.DataFrame(metric_rows))
    print(pd.DataFrame(runtime_rows))
    print(pd.DataFrame(resource_rows))


if __name__ == "__main__":
    main()
