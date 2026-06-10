from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_rows_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
        prob = sigmoid(scale * exp_z + bias)
        scores.append(prob)

    return np.asarray(scores)


def weighted_bce(y, p, w):
    eps = 1e-8
    p = np.clip(p, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(loss, weights=w))


def binary_metrics_from_scores(
    y_true,
    scores,
    threshold=0.5,
    sample_weight=None,
    model_name="model",
    split="test",
    track="track",
):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    recall = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    f1 = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "track": track,
        "model": model_name,
        "split": split,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_rows": int(len(y_true)),
    }


def choose_threshold_for_recall(y_true, scores, sample_weight=None, recall_target=0.80):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    thresholds = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 999)))

    best = None

    for th in thresholds:
        row = binary_metrics_from_scores(
            y_true,
            scores,
            threshold=th,
            sample_weight=sample_weight,
        )

        if row["recall"] >= recall_target:
            if best is None or row["precision"] > best["precision"]:
                best = row

    if best is not None:
        return float(best["threshold"]), "recall_target_precision_max"

    best = None

    for th in thresholds:
        row = binary_metrics_from_scores(
            y_true,
            scores,
            threshold=th,
            sample_weight=sample_weight,
        )

        if best is None or row["f1"] > best["f1"]:
            best = row

    return float(best["threshold"]), "fallback_max_f1"


def counts_to_expectation_z0(counts):
    shots = sum(counts.values())

    if shots == 0:
        return 0.0

    exp = 0.0

    for bitstring, n in counts.items():
        bit_q0 = bitstring.replace(" ", "")[-1]
        z = 1.0 if bit_q0 == "0" else -1.0
        exp += z * n / shots

    return float(exp)


def counts_to_score(counts, scale, bias):
    exp_z = counts_to_expectation_z0(counts)
    return float(sigmoid(scale * exp_z + bias))


def get_counts_from_sampler_pub_result(pub_result):
    data = pub_result.data

    if hasattr(data, "meas"):
        return data.meas.get_counts()

    for attr in dir(data):
        if attr.startswith("_"):
            continue

        obj = getattr(data, attr)

        if hasattr(obj, "get_counts"):
            return obj.get_counts()

    raise ValueError("Could not extract counts from SamplerV2 result data.")


def build_qiskit_vqc_circuit(x, params, n_qubits, layers, measure=True):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(float(x[q]), q)

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qc.rzz(float(0.5 * x[i] * x[j]), i, j)

    circuit_params = params[: layers * n_qubits]
    k = 0

    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(float(circuit_params[k]), q)
            k += 1

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    if measure:
        qc.measure_all()

    return qc
