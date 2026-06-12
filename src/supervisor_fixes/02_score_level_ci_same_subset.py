import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def feature_states(X):
    states = []

    for x in X:
        state = np.array([1.0], dtype=np.float64)

        for angle in x:
            q = np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=np.float64)
            state = np.kron(state, q)

        states.append(state)

    return np.vstack(states)


def quantum_kernel(A, B):
    return np.abs(A @ B.T) ** 2


def ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rz(theta):
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def apply_single_gate_batch(states, gate, qubit, n_qubits):
    states = states.reshape((-1,) + (2,) * n_qubits)
    axes = list(range(n_qubits + 1))
    target_axis = qubit + 1
    states = np.moveaxis(states, target_axis, -1)
    states = states @ gate.T
    states = np.moveaxis(states, -1, target_axis)
    return states.reshape((states.shape[0], 2**n_qubits))


def apply_cnot_batch(states, control, target, n_qubits):
    out = states.copy()

    for idx in range(2**n_qubits):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]

        if bits[control] == 1:
            bits[target] ^= 1
            j = 0
            for b in bits:
                j = (j << 1) | b
            out[:, j] = states[:, idx]

    return out


def vqc_scores(X, params, layers=2):
    n_qubits = X.shape[1]
    states = feature_states(X).astype(np.complex128)

    pos = 0

    for _ in range(layers):
        for q in range(n_qubits):
            states = apply_single_gate_batch(states, ry(params[pos]), q, n_qubits)
            pos += 1
            states = apply_single_gate_batch(states, rz(params[pos]), q, n_qubits)
            pos += 1

        for q in range(n_qubits - 1):
            states = apply_cnot_batch(states, q, q + 1, n_qubits)

    probs = np.abs(states) ** 2
    score = np.zeros(X.shape[0])

    for idx in range(2**n_qubits):
        first_bit = (idx >> (n_qubits - 1)) & 1
        if first_bit == 1:
            score += probs[:, idx]

    return score


def train_vqc(X_train, y_train, layers=2, maxiter=80, seed=42):
    rng = np.random.default_rng(seed)
    n_qubits = X_train.shape[1]
    n_params = layers * n_qubits * 2
    init = rng.normal(0, 0.2, size=n_params)

    y_float = y_train.astype(float)

    def loss(params):
        s = np.clip(vqc_scores(X_train, params, layers), 1e-6, 1 - 1e-6)
        return -np.mean(y_float * np.log(s) + (1 - y_float) * np.log(1 - s))

    result = minimize(
        loss,
        init,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )

    return result.x, result


def select_best_f1_threshold(y_true, scores):
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for th in thresholds:
        pred = (scores >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)

    return best_threshold


def metric_dict(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_rows": int(len(y_true)),
        "threshold": float(threshold),
    }


def bootstrap_ci(y_true, scores, threshold, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        sb = scores[idx]

        # skip invalid AUC bootstrap samples where only one class appears
        if len(np.unique(yb)) < 2:
            continue

        rows.append(metric_dict(yb, sb, threshold))

    boot = pd.DataFrame(rows)

    out = {}

    for metric in ["accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc", "fpr", "fnr"]:
        arr = boot[metric].dropna().to_numpy()
        out[f"{metric}_ci_lower"] = np.percentile(arr, 2.5)
        out[f"{metric}_ci_upper"] = np.percentile(arr, 97.5)
        out[f"{metric}_bootstrap_mean"] = np.mean(arr)
        out[f"{metric}_bootstrap_std"] = np.std(arr)

    return out


def run_classical_model(name, model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()

    if hasattr(model, "predict_proba"):
        valid_scores = model.predict_proba(X_valid)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
    else:
        valid_scores = model.decision_function(X_valid)
        test_scores = model.decision_function(X_test)

    inference_time = time.perf_counter() - t1

    threshold = select_best_f1_threshold(y_valid, valid_scores)
    metrics = metric_dict(y_test, test_scores, threshold)

    return metrics, valid_scores, test_scores, train_time, inference_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_reports)
    out.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.dataset, allow_pickle=True)
    X = raw["X"]
    y = raw["y"].astype(int)
    split = raw["split"].astype(str)

    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    performance_rows = []
    runtime_rows = []
    score_rows = []
    ci_rows = []

    classical_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=args.random_state)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=args.random_state, n_jobs=-1)),
        ("SVM RBF", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=args.random_state)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=args.random_state)),
    ]

    for name, model in classical_models:
        print("Training:", name)
        metrics, valid_scores, test_scores, train_time, inference_time = run_classical_model(
            name, model, X_train, y_train, X_valid, y_valid, X_test, y_test
        )

        threshold = metrics["threshold"]

        row = {
            "evaluation_group": "same_reduced_quantum_compatible_subset",
            "model": name,
            "model_family": "Classical",
        }
        row.update(metrics)
        performance_rows.append(row)

        ci = {
            "model": name,
            "model_family": "Classical",
            "n_bootstrap": args.bootstrap,
            "ci_basis": "score-level bootstrap on same test subset",
        }
        ci.update(bootstrap_ci(y_test, test_scores, threshold, args.bootstrap, args.random_state))
        ci_rows.append(ci)

        runtime_rows.append({
            "model": name,
            "model_family": "Classical",
            "train_rows": len(y_train),
            "valid_rows": len(y_valid),
            "test_rows": len(y_test),
            "training_time_seconds": train_time,
            "valid_plus_test_scoring_time_seconds": inference_time,
            "n_qubits": np.nan,
            "shots": np.nan,
            "circuits_or_state_evaluations": np.nan,
            "compute_cost_type": "Classical CPU runtime on reduced subset",
        })

        for i, s in enumerate(test_scores):
            score_rows.append({
                "model": name,
                "model_family": "Classical",
                "sample_index": i,
                "label": int(y_test[i]),
                "score": float(s),
                "threshold": float(threshold),
                "prediction": int(s >= threshold),
            })

    # Quantum Kernel SVC
    print("Training: Quantum Kernel SVC Statevector")
    t0 = time.perf_counter()
    train_states = feature_states(X_train)
    valid_states = feature_states(X_valid)
    test_states = feature_states(X_test)

    K_train = quantum_kernel(train_states, train_states)
    K_valid = quantum_kernel(valid_states, train_states)
    K_test = quantum_kernel(test_states, train_states)

    qsvc = SVC(kernel="precomputed", class_weight="balanced")
    qsvc.fit(K_train, y_train)
    qsvc_train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    valid_scores = qsvc.decision_function(K_valid)
    test_scores = qsvc.decision_function(K_test)
    qsvc_infer_time = time.perf_counter() - t1

    threshold = select_best_f1_threshold(y_valid, valid_scores)
    metrics = metric_dict(y_test, test_scores, threshold)

    row = {
        "evaluation_group": "same_reduced_quantum_compatible_subset",
        "model": "Quantum Kernel SVC Statevector",
        "model_family": "Quantum simulation",
    }
    row.update(metrics)
    performance_rows.append(row)

    ci = {
        "model": "Quantum Kernel SVC Statevector",
        "model_family": "Quantum simulation",
        "n_bootstrap": args.bootstrap,
        "ci_basis": "score-level bootstrap on same test subset",
    }
    ci.update(bootstrap_ci(y_test, test_scores, threshold, args.bootstrap, args.random_state))
    ci_rows.append(ci)

    runtime_rows.append({
        "model": "Quantum Kernel SVC Statevector",
        "model_family": "Quantum simulation",
        "train_rows": len(y_train),
        "valid_rows": len(y_valid),
        "test_rows": len(y_test),
        "training_time_seconds": qsvc_train_time,
        "valid_plus_test_scoring_time_seconds": qsvc_infer_time,
        "n_qubits": X.shape[1],
        "shots": np.nan,
        "circuits_or_state_evaluations": len(y_train) + len(y_valid) + len(y_test),
        "compute_cost_type": "Statevector simulation on classical hardware",
    })

    for i, s in enumerate(test_scores):
        score_rows.append({
            "model": "Quantum Kernel SVC Statevector",
            "model_family": "Quantum simulation",
            "sample_index": i,
            "label": int(y_test[i]),
            "score": float(s),
            "threshold": float(threshold),
            "prediction": int(s >= threshold),
        })

    # Hybrid IF + Quantum Kernel SVC uses same IF-selected subset, same QKSVC scores.
    hybrid_row = row.copy()
    hybrid_row["model"] = "Isolation Forest + Quantum Kernel SVC Statevector"
    hybrid_row["model_family"] = "Hybrid quantum simulation"
    performance_rows.append(hybrid_row)

    hybrid_ci = ci.copy()
    hybrid_ci["model"] = "Isolation Forest + Quantum Kernel SVC Statevector"
    hybrid_ci["model_family"] = "Hybrid quantum simulation"
    ci_rows.append(hybrid_ci)

    runtime_rows.append({
        "model": "Isolation Forest + Quantum Kernel SVC Statevector",
        "model_family": "Hybrid quantum simulation",
        "train_rows": len(y_train),
        "valid_rows": len(y_valid),
        "test_rows": len(y_test),
        "training_time_seconds": qsvc_train_time,
        "valid_plus_test_scoring_time_seconds": qsvc_infer_time,
        "n_qubits": X.shape[1],
        "shots": np.nan,
        "circuits_or_state_evaluations": len(y_train) + len(y_valid) + len(y_test),
        "compute_cost_type": "Isolation Forest-selected subset + QKSVC statevector simulation",
    })

    for i, s in enumerate(test_scores):
        score_rows.append({
            "model": "Isolation Forest + Quantum Kernel SVC Statevector",
            "model_family": "Hybrid quantum simulation",
            "sample_index": i,
            "label": int(y_test[i]),
            "score": float(s),
            "threshold": float(threshold),
            "prediction": int(s >= threshold),
        })

    # VQC
    print("Training: VQC Statevector")
    t0 = time.perf_counter()
    params, result = train_vqc(
        X_train,
        y_train,
        layers=args.layers,
        maxiter=args.maxiter,
        seed=args.random_state,
    )
    vqc_train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    valid_scores = vqc_scores(X_valid, params, args.layers)
    test_scores = vqc_scores(X_test, params, args.layers)
    vqc_infer_time = time.perf_counter() - t1

    threshold = select_best_f1_threshold(y_valid, valid_scores)
    metrics = metric_dict(y_test, test_scores, threshold)

    row = {
        "evaluation_group": "same_reduced_quantum_compatible_subset",
        "model": "VQC Statevector",
        "model_family": "Quantum simulation",
    }
    row.update(metrics)
    performance_rows.append(row)

    ci = {
        "model": "VQC Statevector",
        "model_family": "Quantum simulation",
        "n_bootstrap": args.bootstrap,
        "ci_basis": "score-level bootstrap on same test subset",
    }
    ci.update(bootstrap_ci(y_test, test_scores, threshold, args.bootstrap, args.random_state))
    ci_rows.append(ci)

    runtime_rows.append({
        "model": "VQC Statevector",
        "model_family": "Quantum simulation",
        "train_rows": len(y_train),
        "valid_rows": len(y_valid),
        "test_rows": len(y_test),
        "training_time_seconds": vqc_train_time,
        "valid_plus_test_scoring_time_seconds": vqc_infer_time,
        "n_qubits": X.shape[1],
        "shots": np.nan,
        "circuits_or_state_evaluations": len(y_train) + len(y_valid) + len(y_test),
        "compute_cost_type": "Statevector simulation on classical hardware",
    })

    for i, s in enumerate(test_scores):
        score_rows.append({
            "model": "VQC Statevector",
            "model_family": "Quantum simulation",
            "sample_index": i,
            "label": int(y_test[i]),
            "score": float(s),
            "threshold": float(threshold),
            "prediction": int(s >= threshold),
        })

    # Hybrid IF + VQC uses same IF-selected subset, same VQC scores.
    hybrid_row = row.copy()
    hybrid_row["model"] = "Isolation Forest + VQC Statevector"
    hybrid_row["model_family"] = "Hybrid quantum simulation"
    performance_rows.append(hybrid_row)

    hybrid_ci = ci.copy()
    hybrid_ci["model"] = "Isolation Forest + VQC Statevector"
    hybrid_ci["model_family"] = "Hybrid quantum simulation"
    ci_rows.append(hybrid_ci)

    runtime_rows.append({
        "model": "Isolation Forest + VQC Statevector",
        "model_family": "Hybrid quantum simulation",
        "train_rows": len(y_train),
        "valid_rows": len(y_valid),
        "test_rows": len(y_test),
        "training_time_seconds": vqc_train_time,
        "valid_plus_test_scoring_time_seconds": vqc_infer_time,
        "n_qubits": X.shape[1],
        "shots": np.nan,
        "circuits_or_state_evaluations": len(y_train) + len(y_valid) + len(y_test),
        "compute_cost_type": "Isolation Forest-selected subset + VQC statevector simulation",
    })

    for i, s in enumerate(test_scores):
        score_rows.append({
            "model": "Isolation Forest + VQC Statevector",
            "model_family": "Hybrid quantum simulation",
            "sample_index": i,
            "label": int(y_test[i]),
            "score": float(s),
            "threshold": float(threshold),
            "prediction": int(s >= threshold),
        })

    perf = pd.DataFrame(performance_rows)
    ci_df = pd.DataFrame(ci_rows)
    runtime = pd.DataFrame(runtime_rows)
    scores = pd.DataFrame(score_rows)

    for df in [perf, ci_df, runtime]:
        for c in df.columns:
            if df[c].dtype.kind in "fc":
                df[c] = df[c].round(6)

    perf.to_csv(out / "same_subset_model_comparison_table.csv", index=False)
    ci_df.to_csv(out / "score_level_bootstrap_ci_table.csv", index=False)
    runtime.to_csv(out / "same_subset_runtime_table.csv", index=False)
    scores.to_csv(out / "score_level_predictions_reduced_subset.csv", index=False)

    with open(out / "same_subset_uncertainty_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ci_method": "Bootstrap resampling from score-level test predictions.",
                "bootstrap_iterations": args.bootstrap,
                "threshold_selection": "Threshold selected on validation split by maximum F1, then fixed for test/bootstrap.",
                "important_note": "This same-subset comparison is valid only for the reduced quantum-compatible subset. It should not be used to claim full-scale quantum superiority.",
                "test_rows": int(len(y_test)),
                "train_rows": int(len(y_train)),
                "valid_rows": int(len(y_valid)),
                "n_qubits": int(X.shape[1]),
            },
            f,
            indent=2,
        )

    print("Performance:")
    print(perf[["model", "accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc", "n_rows"]])

    print("\nCI:")
    print(ci_df[["model", "f1_score_ci_lower", "f1_score_ci_upper", "pr_auc_ci_lower", "pr_auc_ci_upper", "roc_auc_ci_lower", "roc_auc_ci_upper"]])

    print("\nSaved score-level predictions and CI tables to:", out)


if __name__ == "__main__":
    main()
