import argparse
from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd

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

from qiskit import QuantumCircuit


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def stratified_select(X, y, split, split_name, per_class, seed):
    rng = np.random.default_rng(seed)
    mask = split == split_name

    idx_all = []

    for label in [0, 1]:
        idx = np.where(mask & (y == label))[0]

        if len(idx) == 0:
            raise ValueError(f"No rows for split={split_name}, label={label}")

        n = min(per_class, len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        idx_all.append(chosen)

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    return X[idx], y[idx], idx


def build_kernel_circuit(x, z):
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(float(x[q]), q)

    for q in range(n_qubits):
        qc.ry(float(-z[q]), q)

    qc.measure_all()
    return qc


def build_vqc_circuit(x, params, n_qubits, layers):
    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(float(x[q]), q)

    pos = 0

    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(float(params[pos]), q)
            pos += 1
            qc.rz(float(params[pos]), q)
            pos += 1

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    qc.measure_all()
    return qc


def clean_bitstring(bitstring):
    return str(bitstring).replace(" ", "")


def kernel_value_from_counts(counts, n_qubits):
    zero = "0" * n_qubits
    total = sum(counts.values())

    if total == 0:
        return 0.0

    return counts.get(zero, 0) / total


def vqc_score_from_counts(counts):
    total = sum(counts.values())

    if total == 0:
        return 0.0

    score = 0

    for bitstring, count in counts.items():
        b = clean_bitstring(bitstring)

        # Qiskit displays classical bits high-to-low.
        # First character corresponds to the highest-index measured qubit.
        if len(b) > 0 and b[0] == "1":
            score += count

    return score / total


def get_counts_from_pub_result(pub_result):
    data = pub_result.data

    for attr in ["meas", "c", "creg"]:
        if hasattr(data, attr):
            obj = getattr(data, attr)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()

    for attr in dir(data):
        if attr.startswith("_"):
            continue

        try:
            obj = getattr(data, attr)
            if hasattr(obj, "get_counts"):
                return obj.get_counts()
        except Exception:
            pass

    raise RuntimeError("Could not extract counts from Sampler result.")


def select_best_f1_threshold(y_true, scores):
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))

    best_threshold = thresholds[0]
    best_f1 = -1

    for th in thresholds:
        pred = (scores >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    return float(best_threshold)


def metrics_row(track, model, split_name, y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        roc = roc_auc_score(y_true, scores)
    except Exception:
        roc = np.nan

    try:
        pr = average_precision_score(y_true, scores)
    except Exception:
        pr = np.nan

    return {
        "track": track,
        "model": model,
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
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


def connect_service(token):
    from qiskit_ibm_runtime import QiskitRuntimeService

    errors = []

    for channel in ["ibm_quantum_platform", "ibm_quantum"]:
        try:
            service = QiskitRuntimeService(channel=channel, token=token)
            return service, channel
        except Exception as e:
            errors.append({"channel": channel, "error": str(e)})

    raise RuntimeError(f"Could not connect to IBM Runtime. Errors: {errors}")


def backend_name_safe(backend):
    try:
        return backend.name
    except Exception:
        try:
            return backend.name()
        except Exception:
            return str(backend)


def extract_job_payload(job):
    payload = {}

    try:
        payload["job_id"] = job.job_id()
    except Exception as e:
        payload["job_id"] = f"unavailable: {e}"

    try:
        payload["status"] = str(job.status())
    except Exception as e:
        payload["status"] = f"unavailable: {e}"

    try:
        payload["metrics"] = job.metrics()
    except Exception as e:
        payload["metrics"] = f"unavailable: {e}"

    try:
        payload["usage_estimation"] = job.usage_estimation
    except Exception as e:
        payload["usage_estimation"] = f"unavailable: {e}"

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vqc-params", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--valid-per-class", type=int, default=5)
    parser.add_argument("--test-per-class", type=int, default=5)
    parser.add_argument("--backend-name", default="ibm_kingston")
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--dry-run", default="true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dry_run = args.dry_run.lower() in ["true", "1", "yes", "y"]

    out_reports = Path(args.out_reports)
    out_reports.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.dataset, allow_pickle=True)

    X = raw["X"]
    y = raw["y"].astype(int)
    split = raw["split"].astype(str)

    n_qubits = X.shape[1]

    X_train, y_train, idx_train = stratified_select(
        X, y, split, "train", args.train_per_class, args.random_state
    )

    X_valid, y_valid, idx_valid = stratified_select(
        X, y, split, "valid", args.valid_per_class, args.random_state + 1
    )

    X_test, y_test, idx_test = stratified_select(
        X, y, split, "test", args.test_per_class, args.random_state + 2
    )

    vqc_raw = np.load(args.vqc_params, allow_pickle=True)
    vqc_params = vqc_raw["params"]
    vqc_layers = int(vqc_raw["layers"])

    circuits = []
    meta = []

    # Kernel train circuits: upper triangle only
    for i in range(len(X_train)):
        for j in range(i, len(X_train)):
            circuits.append(build_kernel_circuit(X_train[i], X_train[j]))
            meta.append({
                "model_group": "kernel_svc",
                "purpose": "train_kernel",
                "i": i,
                "j": j,
            })

    # Kernel valid circuits
    for i in range(len(X_valid)):
        for j in range(len(X_train)):
            circuits.append(build_kernel_circuit(X_valid[i], X_train[j]))
            meta.append({
                "model_group": "kernel_svc",
                "purpose": "valid_kernel",
                "i": i,
                "j": j,
            })

    # Kernel test circuits
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            circuits.append(build_kernel_circuit(X_test[i], X_train[j]))
            meta.append({
                "model_group": "kernel_svc",
                "purpose": "test_kernel",
                "i": i,
                "j": j,
            })

    # VQC valid circuits
    for i in range(len(X_valid)):
        circuits.append(build_vqc_circuit(X_valid[i], vqc_params, n_qubits, vqc_layers))
        meta.append({
            "model_group": "vqc",
            "purpose": "valid_vqc",
            "i": i,
            "j": -1,
        })

    # VQC test circuits
    for i in range(len(X_test)):
        circuits.append(build_vqc_circuit(X_test[i], vqc_params, n_qubits, vqc_layers))
        meta.append({
            "model_group": "vqc",
            "purpose": "test_vqc",
            "i": i,
            "j": -1,
        })

    inventory_rows = []

    for k, qc in enumerate(circuits):
        row = dict(meta[k])
        row.update({
            "circuit_index": k,
            "n_qubits": qc.num_qubits,
            "n_clbits": qc.num_clbits,
            "depth_untranspiled": qc.depth(),
            "size_untranspiled": qc.size(),
            "ops_untranspiled": dict(qc.count_ops()),
        })
        inventory_rows.append(row)

    write_csv(inventory_rows, out_reports / "phase7_if_quantum_hardware_circuit_inventory.csv")

    if dry_run:
        save_json(
            {
                "mode": "dry_run",
                "n_circuits": len(circuits),
                "n_qubits": n_qubits,
                "shots": args.shots,
                "train_rows": int(len(X_train)),
                "valid_rows": int(len(X_valid)),
                "test_rows": int(len(X_test)),
                "train_per_class": args.train_per_class,
                "valid_per_class": args.valid_per_class,
                "test_per_class": args.test_per_class,
                "note": "No IBM hardware job submitted.",
            },
            out_reports / "phase7_if_quantum_hardware_dry_run_metadata.json",
        )

        print("Dry-run complete. No IBM job submitted.")
        print("Circuits prepared:", len(circuits))
        print(pd.DataFrame(inventory_rows).head())
        return

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")

    if not token:
        raise ValueError("IBM_QUANTUM_TOKEN not found. Load it from Kaggle Secrets first.")

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service, channel = connect_service(token)

    if args.backend_name:
        backend = service.backend(args.backend_name)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=n_qubits,
        )

    backend_name = backend_name_safe(backend)

    print("Connected channel:", channel)
    print("Selected backend:", backend_name)
    print("Circuits to submit:", len(circuits))
    print("Shots:", args.shots)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=args.optimization_level,
        seed_transpiler=args.random_state,
    )

    t0 = time.perf_counter()
    transpiled = pm.run(circuits)
    transpile_time = time.perf_counter() - t0

    transpiled_rows = []

    for k, qc in enumerate(transpiled):
        row = dict(meta[k])
        row.update({
            "circuit_index": k,
            "depth_transpiled": qc.depth(),
            "size_transpiled": qc.size(),
            "ops_transpiled": dict(qc.count_ops()),
        })
        transpiled_rows.append(row)

    write_csv(transpiled_rows, out_reports / "phase7_if_quantum_hardware_transpiled_inventory.csv")

    sampler = Sampler(mode=backend)

    submit_start = time.perf_counter()
    job = sampler.run(transpiled, shots=args.shots)
    submit_end = time.perf_counter()

    job_id = job.job_id()
    print("Submitted IBM job:", job_id)

    result_start = time.perf_counter()
    result = job.result()
    result_end = time.perf_counter()

    total_turnaround = result_end - submit_start
    wait_for_result = result_end - result_start

    counts_rows = []

    K_train = np.zeros((len(X_train), len(X_train)))
    K_valid = np.zeros((len(X_valid), len(X_train)))
    K_test = np.zeros((len(X_test), len(X_train)))

    vqc_valid_scores = np.zeros(len(X_valid))
    vqc_test_scores = np.zeros(len(X_test))

    for k, pub_result in enumerate(result):
        counts = get_counts_from_pub_result(pub_result)
        m = meta[k]

        row = dict(m)
        row.update({
            "circuit_index": k,
            "counts_json": json.dumps(counts),
            "shots_observed": int(sum(counts.values())),
        })
        counts_rows.append(row)

        if m["model_group"] == "kernel_svc":
            val = kernel_value_from_counts(counts, n_qubits)

            if m["purpose"] == "train_kernel":
                i, j = m["i"], m["j"]
                K_train[i, j] = val
                K_train[j, i] = val

            elif m["purpose"] == "valid_kernel":
                K_valid[m["i"], m["j"]] = val

            elif m["purpose"] == "test_kernel":
                K_test[m["i"], m["j"]] = val

        elif m["model_group"] == "vqc":
            score = vqc_score_from_counts(counts)

            if m["purpose"] == "valid_vqc":
                vqc_valid_scores[m["i"]] = score

            elif m["purpose"] == "test_vqc":
                vqc_test_scores[m["i"]] = score

    pd.DataFrame(counts_rows).to_csv(
        out_reports / "phase7_if_quantum_hardware_counts.csv",
        index=False,
    )

    np.fill_diagonal(K_train, 1.0)

    # Hardware Quantum Kernel SVC
    kernel_model = SVC(kernel="precomputed", class_weight="balanced")
    kernel_train_start = time.perf_counter()
    kernel_model.fit(K_train, y_train)
    kernel_train_time = time.perf_counter() - kernel_train_start

    kernel_valid_scores = kernel_model.decision_function(K_valid)
    kernel_threshold = select_best_f1_threshold(y_valid, kernel_valid_scores)

    kernel_test_scores = kernel_model.decision_function(K_test)

    kernel_metrics = metrics_row(
        "isolation_forest_plus_quantum_hardware",
        "isolation_forest_plus_quantum_kernel_svc_ibm_hardware",
        "hardware_test_subset",
        y_test,
        kernel_test_scores,
        kernel_threshold,
    )

    # Hardware VQC
    vqc_threshold = select_best_f1_threshold(y_valid, vqc_valid_scores)

    vqc_metrics = metrics_row(
        "isolation_forest_plus_quantum_hardware",
        "isolation_forest_plus_vqc_ibm_hardware",
        "hardware_test_subset",
        y_test,
        vqc_test_scores,
        vqc_threshold,
    )

    perf_rows = [kernel_metrics, vqc_metrics]

    pd.DataFrame(perf_rows).to_csv(
        out_reports / "phase7_if_quantum_hardware_performance_metrics.csv",
        index=False,
    )

    score_rows = []

    for i in range(len(y_test)):
        score_rows.append({
            "model": "isolation_forest_plus_quantum_kernel_svc_ibm_hardware",
            "sample_index": i,
            "original_dataset_index": int(idx_test[i]),
            "label": int(y_test[i]),
            "score": float(kernel_test_scores[i]),
            "threshold": float(kernel_threshold),
            "prediction": int(kernel_test_scores[i] >= kernel_threshold),
        })

        score_rows.append({
            "model": "isolation_forest_plus_vqc_ibm_hardware",
            "sample_index": i,
            "original_dataset_index": int(idx_test[i]),
            "label": int(y_test[i]),
            "score": float(vqc_test_scores[i]),
            "threshold": float(vqc_threshold),
            "prediction": int(vqc_test_scores[i] >= vqc_threshold),
        })

    pd.DataFrame(score_rows).to_csv(
        out_reports / "phase7_if_quantum_hardware_scores.csv",
        index=False,
    )

    job_payload = extract_job_payload(job)
    metrics_payload = job_payload.get("metrics", {})
    usage = {}

    if isinstance(metrics_payload, dict):
        usage = metrics_payload.get("usage", {})

    total_circuits = len(circuits)
    kernel_circuits = sum(1 for m in meta if m["model_group"] == "kernel_svc")
    vqc_circuits = sum(1 for m in meta if m["model_group"] == "vqc")

    runtime_rows = [
        {
            "track": "isolation_forest_plus_quantum_hardware",
            "model": "isolation_forest_plus_quantum_kernel_svc_ibm_hardware",
            "stage1_model": "Isolation Forest",
            "stage2_model": "Quantum Kernel SVC IBM Hardware",
            "backend_name": backend_name,
            "job_id": job_id,
            "n_qubits": n_qubits,
            "shots": args.shots,
            "model_circuits": kernel_circuits,
            "total_submitted_circuits": total_circuits,
            "kernel_train_time_seconds": kernel_train_time,
            "transpile_time_seconds": transpile_time,
            "wait_for_result_time_seconds": wait_for_result,
            "total_turnaround_time_seconds": total_turnaround,
            "ibm_quantum_seconds": usage.get("quantum_seconds"),
            "ibm_usage_seconds": usage.get("seconds"),
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "test_rows": len(X_test),
            "compute_cost_type": "Real IBM QPU hardware cost measured by circuits, shots, turnaround time, and quantum seconds",
        },
        {
            "track": "isolation_forest_plus_quantum_hardware",
            "model": "isolation_forest_plus_vqc_ibm_hardware",
            "stage1_model": "Isolation Forest",
            "stage2_model": "VQC IBM Hardware",
            "backend_name": backend_name,
            "job_id": job_id,
            "n_qubits": n_qubits,
            "shots": args.shots,
            "model_circuits": vqc_circuits,
            "total_submitted_circuits": total_circuits,
            "kernel_train_time_seconds": np.nan,
            "transpile_time_seconds": transpile_time,
            "wait_for_result_time_seconds": wait_for_result,
            "total_turnaround_time_seconds": total_turnaround,
            "ibm_quantum_seconds": usage.get("quantum_seconds"),
            "ibm_usage_seconds": usage.get("seconds"),
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "test_rows": len(X_test),
            "compute_cost_type": "Real IBM QPU hardware cost measured by circuits, shots, turnaround time, and quantum seconds",
        },
    ]

    pd.DataFrame(runtime_rows).to_csv(
        out_reports / "phase7_if_quantum_hardware_runtime_cost.csv",
        index=False,
    )

    save_json(
        {
            "mode": "ibm_hardware",
            "connected_channel": channel,
            "backend_name": backend_name,
            "job_id": job_id,
            "shots": args.shots,
            "n_qubits": n_qubits,
            "train_per_class": args.train_per_class,
            "valid_per_class": args.valid_per_class,
            "test_per_class": args.test_per_class,
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
            "total_submitted_circuits": total_circuits,
            "kernel_circuits": kernel_circuits,
            "vqc_circuits": vqc_circuits,
            "transpile_time_seconds": transpile_time,
            "submit_overhead_seconds": submit_end - submit_start,
            "wait_for_result_time_seconds": wait_for_result,
            "total_turnaround_time_seconds": total_turnaround,
            "job_payload": job_payload,
            "note": "This run validates Isolation Forest + Quantum Kernel SVC and Isolation Forest + VQC on real IBM Quantum hardware using a small balanced Phase 7 subset.",
        },
        out_reports / "phase7_if_quantum_hardware_job_metadata.json",
    )

    print("IBM hardware Phase 7 complete.")
    print("Job ID:", job_id)
    print("Backend:", backend_name)
    print("Performance:")
    print(pd.DataFrame(perf_rows))
    print("Runtime/cost:")
    print(pd.DataFrame(runtime_rows))


if __name__ == "__main__":
    main()
