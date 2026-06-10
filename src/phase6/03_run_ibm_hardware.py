import argparse
from pathlib import Path
import os
import time
import json
import numpy as np
import pandas as pd

from phase6_common import (
    read_json,
    save_json,
    write_rows_csv,
    build_qiskit_vqc_circuit,
    counts_to_score,
    binary_metrics_from_scores,
)


def get_test_split(data):
    mask = data["split"] == "test"
    return (
        data["X"][mask],
        data["y"][mask].astype(int),
        data["sample_weight"][mask].astype(float),
    )


def stratified_limit(X, y, w, max_per_class, seed):
    rng = np.random.default_rng(seed)
    idx_all = []

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        n = min(max_per_class, len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        idx_all.append(chosen)

    idx = np.concatenate(idx_all)
    rng.shuffle(idx)

    return X[idx], y[idx], w[idx], idx


def extract_job_metrics(job):
    payload = {}

    for attr in ["job_id", "status"]:
        try:
            val = getattr(job, attr)
            payload[attr] = str(val() if callable(val) else val)
        except Exception as e:
            payload[attr] = f"unavailable: {e}"

    try:
        payload["metrics"] = job.metrics()
    except Exception as e:
        payload["metrics"] = f"unavailable: {e}"

    try:
        payload["usage_estimation"] = job.usage_estimation
    except Exception as e:
        payload["usage_estimation"] = f"unavailable: {e}"

    return payload


def get_backend_name(backend):
    try:
        return backend.name
    except Exception:
        pass

    try:
        return backend.name()
    except Exception:
        return str(backend)


def connect_ibm_service(token):
    from qiskit_ibm_runtime import QiskitRuntimeService

    errors = []

    for channel in ["ibm_quantum_platform", "ibm_quantum"]:
        try:
            service = QiskitRuntimeService(channel=channel, token=token)
            return service, channel
        except Exception as e:
            errors.append({"channel": channel, "error": str(e)})

    raise RuntimeError(f"Could not connect to IBM Quantum Runtime. Errors: {errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-dataset", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-test-per-class", type=int, default=40)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--dry-run", default="true")
    parser.add_argument("--backend-name", default="")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dry_run = args.dry_run.lower() in ["true", "1", "yes", "y"]

    out_reports = Path(args.out_reports)
    out_reports.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.phase6_dataset, allow_pickle=True)

    data = {
        "X": raw["X"],
        "y": raw["y"],
        "sample_weight": raw["sample_weight"],
        "split": raw["split"].astype(str),
    }

    X_test, y_test, w_test = get_test_split(data)

    X_hw, y_hw, w_hw, selected_idx = stratified_limit(
        X_test,
        y_test,
        w_test,
        max_per_class=args.max_test_per_class,
        seed=args.random_state,
    )

    param_payload = read_json(args.params)
    params = np.asarray(param_payload["params"], dtype=float)
    n_qubits = int(param_payload["n_qubits"])
    layers = int(param_payload["layers"])
    scale = float(param_payload["scale"])
    bias = float(param_payload["bias"])
    threshold = float(param_payload["threshold"])

    circuits = [
        build_qiskit_vqc_circuit(
            X_hw[i],
            params=params,
            n_qubits=n_qubits,
            layers=layers,
            measure=True,
        )
        for i in range(len(X_hw))
    ]

    circuit_rows = []

    for i, qc in enumerate(circuits):
        circuit_rows.append({
            "sample_index": int(i),
            "original_test_index": int(selected_idx[i]),
            "label": int(y_hw[i]),
            "weight": float(w_hw[i]),
            "n_qubits": int(n_qubits),
            "n_clbits": int(qc.num_clbits),
            "depth_untranspiled": int(qc.depth()),
            "size_untranspiled": int(qc.size()),
            "ops_untranspiled": dict(qc.count_ops()),
        })

    write_rows_csv(circuit_rows, out_reports / "phase6_hardware_circuit_inventory.csv")

    if dry_run:
        save_json(
            {
                "mode": "dry_run",
                "message": "No IBM hardware job submitted.",
                "circuits_prepared": len(circuits),
                "shots": args.shots,
                "max_test_per_class": args.max_test_per_class,
                "n_qubits": n_qubits,
                "threshold": threshold,
            },
            out_reports / "phase6_hardware_dry_run_metadata.json",
        )

        print("Dry-run complete. Circuits prepared but no IBM job submitted.")
        print(pd.DataFrame(circuit_rows).head())
        return

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")

    if not token:
        raise ValueError("IBM_QUANTUM_TOKEN not found. Load it from Kaggle Secrets first.")

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service, used_channel = connect_ibm_service(token)

    if args.backend_name.strip():
        backend = service.backend(args.backend_name.strip())
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=n_qubits,
        )

    backend_name = get_backend_name(backend)

    print("Connected channel:", used_channel)
    print("Selected backend:", backend_name)

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=args.optimization_level,
        seed_transpiler=args.random_state,
    )

    start_transpile = time.perf_counter()
    transpiled_circuits = pm.run(circuits)
    transpile_time = time.perf_counter() - start_transpile

    transpiled_rows = []

    for i, qc in enumerate(transpiled_circuits):
        transpiled_rows.append({
            "sample_index": int(i),
            "label": int(y_hw[i]),
            "depth_transpiled": int(qc.depth()),
            "size_transpiled": int(qc.size()),
            "ops_transpiled": dict(qc.count_ops()),
        })

    write_rows_csv(transpiled_rows, out_reports / "phase6_transpiled_circuit_inventory.csv")

    sampler = Sampler(mode=backend)

    submit_start = time.perf_counter()
    job = sampler.run(transpiled_circuits, shots=args.shots)
    submit_end = time.perf_counter()

    job_id = job.job_id()
    print("Submitted IBM job:", job_id)

    result_start = time.perf_counter()
    result = job.result()
    result_end = time.perf_counter()

    total_turnaround_time = result_end - submit_start
    wait_for_result_time = result_end - result_start

    counts_rows = []
    score_rows = []

    from phase6_common import get_counts_from_sampler_pub_result

    for i, pub_result in enumerate(result):
        counts = get_counts_from_sampler_pub_result(pub_result)
        score = counts_to_score(counts, scale=scale, bias=bias)

        counts_rows.append({
            "sample_index": int(i),
            "label": int(y_hw[i]),
            "counts_json": json.dumps(counts),
            "shots_observed": int(sum(counts.values())),
        })

        score_rows.append({
            "sample_index": int(i),
            "original_test_index": int(selected_idx[i]),
            "label": int(y_hw[i]),
            "sample_weight": float(w_hw[i]),
            "hardware_score": float(score),
            "threshold": threshold,
            "hardware_pred": int(score >= threshold),
        })

    pd.DataFrame(counts_rows).to_csv(out_reports / "phase6_hardware_counts.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out_reports / "phase6_hardware_scores.csv", index=False)

    score_df = pd.DataFrame(score_rows)

    metric = binary_metrics_from_scores(
        score_df["label"].to_numpy(),
        score_df["hardware_score"].to_numpy(),
        threshold=threshold,
        sample_weight=score_df["sample_weight"].to_numpy(),
        model_name="phase6_vqc_ibm_hardware",
        split="hardware_test_subset",
        track="phase6_ibm_hardware_validation",
    )

    write_rows_csv([metric], out_reports / "phase6_hardware_metrics.csv")

    job_payload = extract_job_metrics(job)

    hardware_metadata = {
        "mode": "ibm_hardware",
        "connected_channel": used_channel,
        "backend_name": backend_name,
        "job_id": job_id,
        "shots": args.shots,
        "n_circuits": len(circuits),
        "n_qubits": n_qubits,
        "optimization_level": args.optimization_level,
        "submit_overhead_seconds": submit_end - submit_start,
        "transpile_time_seconds": transpile_time,
        "wait_for_result_time_seconds": wait_for_result_time,
        "total_turnaround_time_seconds": total_turnaround_time,
        "job_payload": job_payload,
        "note": "Queue/execution details are recorded where available from IBM Runtime job metadata. Total turnaround is measured locally from submission to result retrieval.",
    }

    save_json(hardware_metadata, out_reports / "phase6_hardware_job_metadata.json")

    print("IBM hardware execution complete.")
    print(pd.DataFrame([metric]))
    print(hardware_metadata)


if __name__ == "__main__":
    main()
