import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def find_transaction_files():
    files = sorted(Path("/kaggle/input").rglob("*_Trans.csv"))
    if not files:
        raise FileNotFoundError("No *_Trans.csv files found under /kaggle/input.")
    return files


def preprocess_chunk(df):
    out = pd.DataFrame()
    out["amount_received"] = pd.to_numeric(df["Amount Received"], errors="coerce")
    out["amount_paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
    out["amount_delta"] = out["amount_received"] - out["amount_paid"]
    out["amount_ratio"] = out["amount_received"] / (out["amount_paid"].abs() + 1e-9)
    out["log_amount_received"] = np.log1p(out["amount_received"].clip(lower=0))
    out["log_amount_paid"] = np.log1p(out["amount_paid"].clip(lower=0))
    out["label"] = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def feature_states(X):
    states = []
    for x in X:
        state = np.array([1.0], dtype=np.float64)
        for angle in x:
            q = np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=np.float64)
            state = np.kron(state, q)
        states.append(state)
    return np.vstack(states)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-models", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    out_data = Path(args.out_data)
    out_models = Path(args.out_models)
    out_reports = Path(args.out_reports)

    out_data.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    dataset_path = out_data / "phase7_if_quantum_dataset_4q.npz"
    params_path = out_models / "phase7_if_plus_vqc_params.npz"

    if dataset_path.exists() and params_path.exists():
        print("Dataset and VQC params already exist. Skipping creation.")
        return

    files = find_transaction_files()

    if_train_rows = []
    normal_rows = []
    fraud_rows = []

    feature_cols = [
        "amount_received",
        "amount_paid",
        "amount_delta",
        "amount_ratio",
        "log_amount_received",
        "log_amount_paid",
    ]

    for f in files:
        print("Reading:", f)
        for chunk in pd.read_csv(f, chunksize=300000):
            data = preprocess_chunk(chunk)

            normal = data[data["label"] == 0]
            fraud = data[data["label"] == 1]

            if sum(len(x) for x in if_train_rows) < 50000 and len(normal) > 0:
                need = 50000 - sum(len(x) for x in if_train_rows)
                if_train_rows.append(normal.head(need))

            if sum(len(x) for x in normal_rows) < 8000 and len(normal) > 0:
                need = 8000 - sum(len(x) for x in normal_rows)
                normal_rows.append(normal.tail(need))

            if sum(len(x) for x in fraud_rows) < 8000 and len(fraud) > 0:
                need = 8000 - sum(len(x) for x in fraud_rows)
                fraud_rows.append(fraud.head(need))

            if (
                sum(len(x) for x in if_train_rows) >= 50000
                and sum(len(x) for x in normal_rows) >= 8000
                and sum(len(x) for x in fraud_rows) >= 8000
            ):
                break

        if (
            sum(len(x) for x in if_train_rows) >= 50000
            and sum(len(x) for x in normal_rows) >= 8000
            and sum(len(x) for x in fraud_rows) >= 8000
        ):
            break

    if_train = pd.concat(if_train_rows, ignore_index=True)
    normal_pool = pd.concat(normal_rows, ignore_index=True)
    fraud_pool = pd.concat(fraud_rows, ignore_index=True)

    scaler = StandardScaler()
    X_if = scaler.fit_transform(if_train[feature_cols])

    iso = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=args.random_state,
        n_jobs=-1,
    )
    iso.fit(X_if)

    normal_pool["if_score"] = -iso.score_samples(scaler.transform(normal_pool[feature_cols]))
    fraud_pool["if_score"] = -iso.score_samples(scaler.transform(fraud_pool[feature_cols]))

    normal_pool = normal_pool.sort_values("if_score", ascending=False)
    fraud_pool = fraud_pool.sort_values("if_score", ascending=False)

    n_train = 250
    n_valid = 150
    n_test = 150
    n_each = n_train + n_valid + n_test

    normal_sel = normal_pool.head(n_each).copy()
    fraud_sel = fraud_pool.head(n_each).copy()

    normal_sel["split"] = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    fraud_sel["split"] = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test

    normal_sel["label"] = 0
    fraud_sel["label"] = 1

    final = pd.concat([normal_sel, fraud_sel], ignore_index=True)
    final = final.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    q_features = final[
        [
            "log_amount_received",
            "log_amount_paid",
            "amount_delta",
            "amount_ratio",
        ]
    ].copy()

    q_scaler = StandardScaler()
    X_scaled = q_scaler.fit_transform(q_features)

    X_min = X_scaled.min(axis=0)
    X_max = X_scaled.max(axis=0)
    X_angles = 2 * np.pi * ((X_scaled - X_min) / (X_max - X_min + 1e-9)) - np.pi

    X = X_angles.astype(np.float64)
    y = final["label"].astype(int).to_numpy()
    split = final["split"].astype(str).to_numpy()
    if_score = final["if_score"].astype(float).to_numpy()

    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        split=split,
        if_score=if_score,
    )

    final.to_csv(out_reports / "phase7_dataset_source_rows.csv", index=False)

    counts = final.groupby(["split", "label"]).size().reset_index(name="count")
    counts.to_csv(out_reports / "phase7_dataset_counts.csv", index=False)

    X_train = X[split == "train"]
    y_train = y[split == "train"]

    params, result = train_vqc(
        X_train,
        y_train,
        layers=args.layers,
        maxiter=args.maxiter,
        seed=args.random_state,
    )

    np.savez_compressed(
        params_path,
        params=params,
        layers=args.layers,
        maxiter=args.maxiter,
        success=bool(result.success),
        final_loss=float(result.fun),
    )

    metadata = {
        "dataset": str(dataset_path),
        "params": str(params_path),
        "n_rows": int(len(final)),
        "n_qubits": 4,
        "train_per_class": n_train,
        "valid_per_class": n_valid,
        "test_per_class": n_test,
    }

    with open(out_reports / "phase7_dataset_and_vqc_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved dataset:", dataset_path)
    print("Saved VQC params:", params_path)
    print(counts)


if __name__ == "__main__":
    main()
