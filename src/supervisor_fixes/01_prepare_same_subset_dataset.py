import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def find_transaction_files():
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]

    files = []

    for root in roots:
        if root.exists():
            files.extend(root.rglob("*_Trans.csv"))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError("No *_Trans.csv files found under /kaggle/input.")

    return files


def preprocess_chunk(df):
    required = ["Timestamp", "Amount Received", "Amount Paid", "Is Laundering"]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--if-train-normal", type=int, default=50000)
    parser.add_argument("--pool-normal", type=int, default=8000)
    parser.add_argument("--pool-fraud", type=int, default=8000)
    parser.add_argument("--chunksize", type=int, default=300000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_data = Path(args.out_data)
    out_reports = Path(args.out_reports)
    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    files = find_transaction_files()

    print("Transaction files found:")
    for f in files:
        print(" -", f)

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

        for chunk in pd.read_csv(f, chunksize=args.chunksize):
            data = preprocess_chunk(chunk)

            normal = data[data["label"] == 0]
            fraud = data[data["label"] == 1]

            if sum(len(x) for x in if_train_rows) < args.if_train_normal and len(normal) > 0:
                need = args.if_train_normal - sum(len(x) for x in if_train_rows)
                if_train_rows.append(normal.head(need))

            if sum(len(x) for x in normal_rows) < args.pool_normal and len(normal) > 0:
                need = args.pool_normal - sum(len(x) for x in normal_rows)
                normal_rows.append(normal.tail(need))

            if sum(len(x) for x in fraud_rows) < args.pool_fraud and len(fraud) > 0:
                need = args.pool_fraud - sum(len(x) for x in fraud_rows)
                fraud_rows.append(fraud.head(need))

            if (
                sum(len(x) for x in if_train_rows) >= args.if_train_normal
                and sum(len(x) for x in normal_rows) >= args.pool_normal
                and sum(len(x) for x in fraud_rows) >= args.pool_fraud
            ):
                print("Required pools reached.")
                break

        if (
            sum(len(x) for x in if_train_rows) >= args.if_train_normal
            and sum(len(x) for x in normal_rows) >= args.pool_normal
            and sum(len(x) for x in fraud_rows) >= args.pool_fraud
        ):
            break

    if_train = pd.concat(if_train_rows, ignore_index=True)
    normal_pool = pd.concat(normal_rows, ignore_index=True)
    fraud_pool = pd.concat(fraud_rows, ignore_index=True)

    print("Collected IF train normal:", len(if_train))
    print("Collected normal pool:", len(normal_pool))
    print("Collected fraud pool:", len(fraud_pool))

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

    # Balanced reduced subset: 250 train/class, 150 valid/class, 150 test/class.
    n_train = 250
    n_valid = 150
    n_test = 150
    n_each = n_train + n_valid + n_test

    normal_sel = normal_pool.head(n_each).copy()
    fraud_sel = fraud_pool.head(n_each).copy()

    rows = []

    def assign_split(df, label):
        df = df.copy()
        df["label"] = label
        df.iloc[:n_train, df.columns.get_loc("label")] = label
        df["split"] = (
            ["train"] * n_train
            + ["valid"] * n_valid
            + ["test"] * n_test
        )
        return df

    rows.append(assign_split(normal_sel, 0))
    rows.append(assign_split(fraud_sel, 1))

    final = pd.concat(rows, ignore_index=True)

    rng = np.random.default_rng(args.random_state)
    final = final.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    q_features_raw = final[
        [
            "log_amount_received",
            "log_amount_paid",
            "amount_delta",
            "amount_ratio",
        ]
    ].copy()

    q_scaler = StandardScaler()
    X_scaled = q_scaler.fit_transform(q_features_raw)

    # Map to angle range [-pi, pi].
    X_min = X_scaled.min(axis=0)
    X_max = X_scaled.max(axis=0)
    X_angles = 2 * np.pi * ((X_scaled - X_min) / (X_max - X_min + 1e-9)) - np.pi

    X = X_angles.astype(np.float64)
    y = final["label"].astype(int).to_numpy()
    split = final["split"].astype(str).to_numpy()
    if_score = final["if_score"].astype(float).to_numpy()

    np.savez_compressed(
        out_data / "same_reduced_quantum_compatible_dataset.npz",
        X=X,
        y=y,
        split=split,
        if_score=if_score,
    )

    final.to_csv(out_data / "same_reduced_quantum_compatible_dataset_source.csv", index=False)

    counts = final.groupby(["split", "label"]).size().reset_index(name="count")
    counts.to_csv(out_reports / "same_reduced_subset_counts.csv", index=False)

    metadata = {
        "dataset_file": str(out_data / "same_reduced_quantum_compatible_dataset.npz"),
        "source_csv": str(out_data / "same_reduced_quantum_compatible_dataset_source.csv"),
        "n_rows": int(len(final)),
        "n_qubits": 4,
        "features": [
            "log_amount_received",
            "log_amount_paid",
            "amount_delta",
            "amount_ratio",
        ],
        "split_design": {
            "train_per_class": n_train,
            "valid_per_class": n_valid,
            "test_per_class": n_test,
        },
        "note": "Balanced reduced quantum-compatible subset selected using Isolation Forest anomaly ranking. This is for same-subset comparison and uncertainty analysis, not full-scale deployment evaluation.",
    }

    with open(out_reports / "same_reduced_subset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:", out_data / "same_reduced_quantum_compatible_dataset.npz")
    print(counts)


if __name__ == "__main__":
    main()
