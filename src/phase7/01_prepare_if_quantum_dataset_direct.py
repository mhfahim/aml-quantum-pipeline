import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def find_transaction_files(input_root):
    files = sorted(Path(input_root).rglob("*Trans.csv"))

    if not files:
        raise FileNotFoundError(f"No transaction CSV files found under {input_root}")

    return files


def derive_features(chunk):
    chunk = chunk.copy()

    t = pd.to_datetime(chunk["Timestamp"], errors="coerce", format="%Y/%m/%d %H:%M")
    y = pd.to_numeric(chunk["Is Laundering"], errors="coerce").fillna(0).astype(int)

    amount_received = pd.to_numeric(chunk["Amount Received"], errors="coerce").fillna(0.0)
    amount_paid = pd.to_numeric(chunk["Amount Paid"], errors="coerce").fillna(0.0)

    out = pd.DataFrame()
    out["timestamp"] = t
    out["label_clean"] = y

    out["amount_received_num"] = amount_received
    out["amount_paid_num"] = amount_paid
    out["amount_delta_num"] = amount_received - amount_paid
    out["amount_ratio_num"] = amount_received / (amount_paid.abs() + 1e-9)
    out["log_amount_received_num"] = np.log1p(amount_received.abs())
    out["log_amount_paid_num"] = np.log1p(amount_paid.abs())

    out["hour_num"] = t.dt.hour.fillna(0).astype(int)
    out["day_num"] = t.dt.day.fillna(0).astype(int)

    out["same_currency_num"] = (
        chunk["Receiving Currency"].astype(str) == chunk["Payment Currency"].astype(str)
    ).astype(int)

    feature_cols = [
        "amount_received_num",
        "amount_paid_num",
        "amount_delta_num",
        "amount_ratio_num",
        "log_amount_received_num",
        "log_amount_paid_num",
        "hour_num",
        "day_num",
        "same_currency_num",
    ]

    for col in feature_cols:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return out, feature_cols


def append_until(storage, key, data, max_rows):
    current = sum(len(x) for x in storage.get(key, []))
    need = max_rows - current

    if need <= 0 or data.empty:
        return

    storage.setdefault(key, []).append(data.head(need).copy())


def count_storage(storage, key):
    return sum(len(x) for x in storage.get(key, []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--if-train-normal-rows", type=int, default=50000)
    parser.add_argument("--normal-pool-rows", type=int, default=8000)
    parser.add_argument("--fraud-pool-rows", type=int, default=8000)
    parser.add_argument("--train-per-class", type=int, default=250)
    parser.add_argument("--valid-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=300000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_data = Path(args.out_data)
    out_reports = Path(args.out_reports)

    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    files = find_transaction_files(args.input_root)

    storage = {}
    feature_cols = None
    chunks_seen = 0
    rows_seen = 0

    usecols = [
        "Timestamp",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Is Laundering",
    ]

    print("Transaction files found:")
    for f in files:
        print("-", f)

    for file in files:
        print("\nReading:", file)

        for chunk in pd.read_csv(file, usecols=usecols, chunksize=args.chunksize, low_memory=False):
            chunks_seen += 1
            rows_seen += len(chunk)

            feat, feature_cols = derive_features(chunk)

            normal = feat[feat["label_clean"] == 0]
            fraud = feat[feat["label_clean"] == 1]

            append_until(storage, "if_train_normal", normal, args.if_train_normal_rows)

            # Collect normal pool by sampling from each chunk so it is not only from the beginning
            current_normal_pool = count_storage(storage, "normal_pool")
            if current_normal_pool < args.normal_pool_rows and not normal.empty:
                need = args.normal_pool_rows - current_normal_pool
                take_n = min(need, max(1, min(len(normal), 1000)))
                sampled = normal.sample(n=take_n, random_state=args.random_state + chunks_seen)
                storage.setdefault("normal_pool", []).append(sampled.copy())

            append_until(storage, "fraud_pool", fraud, args.fraud_pool_rows)

            if chunks_seen % 50 == 0:
                print(
                    "chunks:", chunks_seen,
                    "rows_seen:", rows_seen,
                    "if_train_normal:", count_storage(storage, "if_train_normal"),
                    "normal_pool:", count_storage(storage, "normal_pool"),
                    "fraud_pool:", count_storage(storage, "fraud_pool"),
                )

            if (
                count_storage(storage, "if_train_normal") >= args.if_train_normal_rows
                and count_storage(storage, "normal_pool") >= args.normal_pool_rows
                and count_storage(storage, "fraud_pool") >= args.fraud_pool_rows
            ):
                print("Required pools reached.")
                break

        if (
            count_storage(storage, "if_train_normal") >= args.if_train_normal_rows
            and count_storage(storage, "normal_pool") >= args.normal_pool_rows
            and count_storage(storage, "fraud_pool") >= args.fraud_pool_rows
        ):
            break

    if_train_df = pd.concat(storage["if_train_normal"], ignore_index=True)
    normal_pool = pd.concat(storage["normal_pool"], ignore_index=True)
    fraud_pool = pd.concat(storage["fraud_pool"], ignore_index=True)

    print("\nCollected:")
    print("IF train normal:", len(if_train_df))
    print("Normal pool:", len(normal_pool))
    print("Fraud pool:", len(fraud_pool))

    if len(fraud_pool) < 500:
        raise RuntimeError(
            f"Not enough fraud rows collected for train/valid/test. Fraud rows collected: {len(fraud_pool)}"
        )

    # Train Isolation Forest using normal transactions only
    scaler_if = StandardScaler()
    X_if_train = scaler_if.fit_transform(if_train_df[feature_cols].to_numpy(dtype=float))

    iso = IsolationForest(
        n_estimators=100,
        max_samples=min(10000, len(X_if_train)),
        contamination=0.10,
        random_state=args.random_state,
        n_jobs=-1,
    )

    iso.fit(X_if_train)

    pool = pd.concat([normal_pool, fraud_pool], ignore_index=True)

    X_pool_if = scaler_if.transform(pool[feature_cols].to_numpy(dtype=float))
    pool["if_anomaly_score"] = -iso.decision_function(X_pool_if)

    # Select most anomalous examples from each class
    needed_per_class = args.train_per_class + args.valid_per_class + args.test_per_class

    selected_parts = []

    for label in [0, 1]:
        sub = pool[pool["label_clean"] == label].copy()
        sub = sub.sort_values("if_anomaly_score", ascending=False)
        selected_parts.append(sub.head(needed_per_class))

    selected = pd.concat(selected_parts, ignore_index=True)

    if selected[selected["label_clean"] == 0].shape[0] < needed_per_class:
        raise RuntimeError("Not enough normal selected rows.")

    if selected[selected["label_clean"] == 1].shape[0] < needed_per_class:
        raise RuntimeError("Not enough fraud selected rows.")

    # Create balanced train/valid/test split by class
    final_parts = []

    for label in [0, 1]:
        sub = selected[selected["label_clean"] == label].copy()
        sub = sub.sample(frac=1, random_state=args.random_state + label).reset_index(drop=True)

        train = sub.iloc[:args.train_per_class].copy()
        valid = sub.iloc[args.train_per_class:args.train_per_class + args.valid_per_class].copy()
        test = sub.iloc[
            args.train_per_class + args.valid_per_class:
            args.train_per_class + args.valid_per_class + args.test_per_class
        ].copy()

        train["split_clean"] = "train"
        valid["split_clean"] = "valid"
        test["split_clean"] = "test"

        final_parts.extend([train, valid, test])

    final_df = pd.concat(final_parts, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    X_raw = final_df[feature_cols].to_numpy(dtype=float)
    y = final_df["label_clean"].to_numpy(dtype=int)
    split = final_df["split_clean"].to_numpy(dtype=str)
    sample_weight = np.ones(len(final_df), dtype=float)

    train_mask = split == "train"

    scaler_q = StandardScaler()
    X_train_scaled = scaler_q.fit_transform(X_raw[train_mask])
    X_all_scaled = scaler_q.transform(X_raw)

    pca = PCA(n_components=args.n_qubits, random_state=args.random_state)
    pca.fit(X_train_scaled)

    X_pca = pca.transform(X_all_scaled)

    mm = MinMaxScaler(feature_range=(0, np.pi))
    mm.fit(pca.transform(X_train_scaled))

    X_quantum = mm.transform(X_pca)

    dataset_path = out_data / "phase7_if_quantum_dataset_4q.npz"

    np.savez_compressed(
        dataset_path,
        X=X_quantum,
        y=y,
        split=split,
        sample_weight=sample_weight,
        feature_cols=np.array(feature_cols, dtype=object),
        if_anomaly_score=final_df["if_anomaly_score"].to_numpy(dtype=float),
    )

    final_df.to_csv(out_data / "phase7_if_candidate_quantum_subset_source.csv", index=False)

    counts_df = (
        pd.DataFrame({"split": split, "label": y})
        .groupby(["split", "label"])
        .size()
        .reset_index(name="rows")
    )

    counts_df.to_csv(out_reports / "phase7_if_quantum_subset_counts.csv", index=False)

    save_json(
        {
            "dataset_path": str(dataset_path),
            "source": "Directly sampled from Kaggle AML transaction CSVs after session reset.",
            "stage1_model": "Isolation Forest",
            "stage2_models_planned": [
                "Quantum Kernel SVC Statevector",
                "VQC Statevector",
            ],
            "rows_seen": int(rows_seen),
            "chunks_seen": int(chunks_seen),
            "if_train_normal_rows": int(len(if_train_df)),
            "normal_pool_rows": int(len(normal_pool)),
            "fraud_pool_rows": int(len(fraud_pool)),
            "selected_rows": int(len(final_df)),
            "n_qubits": int(args.n_qubits),
            "feature_cols": feature_cols,
            "note": "This Phase 7 dataset uses an Isolation Forest-ranked candidate subset, followed by a balanced train/valid/test quantum-compatible split.",
        },
        out_reports / "phase7_if_quantum_dataset_metadata.json",
    )

    print("\nSaved:", dataset_path)
    print(counts_df)
    print("\nMetadata saved.")


if __name__ == "__main__":
    main()
