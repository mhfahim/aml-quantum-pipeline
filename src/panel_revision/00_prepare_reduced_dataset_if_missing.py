import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_transaction_files():
    files = sorted(Path("/kaggle/input").rglob("*_Trans.csv"))
    return files


def preprocess_chunk(df):
    required = ["Amount Received", "Amount Paid", "Is Laundering"]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Existing columns: {list(df.columns)[:20]}")

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
    parser.add_argument("--out-data", default="/kaggle/working/aml_phase7/data")
    parser.add_argument("--out-reports", default="reports/panel_revision")
    parser.add_argument("--train-per-class", type=int, default=250)
    parser.add_argument("--valid-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--normal-pool", type=int, default=20000)
    parser.add_argument("--fraud-pool", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_data = Path(args.out_data)
    out_reports = Path(args.out_reports)

    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    dataset_path = out_data / "phase7_if_quantum_dataset_4q.npz"

    if dataset_path.exists():
        print("Dataset already exists:", dataset_path)
        data = np.load(dataset_path, allow_pickle=True)
        print("Keys:", list(data.keys()))
        print("Rows:", len(data["y"]))
        return

    files = find_transaction_files()

    if not files:
        raise FileNotFoundError(
            "No *_Trans.csv files found under /kaggle/input. Add the IBM AML dataset to the Kaggle notebook input first."
        )

    normal_rows = []
    fraud_rows = []

    for f in files:
        print("Reading:", f)

        for chunk in pd.read_csv(f, chunksize=300000):
            data = preprocess_chunk(chunk)

            normal = data[data["label"] == 0]
            fraud = data[data["label"] == 1]

            if sum(len(x) for x in normal_rows) < args.normal_pool and len(normal) > 0:
                need = args.normal_pool - sum(len(x) for x in normal_rows)
                normal_rows.append(normal.sample(min(need, len(normal)), random_state=args.random_state))

            if sum(len(x) for x in fraud_rows) < args.fraud_pool and len(fraud) > 0:
                need = args.fraud_pool - sum(len(x) for x in fraud_rows)
                fraud_rows.append(fraud.sample(min(need, len(fraud)), random_state=args.random_state))

            if sum(len(x) for x in normal_rows) >= args.normal_pool and sum(len(x) for x in fraud_rows) >= args.fraud_pool:
                break

        if sum(len(x) for x in normal_rows) >= args.normal_pool and sum(len(x) for x in fraud_rows) >= args.fraud_pool:
            break

    normal_pool = pd.concat(normal_rows, ignore_index=True)
    fraud_pool = pd.concat(fraud_rows, ignore_index=True)

    n_train = args.train_per_class
    n_valid = args.valid_per_class
    n_test = args.test_per_class
    n_each = n_train + n_valid + n_test

    if len(normal_pool) < n_each or len(fraud_pool) < n_each:
        raise ValueError(f"Not enough rows. Need {n_each} per class. Got normal={len(normal_pool)}, fraud={len(fraud_pool)}.")

    normal_sel = normal_pool.sample(n_each, random_state=args.random_state).reset_index(drop=True)
    fraud_sel = fraud_pool.sample(n_each, random_state=args.random_state).reset_index(drop=True)

    normal_sel["label"] = 0
    fraud_sel["label"] = 1

    normal_sel["split"] = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    fraud_sel["split"] = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test

    final = pd.concat([normal_sel, fraud_sel], ignore_index=True)
    final = final.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    selected_features = [
        "log_amount_received",
        "log_amount_paid",
        "amount_delta",
        "amount_ratio",
    ]

    X_raw = final[selected_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    x_min = X_scaled.min(axis=0)
    x_max = X_scaled.max(axis=0)

    X_angles = 2 * np.pi * ((X_scaled - x_min) / (x_max - x_min + 1e-9)) - np.pi

    X = X_angles.astype(np.float64)
    y = final["label"].astype(int).to_numpy()
    split = final["split"].astype(str).to_numpy()

    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        split=split,
        feature_names=np.array(selected_features),
    )

    counts = final.groupby(["split", "label"]).size().reset_index(name="count")
    counts.to_csv(out_reports / "sota_reduced_dataset_counts.csv", index=False)

    metadata = {
        "dataset_path": str(dataset_path),
        "total_rows": int(len(final)),
        "train_per_class": int(n_train),
        "valid_per_class": int(n_valid),
        "test_per_class": int(n_test),
        "selected_features": selected_features,
        "random_state": int(args.random_state),
        "note": "Reduced four-feature dataset for transformer-based tabular baseline comparison."
    }

    with open(out_reports / "sota_reduced_dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:", dataset_path)
    print(counts)


if __name__ == "__main__":
    main()
