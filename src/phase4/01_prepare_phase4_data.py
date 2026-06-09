import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import polars as pl
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from phase4_common import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    read_json,
    save_json,
)


def sample_split_class(df, split, label, n, random_state):
    part = df[(df["_split"] == split) & (df["_label"] == label)]
    if len(part) <= n:
        return part
    return part.sample(n=n, random_state=random_state)


def make_quantum_subset(df, q_train_pos, q_train_neg, q_valid_pos, q_valid_neg, q_test_pos, q_test_neg, random_state):
    pieces = [
        sample_split_class(df, "train", 1, q_train_pos, random_state),
        sample_split_class(df, "train", 0, q_train_neg, random_state),
        sample_split_class(df, "valid", 1, q_valid_pos, random_state),
        sample_split_class(df, "valid", 0, q_valid_neg, random_state),
        sample_split_class(df, "test", 1, q_test_pos, random_state),
        sample_split_class(df, "test", 0, q_test_neg, random_state),
    ]

    out = pd.concat(pieces, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--q-train-pos", type=int, default=800)
    parser.add_argument("--q-train-neg", type=int, default=800)
    parser.add_argument("--q-valid-pos", type=int, default=400)
    parser.add_argument("--q-valid-neg", type=int, default=400)
    parser.add_argument("--q-test-pos", type=int, default=500)
    parser.add_argument("--q-test-neg", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    out_data = Path(args.out_data)
    out_reports = Path(args.out_reports)

    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    parquet_files = list(candidate_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No candidate parquet files found in {candidate_dir}")

    manifest = read_json(args.candidate_manifest)
    neg_frac = float(manifest["negative_sample_fraction_inside_flagged_pool"])

    print(f"Reading candidate sample from {candidate_dir}")
    print(f"Parquet parts: {len(parquet_files)}")
    print(f"Negative sampling fraction: {neg_frac}")

    lf = pl.scan_parquet(str(candidate_dir / "*.parquet"))
    df = lf.collect(engine="streaming").to_pandas()

    for c in ALL_FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing required feature: {c}")

    df["_label"] = df["_label"].astype(int)
    df["sample_weight"] = np.where(df["_label"] == 0, 1.0 / neg_frac, 1.0)

    candidate_path = out_data / "phase4_candidate_sample.parquet"
    df.to_parquet(candidate_path, index=False)

    counts = (
        df.groupby(["_split", "_label"])
        .size()
        .reset_index(name="n")
        .sort_values(["_split", "_label"])
    )
    counts.to_csv(out_reports / "phase4_candidate_sample_counts.csv", index=False)

    # Quantum-compatible reduced feature preparation
    train_df = df[df["_split"] == "train"].copy()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    pca = PCA(n_components=args.n_qubits, random_state=args.random_state)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    Z_train = preprocessor.fit_transform(train_df[ALL_FEATURES])
    Q_train_raw = pca.fit_transform(Z_train)
    angle_scaler.fit(Q_train_raw)

    quantum_preprocess = {
        "preprocessor": preprocessor,
        "pca": pca,
        "angle_scaler": angle_scaler,
        "n_qubits": args.n_qubits,
        "features": ALL_FEATURES,
    }

    joblib.dump(quantum_preprocess, out_data / "phase4_quantum_preprocessor.joblib")

    q_df = make_quantum_subset(
        df,
        args.q_train_pos,
        args.q_train_neg,
        args.q_valid_pos,
        args.q_valid_neg,
        args.q_test_pos,
        args.q_test_neg,
        args.random_state,
    )

    Z_q = preprocessor.transform(q_df[ALL_FEATURES])
    X_q = angle_scaler.transform(pca.transform(Z_q))

    np.savez_compressed(
        out_data / f"phase4_quantum_dataset_{args.n_qubits}q.npz",
        X=X_q.astype(np.float64),
        y=q_df["_label"].astype(int).to_numpy(),
        sample_weight=q_df["sample_weight"].astype(float).to_numpy(),
        split=q_df["_split"].astype(str).to_numpy(),
        stage1b_score=q_df["stage1b_score"].astype(float).to_numpy(),
    )

    q_counts = (
        q_df.groupby(["_split", "_label"])
        .size()
        .reset_index(name="n")
        .sort_values(["_split", "_label"])
    )
    q_counts.to_csv(out_reports / "phase4_quantum_subset_counts.csv", index=False)

    metadata = {
        "candidate_sample_path": str(candidate_path),
        "candidate_rows": int(len(df)),
        "candidate_counts": counts.to_dict("records"),
        "candidate_manifest": manifest,
        "negative_sampling_fraction": neg_frac,
        "sample_weight_rule": "positive=1; negative=1/negative_sampling_fraction",
        "quantum_dataset_path": str(out_data / f"phase4_quantum_dataset_{args.n_qubits}q.npz"),
        "n_qubits": args.n_qubits,
        "quantum_subset_rows": int(len(q_df)),
        "quantum_subset_counts": q_counts.to_dict("records"),
        "quantum_feature_method": "numeric/categorical preprocessing -> PCA -> angle scaling to [-pi, pi]",
    }

    save_json(metadata, out_reports / "phase4_data_metadata.json")

    print("Phase 4 data preparation complete.")
    print(counts)
    print(q_counts)


if __name__ == "__main__":
    main()
