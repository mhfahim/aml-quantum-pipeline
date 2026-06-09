import argparse
from pathlib import Path
import time
import joblib
import polars as pl
import numpy as np
import psutil

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from phase3_common import (
    PHASE3_FEATURES,
    read_split_cutoffs,
    add_temporal_split_df,
    add_phase3_features_df,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--train-normal-frac", type=float, default=0.001)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    model_dir = Path(args.model_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)

    lf = pl.scan_parquet(str(parquet_dir / "**/*.parquet"))

    needed_cols = [
        "_ts", "_label", "_source_file",
        "amount_received", "amount_paid",
        "receiving_currency", "payment_currency",
        "payment_format",
    ]
    existing_cols = [c for c in needed_cols if c in lf.collect_schema().names()]

    df = (
        lf.select(existing_cols)
        .collect(engine="streaming")
    )

    df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
    df = add_phase3_features_df(df)

    df = (
        df
        .filter((pl.col("_split") == "train") & (pl.col("_label") == 0))
        .with_columns(
            pl.struct(["_ts", "amount_received", "amount_paid", "_source_file"])
            .hash(seed=42)
            .alias("_hash")
        )
        .filter((pl.col("_hash") % 1_000_000) < int(args.train_normal_frac * 1_000_000))
        .select(PHASE3_FEATURES)
    )

    print("Training sample shape:", df.shape)

    if df.height < 1000:
        raise ValueError("Training sample too small. Increase --train-normal-frac.")

    X_train = df.to_numpy()

    max_samples = min(args.max_samples, X_train.shape[0])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=args.n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train)

    model_path = model_dir / "stage1_isolation_forest.joblib"
    joblib.dump(model, model_path)

    runtime = time.perf_counter() - start

    metadata = {
        "model": "IsolationForest",
        "purpose": "Stage 1 classical anomaly screening",
        "features": PHASE3_FEATURES,
        "training_data": "train split legitimate transactions only",
        "train_normal_frac": args.train_normal_frac,
        "n_estimators": args.n_estimators,
        "max_samples": max_samples,
        "training_sample_rows": int(X_train.shape[0]),
        "training_sample_features": int(X_train.shape[1]),
        "model_path": str(model_path),
        "runtime_seconds": runtime,
        "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
    }

    save_json(metadata, out_dir / "stage1_model_metadata.json")

    print("Stage 1 Isolation Forest trained.")
    print(metadata)


if __name__ == "__main__":
    main()
