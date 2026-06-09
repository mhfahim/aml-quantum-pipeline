import argparse
from pathlib import Path
import time
import joblib
import pyarrow.dataset as ds
import polars as pl
import psutil

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from phase3b_common import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    read_split_cutoffs,
    add_temporal_split_df,
    add_phase3b_features_df,
    existing_scan_columns,
    save_json,
)


def build_model():
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

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--phase1-reports", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--max-pos-rows", type=int, default=180000)
    parser.add_argument("--max-neg-rows", type=int, default=600000)
    parser.add_argument("--neg-sample-frac", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=200000)
    args = parser.parse_args()

    start = time.perf_counter()

    parquet_dir = Path(args.parquet)
    out_dir = Path(args.out)
    model_dir = Path(args.model_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_cutoff, valid_cutoff = read_split_cutoffs(args.phase1_reports)

    dataset = ds.dataset(str(parquet_dir), format="parquet", partitioning="hive")
    scan_cols = existing_scan_columns(dataset.schema.names)

    frames = []
    pos_rows = 0
    neg_rows = 0
    seen_rows = 0
    seen_batches = 0

    for batch in dataset.to_batches(columns=scan_cols, batch_size=args.batch_size):
        if batch.num_rows == 0:
            continue

        seen_batches += 1
        seen_rows += batch.num_rows

        df = pl.from_arrow(batch)
        df = add_temporal_split_df(df, train_cutoff, valid_cutoff)
        df = df.filter(pl.col("_split") == "train")

        if df.height == 0:
            continue

        df = df.with_columns(
            pl.struct(["_ts", "_label", "amount_received", "amount_paid", "_source_file"])
            .hash(seed=314)
            .alias("_sample_hash")
        )

        pos_df = df.filter(pl.col("_label") == 1)

        neg_df = (
            df
            .filter(pl.col("_label") == 0)
            .filter((pl.col("_sample_hash") % 1_000_000) < int(args.neg_sample_frac * 1_000_000))
        )

        remaining_pos = args.max_pos_rows - pos_rows
        remaining_neg = args.max_neg_rows - neg_rows

        if remaining_pos <= 0:
            pos_df = pos_df.head(0)
        elif pos_df.height > remaining_pos:
            pos_df = pos_df.head(remaining_pos)

        if remaining_neg <= 0:
            neg_df = neg_df.head(0)
        elif neg_df.height > remaining_neg:
            neg_df = neg_df.head(remaining_neg)

        keep_df = pl.concat([pos_df, neg_df], how="vertical")

        if keep_df.height > 0:
            keep_df = add_phase3b_features_df(keep_df)
            keep_df = keep_df.select(ALL_FEATURES + ["_label"])
            frames.append(keep_df)

            pos_rows += int((keep_df["_label"] == 1).sum())
            neg_rows += int((keep_df["_label"] == 0).sum())

        if seen_batches % 50 == 0:
            print(f"Seen rows: {seen_rows:,}, positives: {pos_rows:,}, negatives: {neg_rows:,}")

        if pos_rows >= args.max_pos_rows and neg_rows >= args.max_neg_rows:
            break

    if not frames:
        raise ValueError("No training data collected.")

    train_df = pl.concat(frames, how="vertical")

    print("Training dataframe shape:", train_df.shape)
    print("Positive rows:", int((train_df["_label"] == 1).sum()))
    print("Negative rows:", int((train_df["_label"] == 0).sum()))

    train_pd = train_df.to_pandas()

    X_train = train_pd[ALL_FEATURES]
    y_train = train_pd["_label"].astype(int)

    model = build_model()
    model.fit(X_train, y_train)

    model_path = model_dir / "stage1b_high_recall_filter.joblib"
    joblib.dump(model, model_path)

    runtime = time.perf_counter() - start

    metadata = {
        "model_name": "Stage1B High-Recall Classical Screening Filter",
        "model_type": "logistic_regression_balanced",
        "purpose": "Improved Stage 1 candidate generation for AML hybrid pipeline",
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_data": "temporal train split; sampled laundering positives plus sampled legitimate negatives",
        "max_pos_rows": args.max_pos_rows,
        "max_neg_rows": args.max_neg_rows,
        "actual_pos_rows": int((train_df["_label"] == 1).sum()),
        "actual_neg_rows": int((train_df["_label"] == 0).sum()),
        "total_training_rows": int(train_df.height),
        "neg_sample_frac": args.neg_sample_frac,
        "seen_rows_before_stop": int(seen_rows),
        "seen_batches_before_stop": int(seen_batches),
        "model_path": str(model_path),
        "runtime_seconds": runtime,
        "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
        "note": "Training used streaming batch sampling. The full dataset was not loaded into memory.",
    }

    save_json(metadata, out_dir / "stage1b_model_metadata.json")

    print("Stage 3B model trained successfully.")
    print(metadata)


if __name__ == "__main__":
    main()
