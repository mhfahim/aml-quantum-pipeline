import argparse
from pathlib import Path
import time
import psutil
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from phase4_common import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    binary_metrics_from_scores,
    choose_threshold_for_recall,
    read_json,
    save_json,
    write_rows_csv,
    get_stage1b_screening_row,
    estimate_full_hybrid_metrics,
)


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models():
    return {
        "classical_logistic_regression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs",
        ),
        "classical_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=42,
        ),
        "classical_random_forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=14,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-data", required=True)
    parser.add_argument("--phase4-metadata", required=True)
    parser.add_argument("--phase3b-screening", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--out-models", required=True)
    parser.add_argument("--recall-target", type=float, default=0.90)
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_models = Path(args.out_models)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    meta = read_json(args.phase4_metadata)
    target_pct = float(meta["candidate_manifest"]["selected_target_candidate_pct"])

    df = pd.read_parquet(args.phase4_data)

    train_df = df[df["_split"] == "train"].copy()
    valid_df = df[df["_split"] == "valid"].copy()
    test_df = df[df["_split"] == "test"].copy()

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test:", test_df.shape)

    models = build_models()

    metric_rows = []
    full_hybrid_rows = []
    runtime_rows = []

    stage1_test_row = get_stage1b_screening_row(
        args.phase3b_screening,
        split="test",
        target_pct=target_pct,
    )

    for model_name, clf in models.items():
        print(f"\nTraining {model_name}")

        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])

        X_train = train_df[ALL_FEATURES]
        y_train = train_df["_label"].astype(int).to_numpy()
        w_train = train_df["sample_weight"].astype(float).to_numpy()

        X_valid = valid_df[ALL_FEATURES]
        y_valid = valid_df["_label"].astype(int).to_numpy()
        w_valid = valid_df["sample_weight"].astype(float).to_numpy()

        X_test = test_df[ALL_FEATURES]
        y_test = test_df["_label"].astype(int).to_numpy()
        w_test = test_df["sample_weight"].astype(float).to_numpy()

        start_train = time.perf_counter()
        pipe.fit(X_train, y_train, classifier__sample_weight=w_train)
        train_time = time.perf_counter() - start_train

        start_valid = time.perf_counter()
        valid_scores = pipe.predict_proba(X_valid)[:, 1]
        valid_inference_time = time.perf_counter() - start_valid

        threshold, threshold_policy = choose_threshold_for_recall(
            y_valid,
            valid_scores,
            sample_weight=w_valid,
            recall_target=args.recall_target,
        )

        for split_name, split_df in [
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ]:
            X = split_df[ALL_FEATURES]
            y = split_df["_label"].astype(int).to_numpy()
            w = split_df["sample_weight"].astype(float).to_numpy()

            start_pred = time.perf_counter()
            scores = pipe.predict_proba(X)[:, 1]
            inference_time = time.perf_counter() - start_pred

            row = binary_metrics_from_scores(
                y,
                scores,
                threshold=threshold,
                sample_weight=w,
                model_name=model_name,
                split=split_name,
                track="candidate_pool_stage2_weighted",
            )
            row["threshold_policy"] = threshold_policy
            row["recall_target"] = args.recall_target
            row["inference_time_seconds"] = inference_time
            row["inference_rows_per_second"] = len(split_df) / inference_time if inference_time > 0 else None
            metric_rows.append(row)

            if split_name == "test":
                full_row = estimate_full_hybrid_metrics(
                    stage1_test_row,
                    row,
                    model_name="phase3b_screener_plus_" + model_name,
                )
                full_hybrid_rows.append(full_row)

        model_path = out_models / f"{model_name}.joblib"
        joblib.dump(pipe, model_path)

        runtime_rows.append({
            "track": "classical_candidate_pool_stage2",
            "model": model_name,
            "training_time_seconds": train_time,
            "validation_inference_time_seconds": valid_inference_time,
            "training_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
            "model_path": str(model_path),
            "available_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 4),
        })

    write_rows_csv(metric_rows, out_reports / "phase4_classical_candidate_metrics.csv")
    write_rows_csv(full_hybrid_rows, out_reports / "phase4_classical_hybrid_estimated_full_metrics.csv")
    write_rows_csv(runtime_rows, out_reports / "phase4_classical_candidate_runtime.csv")

    save_json(
        {
            "recall_target": args.recall_target,
            "models": list(models.keys()),
            "target_candidate_pct_from_phase3b": target_pct,
            "note": "Classical Stage 2 metrics are weighted using the Phase 3B candidate negative sampling fraction. Full hybrid metrics combine Stage 1 screening counts with Stage 2 weighted candidate-pool confusion matrices.",
        },
        out_reports / "phase4_classical_candidate_metadata.json",
    )

    print("Classical candidate-pool training complete.")
    print(pd.DataFrame(metric_rows))
    print(pd.DataFrame(full_hybrid_rows))


if __name__ == "__main__":
    main()
