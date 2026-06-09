import argparse
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from phase5_common import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    binary_metrics_from_scores,
    choose_threshold_for_recall,
    read_json,
    write_rows_csv,
    save_json,
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
        "classical_logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1),
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
    parser.add_argument("--phase4-models", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--out-models", required=True)
    parser.add_argument("--recall-targets", default="0.80,0.90,0.95")
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_models = Path(args.out_models)
    phase4_models = Path(args.phase4_models)

    out_reports.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    meta = read_json(args.phase4_metadata)
    target_pct = float(meta["candidate_manifest"]["selected_target_candidate_pct"])

    df = pd.read_parquet(args.phase4_data)

    train_df = df[df["_split"] == "train"].copy()
    valid_df = df[df["_split"] == "valid"].copy()
    test_df = df[df["_split"] == "test"].copy()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["_label"].astype(int).to_numpy()
    w_train = train_df["sample_weight"].astype(float).to_numpy()

    X_valid = valid_df[ALL_FEATURES]
    y_valid = valid_df["_label"].astype(int).to_numpy()
    w_valid = valid_df["sample_weight"].astype(float).to_numpy()

    X_test = test_df[ALL_FEATURES]
    y_test = test_df["_label"].astype(int).to_numpy()
    w_test = test_df["sample_weight"].astype(float).to_numpy()

    recall_targets = [float(x.strip()) for x in args.recall_targets.split(",")]

    models = build_models()

    candidate_rows = []
    full_hybrid_rows = []
    runtime_rows = []

    stage1_test_row = get_stage1b_screening_row(
        args.phase3b_screening,
        split="test",
        target_pct=target_pct,
    )

    for model_name, clf in models.items():
        print(f"Model: {model_name}")

        phase4_model_path = phase4_models / f"{model_name}.joblib"

        if phase4_model_path.exists():
            print(f"Loading existing Phase 4 model: {phase4_model_path}")
            pipe = joblib.load(phase4_model_path)
            training_time = 0.0
            training_note = "loaded_existing_phase4_model"
        else:
            print("Training model because Phase 4 model file was not found.")
            pipe = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ])

            start_train = time.perf_counter()
            pipe.fit(X_train, y_train, classifier__sample_weight=w_train)
            training_time = time.perf_counter() - start_train
            training_note = "trained_in_phase5"

            joblib.dump(pipe, out_models / f"{model_name}.joblib")

        start_valid = time.perf_counter()
        valid_scores = pipe.predict_proba(X_valid)[:, 1]
        valid_time = time.perf_counter() - start_valid

        start_test = time.perf_counter()
        test_scores = pipe.predict_proba(X_test)[:, 1]
        test_time = time.perf_counter() - start_test

        for recall_target in recall_targets:
            threshold, policy = choose_threshold_for_recall(
                y_valid,
                valid_scores,
                sample_weight=w_valid,
                recall_target=recall_target,
            )

            row = binary_metrics_from_scores(
                y_test,
                test_scores,
                threshold=threshold,
                sample_weight=w_test,
                model_name=model_name,
                split="test",
                track="phase5_fixed_recall_candidate_pool",
            )

            row["recall_target"] = recall_target
            row["threshold_policy"] = policy
            row["valid_inference_time_seconds"] = valid_time
            row["test_inference_time_seconds"] = test_time
            row["training_time_seconds"] = training_time
            row["training_note"] = training_note
            candidate_rows.append(row)

            full_row = estimate_full_hybrid_metrics(
                stage1_test_row,
                row,
                model_name="phase3b_screener_plus_" + model_name,
                recall_target=recall_target,
            )

            full_hybrid_rows.append(full_row)

        runtime_rows.append({
            "model": model_name,
            "training_time_seconds": training_time,
            "training_note": training_note,
            "valid_inference_time_seconds": valid_time,
            "test_inference_time_seconds": test_time,
            "test_rows": int(len(test_df)),
            "test_rows_per_second": len(test_df) / test_time if test_time > 0 else None,
        })

    write_rows_csv(candidate_rows, out_reports / "phase5_fixed_recall_candidate_pool.csv")
    write_rows_csv(full_hybrid_rows, out_reports / "phase5_fixed_recall_estimated_full_hybrid.csv")
    write_rows_csv(runtime_rows, out_reports / "phase5_fixed_recall_runtime.csv")

    save_json(
        {
            "experiment": "fixed_recall_false_positive_tradeoff",
            "recall_targets": recall_targets,
            "stage1b_target_candidate_pct": target_pct,
            "models": list(models.keys()),
            "note": "Thresholds are selected on validation data to hit target recall; test performance is reported.",
        },
        out_reports / "phase5_fixed_recall_metadata.json",
    )

    print("Fixed-recall tradeoff complete.")
    print(pd.DataFrame(full_hybrid_rows))


if __name__ == "__main__":
    main()
