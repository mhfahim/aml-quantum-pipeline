import argparse
from pathlib import Path
import time
import numpy as np
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
    write_rows_csv,
    save_json,
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
        "novel_logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1),
        "novel_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=42,
        ),
        "novel_random_forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=14,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def add_proxy_typology(df):
    df = df.copy()

    train = df[df["_split"] == "train"]

    q1 = train["log_amount_paid_num"].quantile(0.50)
    q2 = train["log_amount_paid_num"].quantile(0.80)
    q3 = train["log_amount_paid_num"].quantile(0.95)

    bins = [-np.inf, q1, q2, q3, np.inf]
    labels = ["low", "medium", "high", "extreme"]

    df["amount_bin"] = pd.cut(
        df["log_amount_paid_num"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)

    df["currency_pattern"] = np.where(df["same_currency_num"] >= 0.5, "same_currency", "cross_currency")
    df["payment_format_clean"] = df["payment_format"].astype(str).str.replace(" ", "_", regex=False)

    df["proxy_typology"] = (
        df["payment_format_clean"] + "__" +
        df["currency_pattern"] + "__" +
        df["amount_bin"]
    )

    return df


def choose_novel_typology(df, min_pos=50, min_neg=20):
    test = df[df["_split"] == "test"].copy()

    stats = (
        test.groupby("proxy_typology")
        .agg(
            n=("proxy_typology", "size"),
            positives=("_label", "sum"),
        )
        .reset_index()
    )

    stats["negatives"] = stats["n"] - stats["positives"]

    passing = stats[(stats["positives"] >= min_pos) & (stats["negatives"] >= min_neg)].copy()

    if len(passing) > 0:
        passing = passing.sort_values(["positives", "n"], ascending=False)
        return passing.iloc[0]["proxy_typology"], passing

    fallback = stats.sort_values(["positives", "n"], ascending=False)
    return fallback.iloc[0]["proxy_typology"], fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-data", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--recall-target", type=float, default=0.80)
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.phase4_data)
    df["_label"] = df["_label"].astype(int)

    df = add_proxy_typology(df)

    novel_typology, typology_stats = choose_novel_typology(df)

    print("Selected novel typology:", novel_typology)
    print(typology_stats.head(20))

    typology_stats.to_csv(out_reports / "phase5_novel_typology_candidates.csv", index=False)

    train_df = df[(df["_split"] == "train") & (df["proxy_typology"] != novel_typology)].copy()
    valid_df = df[(df["_split"] == "valid") & (df["proxy_typology"] != novel_typology)].copy()

    test_novel = df[(df["_split"] == "test") & (df["proxy_typology"] == novel_typology)].copy()
    test_seen = df[(df["_split"] == "test") & (df["proxy_typology"] != novel_typology)].copy()

    if len(test_novel) == 0:
        raise ValueError("Novel test subset is empty.")

    models = build_models()

    rows = []
    runtime_rows = []

    for model_name, clf in models.items():
        print(f"Training {model_name}")

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

        start_train = time.perf_counter()
        pipe.fit(X_train, y_train, classifier__sample_weight=w_train)
        train_time = time.perf_counter() - start_train

        valid_scores = pipe.predict_proba(X_valid)[:, 1]

        threshold, policy = choose_threshold_for_recall(
            y_valid,
            valid_scores,
            sample_weight=w_valid,
            recall_target=args.recall_target,
        )

        for subset_name, subset_df in [
            ("test_novel_typology", test_novel),
            ("test_seen_typologies", test_seen),
        ]:
            X = subset_df[ALL_FEATURES]
            y = subset_df["_label"].astype(int).to_numpy()
            w = subset_df["sample_weight"].astype(float).to_numpy()

            start_pred = time.perf_counter()
            scores = pipe.predict_proba(X)[:, 1]
            pred_time = time.perf_counter() - start_pred

            row = binary_metrics_from_scores(
                y,
                scores,
                threshold=threshold,
                sample_weight=w,
                model_name=model_name,
                split=subset_name,
                track="phase5_novel_pattern_stress_test",
            )

            row["novel_typology"] = novel_typology
            row["threshold_policy"] = policy
            row["recall_target"] = args.recall_target
            row["prediction_time_seconds"] = pred_time
            rows.append(row)

        runtime_rows.append({
            "model": model_name,
            "training_time_seconds": train_time,
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_novel_rows": int(len(test_novel)),
            "test_seen_rows": int(len(test_seen)),
        })

    write_rows_csv(rows, out_reports / "phase5_novel_pattern_results.csv")
    write_rows_csv(runtime_rows, out_reports / "phase5_novel_pattern_runtime.csv")

    save_json(
        {
            "experiment": "novel_pattern_stress_test",
            "selected_novel_typology": novel_typology,
            "train_policy": "selected proxy typology excluded from train and validation",
            "test_policy": "models evaluated separately on novel typology and seen typologies",
            "recall_target": args.recall_target,
            "proxy_typology_definition": "payment_format + currency_pattern + amount_bin",
        },
        out_reports / "phase5_novel_pattern_metadata.json",
    )

    print("Novel pattern stress test complete.")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
