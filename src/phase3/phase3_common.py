from pathlib import Path
import json
import polars as pl


PHASE3_FEATURES = [
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


BASE_SCAN_COLUMNS = [
    "_ts",
    "_label",
    "_source_file",
    "amount_received",
    "amount_paid",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_split_cutoffs(phase1_reports):
    manifest = read_json(Path(phase1_reports) / "temporal_split_manifest.json")
    return (
        manifest["train_cutoff_raw_microseconds"],
        manifest["valid_cutoff_raw_microseconds"],
    )


def add_temporal_split_df(df, train_cutoff, valid_cutoff):
    ts_int = pl.col("_ts").cast(pl.Int64)

    return df.with_columns(
        pl.when(ts_int <= train_cutoff)
        .then(pl.lit("train"))
        .when(ts_int <= valid_cutoff)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("test"))
        .alias("_split")
    )


def clean_numeric_expr(col_name, new_name):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.replace_all(",", "")
        .str.replace_all(" ", "")
        .cast(pl.Float64, strict=False)
        .alias(new_name)
    )


def add_phase3_features_df(df):
    cols = df.columns

    exprs = []

    if "amount_received" in cols:
        exprs.append(clean_numeric_expr("amount_received", "amount_received_num"))

    if "amount_paid" in cols:
        exprs.append(clean_numeric_expr("amount_paid", "amount_paid_num"))

    if "_ts" in cols:
        exprs.extend([
            pl.col("_ts").dt.hour().cast(pl.Float64).alias("hour_num"),
            pl.col("_ts").dt.weekday().cast(pl.Float64).alias("day_num"),
        ])

    if "receiving_currency" in cols and "payment_currency" in cols:
        exprs.append(
            (pl.col("receiving_currency").cast(pl.Utf8) == pl.col("payment_currency").cast(pl.Utf8))
            .cast(pl.Float64)
            .alias("same_currency_num")
        )

    df = df.with_columns(exprs)

    df = df.with_columns([
        (pl.col("amount_paid_num") - pl.col("amount_received_num")).alias("amount_delta_num"),
        (
            pl.col("amount_paid_num") /
            (pl.col("amount_received_num").abs() + 1.0)
        ).alias("amount_ratio_num"),
        pl.col("amount_received_num").abs().log1p().alias("log_amount_received_num"),
        pl.col("amount_paid_num").abs().log1p().alias("log_amount_paid_num"),
    ])

    for f in PHASE3_FEATURES:
        if f not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f))

    return df


def existing_scan_columns(columns):
    return [c for c in BASE_SCAN_COLUMNS if c in columns]
