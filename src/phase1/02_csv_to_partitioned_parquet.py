import argparse
from pathlib import Path
import polars as pl

from common import (
    normalize_columns,
    find_col,
    TIMESTAMP_CANDIDATES,
    LABEL_CANDIDATES,
    safe_name,
    parse_timestamp_expr,
    parse_label_expr,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Dataset folder path")
    parser.add_argument("--out", required=True, help="Parquet output folder")
    parser.add_argument("--reports", required=True, help="Report output folder")
    parser.add_argument("--batch-size", type=int, default=250000)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    report_dir = Path(args.reports)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.rglob("*.csv"))
    conversion_log = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            reader = pl.read_csv_batched(
                csv_file,
                batch_size=args.batch_size,
                infer_schema_length=0,
                ignore_errors=True,
                try_parse_dates=False,
                low_memory=True,
            )

            source_safe = safe_name(csv_file.stem)
            part_id = 0
            total_rows = 0

            while True:
                batches = reader.next_batches(1)

                if not batches:
                    break

                for df in batches:
                    df = normalize_columns(df)

                    cols = df.columns
                    ts_col = find_col(cols, TIMESTAMP_CANDIDATES)
                    label_col = find_col(cols, LABEL_CANDIDATES)

                    if ts_col is None:
                        raise ValueError(f"No timestamp column found in {csv_file.name}. Columns: {cols}")

                    if label_col is None:
                        raise ValueError(f"No label column found in {csv_file.name}. Columns: {cols}")

                    df = df.with_columns([
                        pl.lit(csv_file.name).alias("_source_file"),
                        parse_timestamp_expr(ts_col).alias("_ts"),
                        parse_label_expr(label_col),
                    ])

                    df = df.with_columns(
                        pl.when(pl.col("_ts").is_not_null())
                        .then(pl.col("_ts").dt.strftime("%Y-%m"))
                        .otherwise(pl.lit("unknown"))
                        .alias("_partition_month")
                    )

                    partitions = df.partition_by("_partition_month", as_dict=True)

                    for month_key, part_df in partitions.items():
                        if isinstance(month_key, tuple):
                            month = month_key[0]
                        else:
                            month = month_key

                        save_dir = out_dir / f"source_file={source_safe}" / f"month={month}"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        save_path = save_dir / f"part_{part_id:08d}.parquet"

                        part_df.drop("_partition_month").write_parquet(
                            save_path,
                            compression=args.compression,
                            statistics=True,
                        )

                        part_id += 1
                        total_rows += part_df.height

            conversion_log.append({
                "file": csv_file.name,
                "status": "success",
                "rows_written": total_rows,
                "parts_written": part_id,
            })

        except Exception as e:
            conversion_log.append({
                "file": csv_file.name,
                "status": "failed",
                "error": str(e),
            })
            print(f"FAILED: {csv_file.name} -> {e}")

    save_json(conversion_log, report_dir / "conversion_log.json")

    print("\nCSV to Parquet conversion complete.")


if __name__ == "__main__":
    main()
