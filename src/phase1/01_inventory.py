import argparse
from pathlib import Path
import polars as pl

from common import normalize_columns, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Dataset folder path")
    parser.add_argument("--out", required=True, help="Report output folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.rglob("*.csv"))

    records = []

    for file in csv_files:
        try:
            sample = pl.read_csv(
                file,
                n_rows=5,
                infer_schema_length=100,
                ignore_errors=True,
                try_parse_dates=False,
            )

            sample = normalize_columns(sample)

            records.append({
                "file_name": file.name,
                "path": str(file),
                "size_gb": round(file.stat().st_size / (1024 ** 3), 4),
                "n_columns": len(sample.columns),
                "columns": ", ".join(sample.columns),
                "status": "success"
            })

        except Exception as e:
            records.append({
                "file_name": file.name,
                "path": str(file),
                "size_gb": round(file.stat().st_size / (1024 ** 3), 4),
                "n_columns": None,
                "columns": None,
                "status": "failed",
                "error": str(e)
            })

    df = pl.DataFrame(records)

    df.write_csv(out_dir / "dataset_inventory.csv")

    save_json(
        {
            "n_csv_files": len(csv_files),
            "total_csv_size_gb": round(
                sum(f.stat().st_size for f in csv_files) / (1024 ** 3), 4
            ),
            "files": records,
        },
        out_dir / "dataset_inventory.json",
    )

    print("Dataset inventory complete.")
    print(df)


if __name__ == "__main__":
    main()
