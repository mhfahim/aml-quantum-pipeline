import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from phase6_common import save_json


def get_split(data, split_name):
    mask = data["split"] == split_name
    return {
        "X": data["X"][mask],
        "y": data["y"][mask].astype(int),
        "sample_weight": data["sample_weight"][mask].astype(float),
        "stage1b_score": data["stage1b_score"][mask].astype(float),
        "split": data["split"][mask].astype(str),
    }


def stratified_sample(split_data, n_pos, n_neg, seed):
    rng = np.random.default_rng(seed)

    y = split_data["y"]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = min(n_pos, len(pos_idx))
    n_neg = min(n_neg, len(neg_idx))

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)

    idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx)

    return {
        "X": split_data["X"][idx],
        "y": split_data["y"][idx],
        "sample_weight": split_data["sample_weight"][idx],
        "stage1b_score": split_data["stage1b_score"][idx],
        "split": split_data["split"][idx],
    }


def combine(parts):
    return {
        "X": np.concatenate([p["X"] for p in parts], axis=0),
        "y": np.concatenate([p["y"] for p in parts], axis=0),
        "sample_weight": np.concatenate([p["sample_weight"] for p in parts], axis=0),
        "stage1b_score": np.concatenate([p["stage1b_score"] for p in parts], axis=0),
        "split": np.concatenate([p["split"] for p in parts], axis=0),
    }


def count_rows(data):
    rows = []

    for split in ["train", "valid", "test"]:
        for label in [0, 1]:
            mask = (data["split"] == split) & (data["y"] == label)
            rows.append({
                "split": split,
                "label": label,
                "n": int(mask.sum()),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-quantum-dataset", required=True)
    parser.add_argument("--out-data", required=True)
    parser.add_argument("--out-reports", required=True)
    parser.add_argument("--train-per-class", type=int, default=250)
    parser.add_argument("--valid-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_data = Path(args.out_data)
    out_reports = Path(args.out_reports)

    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.phase4_quantum_dataset, allow_pickle=True)

    data = {
        "X": raw["X"],
        "y": raw["y"].astype(int),
        "sample_weight": raw["sample_weight"].astype(float),
        "split": raw["split"].astype(str),
        "stage1b_score": raw["stage1b_score"].astype(float),
    }

    train = stratified_sample(
        get_split(data, "train"),
        args.train_per_class,
        args.train_per_class,
        args.random_state,
    )

    valid = stratified_sample(
        get_split(data, "valid"),
        args.valid_per_class,
        args.valid_per_class,
        args.random_state + 1,
    )

    test = stratified_sample(
        get_split(data, "test"),
        args.test_per_class,
        args.test_per_class,
        args.random_state + 2,
    )

    phase6_data = combine([train, valid, test])

    out_npz = out_data / "phase6_hardware_subset_4q.npz"

    np.savez_compressed(
        out_npz,
        X=phase6_data["X"],
        y=phase6_data["y"],
        sample_weight=phase6_data["sample_weight"],
        split=phase6_data["split"],
        stage1b_score=phase6_data["stage1b_score"],
    )

    counts = count_rows(phase6_data)
    pd.DataFrame(counts).to_csv(out_reports / "phase6_subset_counts.csv", index=False)

    metadata = {
        "phase6_subset_path": str(out_npz),
        "n_qubits": int(phase6_data["X"].shape[1]),
        "total_rows": int(len(phase6_data["y"])),
        "counts": counts,
        "train_per_class": args.train_per_class,
        "valid_per_class": args.valid_per_class,
        "test_per_class": args.test_per_class,
        "random_state": args.random_state,
        "purpose": "Fixed small balanced subset for IBM Quantum hardware validation.",
    }

    save_json(metadata, out_reports / "phase6_subset_metadata.json")

    print("Phase 6 subset prepared.")
    print(pd.DataFrame(counts))
    print(metadata)


if __name__ == "__main__":
    main()
