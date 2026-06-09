from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "receiving_currency",
    "payment_currency",
    "payment_format",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_rows_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def safe_auc(y_true, scores, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, scores, sample_weight=sample_weight)


def safe_pr_auc(y_true, scores, sample_weight=None):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, scores, sample_weight=sample_weight)


def binary_metrics_from_scores(
    y_true,
    scores,
    threshold=0.5,
    sample_weight=None,
    model_name="model",
    split="test",
    track="track",
):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    recall = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    f1 = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "track": track,
        "model": model_name,
        "split": split,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_auc(y_true, scores, sample_weight),
        "pr_auc": safe_pr_auc(y_true, scores, sample_weight),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_rows": int(len(y_true)),
    }


def choose_threshold_for_recall(y_true, scores, sample_weight=None, recall_target=0.90):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    thresholds = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 999)))

    best = None

    for th in thresholds:
        row = binary_metrics_from_scores(
            y_true,
            scores,
            threshold=th,
            sample_weight=sample_weight,
        )

        if row["recall"] >= recall_target:
            if best is None:
                best = row
            elif row["precision"] > best["precision"]:
                best = row

    if best is not None:
        return float(best["threshold"]), "recall_target_precision_max"

    best = None

    for th in thresholds:
        row = binary_metrics_from_scores(
            y_true,
            scores,
            threshold=th,
            sample_weight=sample_weight,
        )

        if best is None or row["f1"] > best["f1"]:
            best = row

    return float(best["threshold"]), "fallback_max_f1"


def get_stage1b_screening_row(screening_csv, split, target_pct):
    df = pd.read_csv(screening_csv)
    df["_target_round"] = df["target_candidate_pct"].astype(float).round(6)
    target_pct = round(float(target_pct), 6)

    row = df[(df["split"] == split) & (df["_target_round"] == target_pct)]

    if len(row) == 0:
        raise ValueError(f"No Stage 1B row for split={split}, target_pct={target_pct}")

    return row.iloc[0].to_dict()


def estimate_full_hybrid_metrics(stage1_row, candidate_metric_row, model_name, recall_target):
    total_rows = float(stage1_row["total_rows"])
    total_legit = float(stage1_row["total_legit"])
    total_fraud = float(stage1_row["total_fraud"])

    candidate_legit = float(stage1_row["candidate_legit"])
    candidate_fraud = float(stage1_row["candidate_fraud"])

    noncandidate_legit = max(total_legit - candidate_legit, 0)
    noncandidate_fraud = max(total_fraud - candidate_fraud, 0)

    cand_tn = float(candidate_metric_row["tn"])
    cand_fp = float(candidate_metric_row["fp"])
    cand_fn = float(candidate_metric_row["fn"])
    cand_tp = float(candidate_metric_row["tp"])

    full_tn = cand_tn + noncandidate_legit
    full_fp = cand_fp
    full_fn = cand_fn + noncandidate_fraud
    full_tp = cand_tp

    accuracy = (full_tp + full_tn) / total_rows if total_rows > 0 else 0
    precision = full_tp / (full_tp + full_fp) if (full_tp + full_fp) > 0 else 0
    recall = full_tp / (full_tp + full_fn) if (full_tp + full_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = full_fp / (full_fp + full_tn) if (full_fp + full_tn) > 0 else 0
    fnr = full_fn / (full_fn + full_tp) if (full_fn + full_tp) > 0 else 0

    return {
        "track": "phase5_fixed_recall_estimated_full_hybrid",
        "model": model_name,
        "split": "test",
        "recall_target": recall_target,
        "threshold": candidate_metric_row["threshold"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": None,
        "pr_auc": None,
        "fpr": fpr,
        "fnr": fnr,
        "tn": full_tn,
        "fp": full_fp,
        "fn": full_fn,
        "tp": full_tp,
        "n_rows": total_rows,
        "note": "Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix.",
    }


def feature_map_states(X):
    X = np.asarray(X, dtype=float)
    n_samples, n_qubits = X.shape
    dim = 2 ** n_qubits

    states = np.zeros((n_samples, dim), dtype=np.complex128)

    for s in range(n_samples):
        x = X[s]

        for basis in range(dim):
            amp = 1.0 + 0j
            z_values = []

            for q in range(n_qubits):
                bit = (basis >> q) & 1

                if bit == 0:
                    amp *= np.cos(x[q] / 2.0)
                    z_values.append(1.0)
                else:
                    amp *= np.sin(x[q] / 2.0)
                    z_values.append(-1.0)

            phase = 0.0
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    phase += 0.25 * x[i] * x[j] * z_values[i] * z_values[j]

            states[s, basis] = amp * np.exp(1j * phase)

        norm = np.linalg.norm(states[s])
        if norm > 0:
            states[s] /= norm

    return states


def quantum_kernel(A, B):
    return np.abs(A @ B.conj().T) ** 2
