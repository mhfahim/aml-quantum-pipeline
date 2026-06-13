import argparse
import json
import time
from pathlib import Path

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

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true, scores, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) >= 2 else np.nan,
        "pr_auc": average_precision_score(y_true, scores) if len(np.unique(y_true)) >= 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_rows": int(len(y_true)),
        "predicted_positive_rate": float(np.mean(pred == 1)),
        "actual_positive_rate": float(np.mean(y_true == 1)),
    }

    return out


def best_threshold_by_f1(y_true, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t = 0.5
    best_f1 = -1

    for t in candidates:
        pred = (scores >= t).astype(int)
        val = f1_score(y_true, pred, zero_division=0)

        if val > best_f1:
            best_f1 = val
            best_t = float(t)

    return best_t


class FTTransformerStyle(nn.Module):
    def __init__(self, n_features, d_token=32, n_heads=4, n_layers=2, dropout=0.15):
        super().__init__()

        self.n_features = n_features
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out).squeeze(-1)


class AutoIntStyle(nn.Module):
    def __init__(self, n_features, d_token=32, n_heads=4, n_layers=3, dropout=0.15):
        super().__init__()

        self.n_features = n_features
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(n_features * d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        for attn, norm in zip(self.attn_layers, self.norms):
            attended, _ = attn(tokens, tokens, tokens, need_weights=False)
            tokens = norm(tokens + self.dropout(attended))

        flat = tokens.reshape(tokens.size(0), -1)
        return self.head(flat).squeeze(-1)


def predict_scores(model, X, device, batch_size=256):
    model.eval()
    scores = []

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(probs)

    return np.concatenate(scores)


def train_one_model(name, model, X_train, y_train, X_valid, y_valid, device, epochs=250, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=25):
    start = time.perf_counter()

    model = model.to(device)

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_valid_f1 = -1
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        valid_scores = predict_scores(model, X_valid, device)
        threshold = best_threshold_by_f1(y_valid, valid_scores)
        valid_pred = (valid_scores >= threshold).astype(int)
        valid_f1 = f1_score(y_valid, valid_pred, zero_division=0)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.perf_counter() - start

    valid_scores = predict_scores(model, X_valid, device)
    threshold = best_threshold_by_f1(y_valid, valid_scores)

    return model, threshold, train_time, best_epoch, best_valid_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/kaggle/working/aml_phase7/data/phase7_if_quantum_dataset_4q.npz")
    parser.add_argument("--out-dir", default="reports/panel_revision")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.random_state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(int)
    split = data["split"].astype(str)

    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = X_train.shape[1]

    model_builders = {
        "FT-Transformer-style Tabular Classifier": lambda: FTTransformerStyle(n_features=n_features),
        "AutoInt-style Attention Tabular Classifier": lambda: AutoIntStyle(n_features=n_features),
    }

    performance_rows = []
    runtime_rows = []
    prediction_rows = []

    for model_name, builder in model_builders.items():
        print(f"\nTraining {model_name}")
        model = builder()

        model, threshold, train_time, best_epoch, best_valid_f1 = train_one_model(
            name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
        )

        infer_start = time.perf_counter()
        test_scores = predict_scores(model, X_test, device)
        inference_time = time.perf_counter() - infer_start

        test_pred = (test_scores >= threshold).astype(int)
        m = metrics(y_test, test_scores, test_pred)

        row = {
            "scope": "same_reduced_four_feature_subset",
            "model": model_name,
            "model_type": "Transformer-based tabular baseline",
            "threshold": threshold,
            "best_epoch": int(best_epoch),
            "best_valid_f1": float(best_valid_f1),
        }
        row.update(m)
        performance_rows.append(row)

        runtime_rows.append({
            "scope": "same_reduced_four_feature_subset",
            "model": model_name,
            "device": device,
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "training_time_seconds": float(train_time),
            "test_inference_time_seconds": float(inference_time),
            "total_runtime_seconds": float(train_time + inference_time),
            "best_epoch": int(best_epoch),
        })

        for i in range(len(y_test)):
            prediction_rows.append({
                "scope": "same_reduced_four_feature_subset",
                "model": model_name,
                "sample_id": str(i),
                "split": "test",
                "label": int(y_test[i]),
                "score": float(test_scores[i]),
                "pred": int(test_pred[i]),
                "threshold": float(threshold),
            })

    perf = pd.DataFrame(performance_rows)
    runtime = pd.DataFrame(runtime_rows)
    preds = pd.DataFrame(prediction_rows)

    for df in [perf, runtime]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(6)

    perf.to_csv(out_dir / "sota_transformer_model_comparison_table.csv", index=False)
    runtime.to_csv(out_dir / "sota_transformer_runtime_table.csv", index=False)
    preds.to_csv(out_dir / "sota_transformer_predictions_reduced_subset.csv", index=False)

    metadata = {
        "dataset": args.dataset,
        "models_added": list(model_builders.keys()),
        "evaluation_scope": "same reduced four-feature subset",
        "features": "same four quantum-compatible features represented in the dataset",
        "random_state": args.random_state,
        "device": device,
        "note": "Transformer-based tabular baselines added for panel revision. FPR is included as the false-positive metric."
    }

    with open(out_dir / "sota_transformer_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nPerformance:")
    print(perf.to_string(index=False))

    print("\nRuntime:")
    print(runtime.to_string(index=False))

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
