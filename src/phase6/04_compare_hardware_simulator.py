import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from phase6_common import (
    read_json,
    save_json,
    write_rows_csv,
    feature_map_states,
    vqc_scores_from_params,
    binary_metrics_from_scores,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-dataset", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--hardware-scores", required=True)
    parser.add_argument("--out-reports", required=True)
    args = parser.parse_args()

    out_reports = Path(args.out_reports)
    out_reports.mkdir(parents=True, exist_ok=True)

    if not Path(args.hardware_scores).exists():
        raise FileNotFoundError("Hardware scores file not found. Run the IBM hardware job first.")

    raw = np.load(args.phase6_dataset, allow_pickle=True)

    test_mask = raw["split"].astype(str) == "test"
    X_test = raw["X"][test_mask]
    y_test = raw["y"][test_mask].astype(int)
    w_test = raw["sample_weight"][test_mask].astype(float)

    params_payload = read_json(args.params)
    params = np.asarray(params_payload["params"], dtype=float)
    n_qubits = int(params_payload["n_qubits"])
    layers = int(params_payload["layers"])
    threshold = float(params_payload["threshold"])

    hw = pd.read_csv(args.hardware_scores)

    if "original_test_index" in hw.columns:
        selected_idx = hw["original_test_index"].astype(int).to_numpy()
    else:
        selected_idx = hw["sample_index"].astype(int).to_numpy()

    X_selected = X_test[selected_idx]
    y_selected = y_test[selected_idx]
    w_selected = w_test[selected_idx]

    states = feature_map_states(X_selected)
    simulator_scores = vqc_scores_from_params(states, params, n_qubits, layers)

    rows = []

    for i, row in hw.iterrows():
        hardware_score = float(row["hardware_score"])
        simulator_score = float(simulator_scores[i])

        rows.append({
            "sample_index": int(row["sample_index"]),
            "original_test_index": int(selected_idx[i]),
            "label": int(row["label"]),
            "sample_weight": float(row["sample_weight"]),
            "simulator_score": simulator_score,
            "hardware_score": hardware_score,
            "absolute_score_difference": abs(simulator_score - hardware_score),
            "simulator_pred": int(simulator_score >= threshold),
            "hardware_pred": int(hardware_score >= threshold),
            "agreement": int((simulator_score >= threshold) == (hardware_score >= threshold)),
        })

    comp = pd.DataFrame(rows)
    comp.to_csv(out_reports / "phase6_simulator_vs_hardware_scores.csv", index=False)

    sim_metric = binary_metrics_from_scores(
        y_selected,
        simulator_scores,
        threshold=threshold,
        sample_weight=w_selected,
        model_name="phase6_vqc_statevector_on_hardware_subset",
        split="hardware_test_subset",
        track="phase6_simulator_reference_on_hardware_subset",
    )

    hw_metric = binary_metrics_from_scores(
        comp["label"].to_numpy(),
        comp["hardware_score"].to_numpy(),
        threshold=threshold,
        sample_weight=comp["sample_weight"].to_numpy(),
        model_name="phase6_vqc_ibm_hardware",
        split="hardware_test_subset",
        track="phase6_ibm_hardware_validation",
    )

    write_rows_csv(
        [sim_metric, hw_metric],
        out_reports / "phase6_simulator_hardware_metric_comparison.csv",
    )

    summary = {
        "n_samples": int(len(comp)),
        "mean_absolute_score_difference": float(comp["absolute_score_difference"].mean()),
        "median_absolute_score_difference": float(comp["absolute_score_difference"].median()),
        "prediction_agreement_rate": float(comp["agreement"].mean()),
        "threshold": threshold,
    }

    save_json(summary, out_reports / "phase6_simulator_vs_hardware_summary.json")

    print("Simulator vs hardware comparison complete.")
    print(pd.DataFrame([sim_metric, hw_metric]))
    print(summary)


if __name__ == "__main__":
    main()
