import argparse
from pathlib import Path
import pandas as pd
import json


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_json_if_exists(path):
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True)
    args = parser.parse_args()

    reports = Path(args.reports)

    subset_counts = read_csv_if_exists(reports / "phase6_subset_counts.csv")
    statevector_metrics = read_csv_if_exists(reports / "phase6_statevector_reference_metrics.csv")
    hardware_metrics = read_csv_if_exists(reports / "phase6_hardware_metrics.csv")
    circuit_inventory = read_csv_if_exists(reports / "phase6_hardware_circuit_inventory.csv")
    transpiled_inventory = read_csv_if_exists(reports / "phase6_transpiled_circuit_inventory.csv")
    sim_hw_metrics = read_csv_if_exists(reports / "phase6_simulator_hardware_metric_comparison.csv")

    subset_meta = read_json_if_exists(reports / "phase6_subset_metadata.json")
    train_meta = read_json_if_exists(reports / "phase6_statevector_training_metadata.json")
    hardware_meta = read_json_if_exists(reports / "phase6_hardware_job_metadata.json")
    sim_hw_summary = read_json_if_exists(reports / "phase6_simulator_vs_hardware_summary.json")

    rows = []

    rows.append("# Phase 6 Report: IBM Quantum Hardware Validation\n")

    rows.append("## 1. Purpose\n")
    rows.append(
        "Phase 6 validates the reduced-feature quantum AML model on real IBM Quantum hardware. "
        "The purpose is not to claim full-scale quantum advantage, but to evaluate hardware feasibility, "
        "runtime burden, transpiled circuit resources, and simulator-vs-hardware agreement.\n"
    )

    rows.append("## 2. Hardware Validation Subset\n")
    if not subset_counts.empty:
        rows.append(subset_counts.to_markdown(index=False))
    if subset_meta:
        rows.append(f"\nTotal subset rows: **{subset_meta.get('total_rows')}**")
        rows.append(f"Number of qubits: **{subset_meta.get('n_qubits')}**")
    rows.append("")

    rows.append("## 3. Statevector Reference Model\n")
    if not statevector_metrics.empty:
        rows.append(statevector_metrics.to_markdown(index=False))
    if train_meta:
        rows.append(f"\nTraining time seconds: **{train_meta.get('training_time_seconds')}**")
        rows.append(f"Optimizer final loss: **{train_meta.get('optimizer_final_loss')}**")
        rows.append(f"Trainable parameters: **{train_meta.get('trainable_parameters')}**")
    rows.append("")

    rows.append("## 4. Circuit Resource Summary\n")
    if not circuit_inventory.empty:
        rows.append("Untranspiled circuit inventory summary:\n")
        rows.append(
            circuit_inventory[
                ["n_qubits", "depth_untranspiled", "size_untranspiled"]
            ].describe().to_markdown()
        )

    if not transpiled_inventory.empty:
        rows.append("\nTranspiled circuit inventory summary:\n")
        rows.append(
            transpiled_inventory[
                ["depth_transpiled", "size_transpiled"]
            ].describe().to_markdown()
        )
    else:
        rows.append("\nTranspiled circuit inventory not found.")
    rows.append("")

    rows.append("## 5. IBM Hardware Job Metadata\n")
    if hardware_meta:
        rows.append(f"- Backend: **{hardware_meta.get('backend_name')}**")
        rows.append(f"- Connected channel: **{hardware_meta.get('connected_channel')}**")
        rows.append(f"- Job ID: **{hardware_meta.get('job_id')}**")
        rows.append(f"- Shots: **{hardware_meta.get('shots')}**")
        rows.append(f"- Number of circuits: **{hardware_meta.get('n_circuits')}**")
        rows.append(f"- Number of qubits: **{hardware_meta.get('n_qubits')}**")
        rows.append(f"- Transpile time seconds: **{hardware_meta.get('transpile_time_seconds')}**")
        rows.append(f"- Total turnaround time seconds: **{hardware_meta.get('total_turnaround_time_seconds')}**")
        rows.append(f"- Wait-for-result time seconds: **{hardware_meta.get('wait_for_result_time_seconds')}**")

        metrics = hardware_meta.get("job_payload", {}).get("metrics", {})
        usage = metrics.get("usage", {})
        bss = metrics.get("bss", {})

        if usage:
            rows.append(f"- IBM usage quantum seconds: **{usage.get('quantum_seconds')}**")
            rows.append(f"- IBM usage status: **{usage.get('status')}**")
        if bss:
            rows.append(f"- IBM BSS seconds: **{bss.get('seconds')}**")
    else:
        rows.append("No real IBM hardware metadata found.")
    rows.append("")

    rows.append("## 6. Hardware Metrics\n")
    if not hardware_metrics.empty:
        rows.append(hardware_metrics.to_markdown(index=False))
    else:
        rows.append("Hardware metrics not found.")
    rows.append("")

    rows.append("## 7. Simulator vs Hardware Agreement\n")
    if not sim_hw_metrics.empty:
        rows.append(sim_hw_metrics.to_markdown(index=False))

    if sim_hw_summary:
        rows.append(f"\nPrediction agreement rate: **{sim_hw_summary.get('prediction_agreement_rate')}**")
        rows.append(f"Mean absolute score difference: **{sim_hw_summary.get('mean_absolute_score_difference')}**")
        rows.append(f"Median absolute score difference: **{sim_hw_summary.get('median_absolute_score_difference')}**")
    else:
        rows.append("Simulator-vs-hardware comparison not found.")
    rows.append("")

    rows.append("## 8. Interpretation\n")
    rows.append(
        "The IBM hardware run confirms that the 4-qubit reduced AML quantum classifier can be executed on real quantum hardware. "
        "However, this remains a small-scale feasibility and resource validation experiment. "
        "Together with Phases 4 and 5, the result supports a resource-aware conclusion: classical and hybrid models remain more practical for large-scale AML detection, while quantum models are currently more suitable for reduced-feature experimental validation and hardware feasibility testing."
    )

    out_path = reports / "PHASE6_IBM_HARDWARE_VALIDATION_REPORT.md"
    out_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
