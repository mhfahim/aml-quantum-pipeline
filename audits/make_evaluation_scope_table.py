from pathlib import Path
import pandas as pd

OUT = Path("reports/supervisor_fixes")
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "evaluation_group": "Large-scale classical models",
        "example_models": "Logistic Regression, HistGradientBoosting, Random Forest",
        "test_size": "Large / candidate or full-scale test set",
        "purpose": "Practical AML model evaluation",
        "directly_comparable_with": "Other models evaluated on the same large/candidate test set only",
        "claim_allowed": "Can discuss practical large-scale performance",
        "claim_not_allowed": "Do not compare directly with small quantum subset results as a single leaderboard"
    },
    {
        "evaluation_group": "Standalone quantum simulation",
        "example_models": "Quantum Kernel SVC, VQC Statevector",
        "test_size": "Reduced quantum-compatible subset",
        "purpose": "Quantum simulation comparison under qubit/resource limits",
        "directly_comparable_with": "Other models evaluated on the same reduced subset only",
        "claim_allowed": "Can discuss reduced-feature quantum behaviour",
        "claim_not_allowed": "Do not claim full-scale superiority over classical models"
    },
    {
        "evaluation_group": "Isolation Forest + quantum hybrid simulation",
        "example_models": "Isolation Forest + Quantum Kernel SVC, Isolation Forest + VQC",
        "test_size": "Small Isolation Forest-selected quantum-compatible subset",
        "purpose": "Hybrid quantum simulation feasibility and subset performance",
        "directly_comparable_with": "Other Phase 7 hybrid quantum simulation models only",
        "claim_allowed": "Can say strong within this small hybrid simulation setup",
        "claim_not_allowed": "Do not call it the best overall AML model"
    },
    {
        "evaluation_group": "IBM quantum hardware feasibility",
        "example_models": "VQC hardware, Isolation Forest + Quantum Kernel hardware, Isolation Forest + VQC hardware",
        "test_size": "Very small hardware subset",
        "purpose": "Real-device execution and resource-cost validation",
        "directly_comparable_with": "Other hardware models in the same feasibility run only",
        "claim_allowed": "Can say circuit executed successfully and one model was better within the small feasibility test",
        "claim_not_allowed": "Do not make strong performance claims"
    }
]

df = pd.DataFrame(rows)
df.to_csv(OUT / "evaluation_scope_and_claims_table.csv", index=False)

with open(OUT / "evaluation_scope_and_claims_table.txt", "w", encoding="utf-8") as f:
    f.write(df.to_string(index=False))

print(df.to_string(index=False))
print("\nCreated reports/supervisor_fixes/evaluation_scope_and_claims_table.csv")
