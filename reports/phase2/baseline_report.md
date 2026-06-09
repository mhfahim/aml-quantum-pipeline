# Phase 2 Classical Baseline Report

## Important Note

These are starter leakage-safe baselines. Amount fields were cast from string to numeric. Categorical fields are intentionally excluded for this first conservative baseline.

## Numeric Features Used

- `amount_received_num`

## Metrics

- logistic_regression_balanced | train | F1=0.0205, ROC-AUC=0.6549, PR-AUC=0.0976, Precision=0.0935, Recall=0.0115
- logistic_regression_balanced | valid | F1=0.0210, ROC-AUC=0.6653, PR-AUC=0.1337, Precision=0.1368, Recall=0.0114
- logistic_regression_balanced | test | F1=0.0234, ROC-AUC=0.6809, PR-AUC=0.1768, Precision=0.1764, Recall=0.0126
- hist_gradient_boosting | train | F1=0.0000, ROC-AUC=0.7208, PR-AUC=0.1484, Precision=0.0000, Recall=0.0000
- hist_gradient_boosting | valid | F1=0.0000, ROC-AUC=0.7263, PR-AUC=0.1952, Precision=0.0000, Recall=0.0000
- hist_gradient_boosting | test | F1=0.0000, ROC-AUC=0.7544, PR-AUC=0.2768, Precision=0.0000, Recall=0.0000