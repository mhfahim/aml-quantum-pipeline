# Phase 5 Report: Conditional Advantage Tests

## 1. Purpose

Phase 5 examines the conditions under which classical, quantum, and hybrid AML detection models perform better. Unlike Phase 4, which provides the main model comparison, Phase 5 focuses on conditional decision settings: label scarcity, fixed-recall false-positive reduction, novel-pattern generalization, and scalability/cost trade-offs.

## 2. Label Scarcity Experiment

This experiment tests reduced classical models and the quantum kernel model under limited labeled training data. The purpose is to identify whether quantum-style reduced feature models become more competitive when fewer labeled samples are available.

| model                          |   train_fraction |    mean_f1 |      std_f1 |   mean_pr_auc |   std_pr_auc |   mean_recall |   std_recall |   mean_precision |   std_precision |   mean_training_time |
|:-------------------------------|-----------------:|-----------:|------------:|--------------:|-------------:|--------------:|-------------:|-----------------:|----------------:|---------------------:|
| quantum_kernel_svc_statevector |             0.1  | 0.0100403  | 5.35907e-05 |    0.00588019 |  0.00123729  |      0.888    |    0.0935521 |       0.00504894 |     2.99008e-05 |           0.00383032 |
| quantum_kernel_svc_statevector |             0.25 | 0.0104137  | 0.000543125 |    0.0071316  |  0.000342392 |      0.908    |    0.0610246 |       0.0052369  |     0.000272739 |           0.00770059 |
| quantum_kernel_svc_statevector |             0.5  | 0.0100042  | 0.0002937   |    0.00511424 |  0.000449623 |      0.877333 |    0.0508462 |       0.00503092 |     0.000149646 |           0.0141151  |
| quantum_kernel_svc_statevector |             1    | 0.0100835  | 3.8965e-05  |    0.00584685 |  0.00129439  |      0.945333 |    0.0845064 |       0.00506897 |     2.22772e-05 |           0.0407579  |
| reduced_logistic_regression    |             0.1  | 0.0103744  | 8.34381e-05 |    0.00551306 |  0.00136351  |      0.892667 |    0.0611991 |       0.00521767 |     4.43955e-05 |           1.66759    |
| reduced_logistic_regression    |             0.25 | 0.0103335  | 0.000205925 |    0.00752108 |  0.00107981  |      0.925333 |    0.0057735 |       0.00519579 |     0.000103944 |           0.486207   |
| reduced_logistic_regression    |             0.5  | 0.0104521  | 0.000272766 |    0.00696975 |  0.000242391 |      0.89     |    0.0588897 |       0.00525711 |     0.000140042 |           0.0259226  |
| reduced_logistic_regression    |             1    | 0.0104983  | 0           |    0.0069025  |  0           |      0.934    |    0         |       0.0052788  |     0           |           0.0125004  |
| reduced_mlp                    |             0.1  | 0.00988641 | 0.000345163 |    0.00466507 |  0.000279116 |      0.909333 |    0.125033  |       0.00497043 |     0.000170544 |           0.0390057  |
| reduced_mlp                    |             0.25 | 0.00979334 | 0.000496059 |    0.00558903 |  0.00115815  |      0.868    |    0.10949   |       0.00492457 |     0.000247235 |           0.0747929  |
| reduced_mlp                    |             0.5  | 0.0100885  | 0.000181503 |    0.00605713 |  0.00251274  |      0.882667 |    0.0869329 |       0.00507336 |     8.8831e-05  |           0.0893995  |
| reduced_mlp                    |             1    | 0.0100762  | 0.000315083 |    0.00550242 |  0.00114971  |      0.859333 |    0.0737925 |       0.0050679  |     0.000157002 |           0.142785   |
| reduced_svm_rbf                |             0.1  | 0.00990572 | 4.00739e-05 |    0.00613665 |  0.00231206  |      0.867333 |    0.10366   |       0.00498156 |     1.71332e-05 |           0.00748833 |
| reduced_svm_rbf                |             0.25 | 0.0101673  | 0.00014726  |    0.00569312 |  0.0012416   |      0.984667 |    0.0113725 |       0.00511003 |     7.43634e-05 |           0.0276536  |
| reduced_svm_rbf                |             0.5  | 0.0100667  | 0.000131584 |    0.00746271 |  0.00514011  |      0.908    |    0.0225389 |       0.00506145 |     6.69792e-05 |           0.0958834  |
| reduced_svm_rbf                |             1    | 0.00994752 | 0.000373358 |    0.00485725 |  0.000303962 |      0.920667 |    0.114023  |       0.00500097 |     0.000186098 |           0.361344   |

Best label-scarcity model by mean PR-AUC: **reduced_logistic_regression** at train fraction **0.25**.
Best label-scarcity model by mean F1: **reduced_logistic_regression** at train fraction **1.0**.

## 3. Fixed-Recall False-Positive Trade-Off

This experiment evaluates Stage 2 classical models at fixed recall targets. For AML systems, this is important because investigators often require high recall while trying to reduce unnecessary alerts.

| track                                     | model                                                  | split   |   recall_target |   threshold |   accuracy |   precision |   recall |        f1 |   roc_auc |   pr_auc |       fpr |      fnr |          tn |         fp |    fn |    tp |      n_rows | note                                                                                                    |
|:------------------------------------------|:-------------------------------------------------------|:--------|----------------:|------------:|-----------:|------------:|---------:|----------:|----------:|---------:|----------:|---------:|------------:|-----------:|------:|------:|------------:|:--------------------------------------------------------------------------------------------------------|
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_logistic_regression    | test    |            0.8  |  0.00428025 |   0.946608 |   0.0195102 | 0.754597 | 0.038037  |       nan |      nan | 0.0529477 | 0.245403 | 6.11183e+07 | 3.417e+06  | 22112 | 67993 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_logistic_regression    | test    |            0.9  |  0.00371405 |   0.932478 |   0.0168749 | 0.826136 | 0.0330741 |       nan |      nan | 0.0672004 | 0.173864 | 6.01985e+07 | 4.3368e+06 | 15666 | 74439 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_logistic_regression    | test    |            0.95 |  0.00325911 |   0.921481 |   0.0151374 | 0.861562 | 0.029752  |       nan |      nan | 0.0782641 | 0.138438 | 5.94845e+07 | 5.0508e+06 | 12474 | 77631 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_hist_gradient_boosting | test    |            0.8  |  0.00495898 |   0.966385 |   0.0301119 | 0.73633  | 0.0578578 |       nan |      nan | 0.0331136 | 0.26367  | 6.23983e+07 | 2.137e+06  | 23758 | 66347 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_hist_gradient_boosting | test    |            0.9  |  0.00365937 |   0.952891 |   0.0237217 | 0.813362 | 0.046099  |       nan |      nan | 0.0467372 | 0.186638 | 6.15191e+07 | 3.0162e+06 | 16817 | 73288 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_hist_gradient_boosting | test    |            0.95 |  0.00245002 |   0.939484 |   0.019404  | 0.853493 | 0.0379453 |       nan |      nan | 0.0602213 | 0.146507 | 6.06489e+07 | 3.8864e+06 | 13201 | 76904 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_random_forest          | test    |            0.8  |  0.00559182 |   0.967982 |   0.0317764 | 0.740924 | 0.0609393 |       nan |      nan | 0.0315207 | 0.259076 | 6.25011e+07 | 2.0342e+06 | 23344 | 66761 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_random_forest          | test    |            0.9  |  0.00336521 |   0.956048 |   0.0254367 | 0.814605 | 0.0493328 |       nan |      nan | 0.0435761 | 0.185395 | 6.17231e+07 | 2.8122e+06 | 16705 | 73400 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |
| phase5_fixed_recall_estimated_full_hybrid | phase3b_screener_plus_classical_random_forest          | test    |            0.95 |  0.00204692 |   0.944984 |   0.0212914 | 0.852461 | 0.0415451 |       nan |      nan | 0.0547111 | 0.147539 | 6.10045e+07 | 3.5308e+06 | 13294 | 76811 | 6.46374e+07 | Estimated by combining Phase 3B screening counts with weighted Stage 2 candidate-pool confusion matrix. |

At recall target **0.8**, the lowest estimated full-pipeline FPR is from **phase3b_screener_plus_classical_random_forest** with FPR=0.0315207223802071 and recall=0.7409244769990566.
At recall target **0.8**, the best F1 is from **phase3b_screener_plus_classical_random_forest** with F1=0.0609392870867422.

At recall target **0.9**, the lowest estimated full-pipeline FPR is from **phase3b_screener_plus_classical_random_forest** with FPR=0.0435761358163497 and recall=0.8146051828422396.
At recall target **0.9**, the best F1 is from **phase3b_screener_plus_classical_random_forest** with F1=0.049332847174031.

At recall target **0.95**, the lowest estimated full-pipeline FPR is from **phase3b_screener_plus_classical_random_forest** with FPR=0.0547111230852598 and recall=0.8524610177015703.
At recall target **0.95**, the best F1 is from **phase3b_screener_plus_classical_random_forest** with F1=0.0415451051405786.

## 4. Novel-Pattern Stress Test

This experiment creates a proxy AML typology using payment format, currency behavior, and amount bands. One typology is excluded from training and validation, then evaluated separately as a novel-pattern test subset.

Selected novel proxy typology: **ACH__same_currency__medium**

| track                            | model                        | split                |   threshold |   accuracy |   precision |   recall |        f1 |   roc_auc |    pr_auc |      fpr |       fnr |              tn |              fp |   fn |    tp |   n_rows | novel_typology             | threshold_policy            |   recall_target |   prediction_time_seconds |
|:---------------------------------|:-----------------------------|:---------------------|------------:|-----------:|------------:|---------:|----------:|----------:|----------:|---------:|----------:|----------------:|----------------:|-----:|------:|---------:|:---------------------------|:----------------------------|----------------:|--------------------------:|
| phase5_novel_pattern_stress_test | novel_logistic_regression    | test_novel_typology  |  0.00403087 |   0.151256 |   0.0404049 | 0.942027 | 0.0774863 |  0.556908 | 0.0427287 | 0.879843 | 0.0579729 | 116000          | 849400          | 2201 | 35765 |    42793 | ACH__same_currency__medium | recall_target_precision_max |             0.8 |                  0.053095 |
| phase5_novel_pattern_stress_test | novel_logistic_regression    | test_seen_typologies |  0.00403087 |   0.352478 |   0.0103551 | 0.835856 | 0.0204568 |  0.636956 | 0.0130965 | 0.651465 | 0.164144  |      1.8158e+06 |      3.394e+06  | 6974 | 35513 |    68536 | ACH__same_currency__medium | recall_target_precision_max |             0.8 |                  0.165936 |
| phase5_novel_pattern_stress_test | novel_hist_gradient_boosting | test_novel_typology  |  0.00539699 |   0.302304 |   0.049191  | 0.95143  | 0.0935454 |  0.747354 | 0.0910254 | 0.723224 | 0.0485698 | 267200          | 698200          | 1844 | 36122 |    42793 | ACH__same_currency__medium | recall_target_precision_max |             0.8 |                  0.32754  |
| phase5_novel_pattern_stress_test | novel_hist_gradient_boosting | test_seen_typologies |  0.00539699 |   0.654539 |   0.0185962 | 0.805541 | 0.0363533 |  0.805348 | 0.0521959 | 0.346693 | 0.194459  |      3.4036e+06 |      1.8062e+06 | 8262 | 34225 |    68536 | ACH__same_currency__medium | recall_target_precision_max |             0.8 |                  0.52336  |
| phase5_novel_pattern_stress_test | novel_random_forest          | test_novel_typology  |  0.00581755 |   0.17136  |   0.0432187 | 0.9887   | 0.0828172 |  0.71657  | 0.0844952 | 0.860783 | 0.0112996 | 134400          | 831000          |  429 | 37537 |    42793 | ACH__same_currency__medium | recall_target_precision_max |             0.8 |                  0.140063 |
| phase5_novel_pattern_stress_test | novel_random_forest          | test_seen_typologies |  0.00581755 |   0.654695 |   0.0186161 | 0.806058 | 0.0363917 |  0.803791 | 0.0478161 | 0.346539 | 0.193942  |      3.4044e+06 |      1.8054e+06 | 8240 | 34247 |    68536 | ACH__same_currency__medium | recall_target_precision_max |             0.8 |                  0.265926 |

Best novel-pattern model by F1: **novel_hist_gradient_boosting** with F1=0.093545413110135.
Best novel-pattern model by recall: **novel_random_forest** with recall=0.988700416161829.

## 5. Scalability and Cost Trade-Off

This section summarizes practical workload reduction and runtime/cost implications. Real IBM quantum hardware queue time and execution time are reserved for Phase 6.

| component                                  |   test_total_rows |   test_candidate_rows |   actual_candidate_pct |   transactions_removed_before_stage2_pct |   fraud_retention |   stage2_workload_reduction_factor | interpretation                                                                                |   n_qubits |   statevector_dimension |   estimated_circuit_depth | shots             | hardware_queue_time_seconds   | hardware_execution_time_seconds   |
|:-------------------------------------------|------------------:|----------------------:|-----------------------:|-----------------------------------------:|------------------:|-----------------------------------:|:----------------------------------------------------------------------------------------------|-----------:|------------------------:|--------------------------:|:------------------|:------------------------------|:----------------------------------|
| phase3b_stage1_screening                   |       6.46374e+07 |           6.26766e+06 |              0.0969665 |                                 0.903034 |          0.892881 |                            10.3128 | Stage 1 reduces the transaction universe before expensive Stage 2 classical/quantum analysis. |        nan |                     nan |                       nan | nan               | nan                           | nan                               |
| quantum_kernel_svc_statevector             |     nan           |         nan           |            nan         |                               nan        |        nan        |                           nan      | Hardware queue and execution cost will be measured in Phase 6.                                |          4 |                      16 |                       nan | statevector_exact | Phase 6 only                  | Phase 6 only                      |
| variational_quantum_classifier_statevector |     nan           |         nan           |            nan         |                               nan        |        nan        |                           nan      | Hardware queue and execution cost will be measured in Phase 6.                                |          4 |                      16 |                        19 | statevector_exact | Phase 6 only                  | Phase 6 only                      |

Phase 3B Stage 1 retained fraud at **0.8928805282725709** while passing only **0.0969664515477856** of test transactions to Stage 2. This gives an approximate Stage 2 workload reduction factor of **10.312845154565585**.

Runtime summary:

| model                                                  | track                                |   test_recall |   test_precision |    test_f1 |   test_fpr |   inference_rows_per_second |   inference_time_seconds |   estimated_seconds_per_1m_candidate_rows |
|:-------------------------------------------------------|:-------------------------------------|--------------:|-----------------:|-----------:|-----------:|----------------------------:|-------------------------:|------------------------------------------:|
| classical_logistic_regression                          | candidate_pool_stage2_weighted       |      0.925248 |       0.0168749  | 0.0331452  |  0.702293  |            686849           |              0.162087    |                                  1.45592  |
| classical_hist_gradient_boosting                       | candidate_pool_stage2_weighted       |      0.910942 |       0.0237217  | 0.0462393  |  0.488438  |            270467           |              0.411618    |                                  3.69732  |
| classical_random_forest                                | candidate_pool_stage2_weighted       |      0.912334 |       0.0254367  | 0.0494934  |  0.455402  |            233163           |              0.477472    |                                  4.28884  |
| phase3b_screener_plus_classical_logistic_regression    | estimated_full_hybrid_pipeline       |      0.826136 |       0.0168749  | 0.0330741  |  0.0672004 |               nan           |            nan           |                                nan        |
| phase3b_screener_plus_classical_hist_gradient_boosting | estimated_full_hybrid_pipeline       |      0.813362 |       0.0237217  | 0.046099   |  0.0467372 |               nan           |            nan           |                                nan        |
| phase3b_screener_plus_classical_random_forest          | estimated_full_hybrid_pipeline       |      0.814605 |       0.0254367  | 0.0493328  |  0.0435761 |               nan           |            nan           |                                nan        |
| reduced_logistic_regression                            | reduced_quantum_compatible_classical |      0.934    |       0.0052788  | 0.0104983  |  0.88      |                 3.31041e+06 |              0.000302077 |                                  0.302077 |
| reduced_svm_rbf                                        | reduced_quantum_compatible_classical |      0.944    |       0.00496466 | 0.00987737 |  0.946     |             17522.5         |              0.0570694   |                                 57.0694   |
| reduced_mlp                                            | reduced_quantum_compatible_classical |      0.886    |       0.00516058 | 0.0102614  |  0.854     |                 2.31657e+06 |              0.000431672 |                                  0.431672 |
| quantum_kernel_svc_statevector                         | quantum_statevector_reduced_features |      0.944    |       0.00514879 | 0.0102417  |  0.912     |             86797.2         |              0.0115211   |                                 11.5211   |
| variational_quantum_classifier_statevector             | quantum_statevector_reduced_features |      0.99     |       0.0049453  | 0.00984144 |  0.996     |              6255.98        |              0.159847    |                                159.847    |

## 6. Phase 5 Decision Summary

Phase 5 supports condition-based conclusions rather than a single universal winner. Classical and hybrid models remain the most practical for deployment because they provide stronger false-positive control and runtime scalability. Quantum models remain useful as reduced-feature experimental comparators, especially under constrained feature and label settings, but Phase 4 and Phase 5 evidence should be interpreted carefully because current quantum models still have weak precision and high false-positive rates. The strongest current practical direction remains the Phase 3B high-recall screener followed by a classical Stage 2 model.