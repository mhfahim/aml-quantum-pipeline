# Phase 3B Report: Improved High-Recall Stage 1 Screening

## 1. Purpose

Phase 3B improves the initial Stage 1 Isolation Forest baseline by training a high-recall classical screening model. The objective is not final classification, but candidate generation: retaining as many laundering cases as possible while reducing the transaction universe before Phase 4 quantum and reduced-feature comparisons.

## 2. Model

- Model name: **Stage1B High-Recall Classical Screening Filter**
- Model type: **logistic_regression_balanced**
- Training rows: **780,000**
- Positive rows: **180,000**
- Negative rows: **600,000**
- Runtime seconds: **84.15**
- Features: `['amount_received_num', 'amount_paid_num', 'amount_delta_num', 'amount_ratio_num', 'log_amount_received_num', 'log_amount_paid_num', 'hour_num', 'day_num', 'same_currency_num', 'receiving_currency', 'payment_currency', 'payment_format']`

## 3. Validation Thresholds

- Validation ROC-AUC: **0.9041305085762938**
- Validation PR-AUC: **0.014235859021642327**
- Target candidate percentage `0.1`: threshold `0.7876329609654898`
- Target candidate percentage `0.2`: threshold `0.24392068517543838`
- Target candidate percentage `0.3`: threshold `0.20166081968076038`
- Target candidate percentage `0.4`: threshold `0.17943432358783462`
- Target candidate percentage `0.5`: threshold `0.16260626611483847`
- Target candidate percentage `0.6`: threshold `0.14766174578846247`
- Target candidate percentage `0.7`: threshold `0.13378694790016313`

## 4. Full-Dataset Screening Results

- target=0.1 | split=train | actual_candidate_pct=0.095269 | fraud_retention=0.810855 | candidate_fraud_prevalence=0.006464 | enrichment=8.511171066454832
- target=0.1 | split=valid | actual_candidate_pct=0.100377 | fraud_retention=0.848097 | candidate_fraud_prevalence=0.008753 | enrichment=8.449120846159547
- target=0.1 | split=test | actual_candidate_pct=0.096966 | fraud_retention=0.892881 | candidate_fraud_prevalence=0.012836 | enrichment=9.208138629601743
- target=0.2 | split=train | actual_candidate_pct=0.193954 | fraud_retention=0.841816 | candidate_fraud_prevalence=0.003296 | enrichment=4.34027942879992
- target=0.2 | split=valid | actual_candidate_pct=0.200676 | fraud_retention=0.873917 | candidate_fraud_prevalence=0.004512 | enrichment=4.3548715751313685
- target=0.2 | split=test | actual_candidate_pct=0.196270 | fraud_retention=0.912713 | candidate_fraud_prevalence=0.006483 | enrichment=4.650280942834235
- target=0.3 | split=train | actual_candidate_pct=0.290830 | fraud_retention=0.866775 | candidate_fraud_prevalence=0.002264 | enrichment=2.9803519294544145
- target=0.3 | split=valid | actual_candidate_pct=0.300663 | fraud_retention=0.892868 | candidate_fraud_prevalence=0.003077 | enrichment=2.96966704524492
- target=0.3 | split=test | actual_candidate_pct=0.294452 | fraud_retention=0.926652 | candidate_fraud_prevalence=0.004387 | enrichment=3.147042851065289
- target=0.4 | split=train | actual_candidate_pct=0.386873 | fraud_retention=0.890001 | candidate_fraud_prevalence=0.001747 | enrichment=2.300502591266484
- target=0.4 | split=valid | actual_candidate_pct=0.401147 | fraud_retention=0.910833 | candidate_fraud_prevalence=0.002352 | enrichment=2.270571686416423
- target=0.4 | split=test | actual_candidate_pct=0.390318 | fraud_retention=0.939138 | candidate_fraud_prevalence=0.003354 | enrichment=2.4060836720753493
- target=0.5 | split=train | actual_candidate_pct=0.480298 | fraud_retention=0.911490 | candidate_fraud_prevalence=0.001441 | enrichment=1.8977598105671258
- target=0.5 | split=valid | actual_candidate_pct=0.500011 | fraud_retention=0.927737 | candidate_fraud_prevalence=0.001922 | enrichment=1.855433190160923
- target=0.5 | split=test | actual_candidate_pct=0.483013 | fraud_retention=0.951013 | candidate_fraud_prevalence=0.002745 | enrichment=1.9689172788774423
- target=0.6 | split=train | actual_candidate_pct=0.574645 | fraud_retention=0.932731 | candidate_fraud_prevalence=0.001233 | enrichment=1.6231435428026924
- target=0.6 | split=valid | actual_candidate_pct=0.600409 | fraud_retention=0.945404 | candidate_fraud_prevalence=0.001631 | enrichment=1.5746003157186632
- target=0.6 | split=test | actual_candidate_pct=0.577111 | fraud_retention=0.963154 | candidate_fraud_prevalence=0.002326 | enrichment=1.6689238586514856
- target=0.7 | split=train | actual_candidate_pct=0.669235 | fraud_retention=0.953595 | candidate_fraud_prevalence=0.001082 | enrichment=1.424904071105571
- target=0.7 | split=valid | actual_candidate_pct=0.701287 | fraud_retention=0.962517 | candidate_fraud_prevalence=0.001422 | enrichment=1.3725011920270684
- target=0.7 | split=test | actual_candidate_pct=0.671040 | fraud_retention=0.975384 | candidate_fraud_prevalence=0.002026 | enrichment=1.4535410264052102

## 5. Candidate Pool Sample

- Selected target candidate percentage: **0.1**
- Candidate parts written: **5215**
- Sample rows kept: **528,842**
- Seen rows: **430,920,901**
- Positive policy: **all flagged positives retained**
- Negative sample fraction: **0.005**

## 6. Candidate Pool Counts

- test | label 0: **30,876**
- test | label 1: **80,453**
- train | label 0: **142,965**
- train | label 1: **185,762**
- valid | label 0: **31,994**
- valid | label 1: **56,792**

## 7. Interpretation

Phase 3B should be used as the stronger Stage 1 candidate generator if it achieves substantially higher fraud retention than the Phase 3 Isolation Forest baseline. The candidate pool created here is the recommended input for Phase 4 reduced-feature classical, quantum, and hybrid comparisons. Phase 3 remains useful as an unsupervised baseline, while Phase 3B provides the operationally stronger high-recall screening track.