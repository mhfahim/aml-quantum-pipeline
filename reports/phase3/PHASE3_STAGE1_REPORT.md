# Phase 3 Report: Stage 1 Classical Anomaly Filtering

## 1. Purpose

Phase 3 implements the first stage of the two-stage AML pipeline. A classical anomaly filter screens the full transaction dataset and produces a high-suspicion candidate pool for later quantum and hybrid analysis.

## 2. Stage 1 Model

- Model: **IsolationForest**
- Training data: **sampled legitimate transactions from temporal train split**
- Training sample rows: **120000**
- Features: `['amount_received_num', 'amount_paid_num', 'amount_delta_num', 'amount_ratio_num', 'log_amount_received_num', 'log_amount_paid_num', 'hour_num', 'day_num', 'same_currency_num']`
- Runtime seconds: **78.49**
- Training mode: **memory-safe streaming sample; full dataset was not loaded into RAM**

## 3. Thresholds

- Score definition: higher score = more anomalous.
- Target candidate percentage `0.05`: threshold `-0.0016407908742604727`
- Target candidate percentage `0.1`: threshold `-0.04553405685691104`
- Target candidate percentage `0.15`: threshold `-0.06432426733228178`

## 4. Full-Dataset Screening Results

See `stage1_screening_metrics.csv` for full split-level results.

### Target candidate percentage: 0.05
- train: candidate rows `14,656,680` / `301,646,401` (0.048589), fraud retained `6,131` / `229,094` (0.026762)
- valid: candidate rows `3,235,051` / `64,637,062` (0.050049), fraud retained `1,799` / `66,964` (0.026865)
- test: candidate rows `3,728,786` / `64,637,438` (0.057688), fraud retained `2,318` / `90,105` (0.025726)

### Target candidate percentage: 0.1
- train: candidate rows `29,957,076` / `301,646,401` (0.099312), fraud retained `14,344` / `229,094` (0.062612)
- valid: candidate rows `6,470,360` / `64,637,062` (0.100103), fraud retained `4,484` / `66,964` (0.066961)
- test: candidate rows `8,204,036` / `64,637,438` (0.126924), fraud retained `5,719` / `90,105` (0.063470)

### Target candidate percentage: 0.15
- train: candidate rows `45,039,525` / `301,646,401` (0.149312), fraud retained `24,730` / `229,094` (0.107947)
- valid: candidate rows `9,676,501` / `64,637,062` (0.149705), fraud retained `7,663` / `66,964` (0.114435)
- test: candidate rows `11,615,832` / `64,637,438` (0.179707), fraud retained `9,841` / `90,105` (0.109217)

## 5. Candidate Pool Sample

- Target threshold used: **0.1**
- Candidate parts written: **4038**
- Sample rows kept: **247,632**
- Seen rows: **430,920,901**
- Positive policy: **all flagged positives retained**
- Negative sample fraction: **0.005**

## 6. Candidate Pool Counts

- test | label 0: **41,000**
- test | label 1: **5,719**
- train | label 0: **149,701**
- train | label 1: **14,344**
- valid | label 0: **32,384**
- valid | label 1: **4,484**

## 7. Interpretation

This phase validates the upstream screening layer of the hybrid AML pipeline. The Isolation Forest baseline reduces the transaction universe into a candidate pool, but its fraud-retention performance should be treated as an initial unsupervised baseline rather than the final Stage 1 filter. The candidate pool produced here will be used in Phase 4 for reduced-feature classical, quantum, and hybrid comparisons.