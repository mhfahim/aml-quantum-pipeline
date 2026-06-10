# Phase 6 Report: IBM Quantum Hardware Validation

## 1. Purpose

Phase 6 validates the reduced-feature quantum AML model on real IBM Quantum hardware. The purpose is not to claim full-scale quantum advantage, but to evaluate hardware feasibility, runtime burden, transpiled circuit resources, and simulator-vs-hardware agreement.

## 2. Hardware Validation Subset

| split   |   label |   n |
|:--------|--------:|----:|
| train   |       0 | 250 |
| train   |       1 | 250 |
| valid   |       0 | 150 |
| valid   |       1 | 150 |
| test    |       0 | 100 |
| test    |       1 | 100 |

Total subset rows: **1000**
Number of qubits: **4**

## 3. Statevector Reference Model

| track                        | model                  | split   |   threshold |   accuracy |   precision |   recall |         f1 |      fpr |       fnr |   tn |    fp |   fn |   tp |   n_rows |   inference_time_seconds | threshold_policy            |   recall_target |
|:-----------------------------|:-----------------------|:--------|------------:|-----------:|------------:|---------:|-----------:|---------:|----------:|-----:|------:|-----:|-----:|---------:|-------------------------:|:----------------------------|----------------:|
| phase6_statevector_reference | phase6_vqc_statevector | train   |   0.0049313 |  0.0524975 |  0.00497512 | 0.952    | 0.00989852 | 0.952    | 0.048     | 2400 | 47600 |   12 |  238 |      500 |                0.0787675 | recall_target_precision_max |             0.8 |
| phase6_statevector_reference | phase6_vqc_statevector | valid   |   0.0049313 |  0.0380763 |  0.00507754 | 0.986667 | 0.0101031  | 0.966667 | 0.0133333 | 1000 | 29000 |    2 |  148 |      300 |                0.0493795 | recall_target_precision_max |             0.8 |
| phase6_statevector_reference | phase6_vqc_statevector | test    |   0.0049313 |  0.0244776 |  0.00467195 | 0.92     | 0.00929669 | 0.98     | 0.08      |  400 | 19600 |    8 |   92 |      200 |                0.0321948 | recall_target_precision_max |             0.8 |

Training time seconds: **10.161038201000338**
Optimizer final loss: **0.03134558419174267**
Trainable parameters: **10**

## 4. Circuit Resource Summary

Untranspiled circuit inventory summary:

|       |   n_qubits |   depth_untranspiled |   size_untranspiled |
|:------|-----------:|---------------------:|--------------------:|
| count |         80 |                   80 |                  80 |
| mean  |          4 |                   13 |                  28 |
| std   |          0 |                    0 |                   0 |
| min   |          4 |                   13 |                  28 |
| 25%   |          4 |                   13 |                  28 |
| 50%   |          4 |                   13 |                  28 |
| 75%   |          4 |                   13 |                  28 |
| max   |          4 |                   13 |                  28 |

Transpiled circuit inventory summary:

|       |   depth_transpiled |   size_transpiled |
|:------|-------------------:|------------------:|
| count |                 80 |                80 |
| mean  |                 95 |               159 |
| std   |                  0 |                 0 |
| min   |                 95 |               159 |
| 25%   |                 95 |               159 |
| 50%   |                 95 |               159 |
| 75%   |                 95 |               159 |
| max   |                 95 |               159 |

## 5. IBM Hardware Job Metadata

- Backend: **ibm_kingston**
- Connected channel: **ibm_quantum_platform**
- Job ID: **d8kaffrqv2lc73851sog**
- Shots: **1024**
- Number of circuits: **80**
- Number of qubits: **4**
- Transpile time seconds: **6.320522669999264**
- Total turnaround time seconds: **33.49553696200019**
- Wait-for-result time seconds: **31.04764909899859**
- IBM usage quantum seconds: **24**
- IBM usage status: **complete**
- IBM BSS seconds: **24**

## 6. Hardware Metrics

| track                          | model                   | split                |   threshold |   accuracy |   precision |   recall |         f1 |   fpr |   fnr |   tn |   fp |   fn |   tp |   n_rows |
|:-------------------------------|:------------------------|:---------------------|------------:|-----------:|------------:|---------:|-----------:|------:|------:|-----:|-----:|-----:|-----:|---------:|
| phase6_ibm_hardware_validation | phase6_vqc_ibm_hardware | hardware_test_subset |   0.0049313 | 0.00497512 |  0.00497512 |        1 | 0.00990099 |     1 |     0 |    0 | 8000 |    0 |   40 |       80 |

## 7. Simulator vs Hardware Agreement

| track                                         | model                                     | split                |   threshold |   accuracy |   precision |   recall |         f1 |   fpr |   fnr |   tn |   fp |   fn |   tp |   n_rows |
|:----------------------------------------------|:------------------------------------------|:---------------------|------------:|-----------:|------------:|---------:|-----------:|------:|------:|-----:|-----:|-----:|-----:|---------:|
| phase6_simulator_reference_on_hardware_subset | phase6_vqc_statevector_on_hardware_subset | hardware_test_subset |   0.0049313 | 0.00460199 |  0.00460371 |    0.925 | 0.00916182 |     1 | 0.075 |    0 | 8000 |    3 |   37 |       80 |
| phase6_ibm_hardware_validation                | phase6_vqc_ibm_hardware                   | hardware_test_subset |   0.0049313 | 0.00497512 |  0.00497512 |    1     | 0.00990099 |     1 | 0     |    0 | 8000 |    0 |   40 |       80 |

Prediction agreement rate: **0.9625**
Mean absolute score difference: **0.0018181603925227642**
Median absolute score difference: **0.0018312052382313082**

## 8. Interpretation

The IBM hardware run confirms that the 4-qubit reduced AML quantum classifier can be executed on real quantum hardware. However, this remains a small-scale feasibility and resource validation experiment. Together with Phases 4 and 5, the result supports a resource-aware conclusion: classical and hybrid models remain more practical for large-scale AML detection, while quantum models are currently more suitable for reduced-feature experimental validation and hardware feasibility testing.