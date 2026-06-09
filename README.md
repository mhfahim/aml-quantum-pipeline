# AML Quantum Pipeline

This repository contains the full experimental pipeline for a two-stage AML fraud detection research project.

## Core idea

Stage 1: Classical anomaly detection filters the full IBM AML transaction dataset.

Stage 2: Quantum models classify or analyze the high-suspicion candidate pool.

## Phase 1

Full dataset processing:

- dataset inventory
- CSV to partitioned Parquet conversion
- label distribution
- missing values
- duplicate checks
- temporal train/validation/test split

## Important

Raw dataset files and large Parquet outputs must not be pushed to GitHub.
Only code and small reports should be committed.
