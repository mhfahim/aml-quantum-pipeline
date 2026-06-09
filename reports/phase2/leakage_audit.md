# Phase 2 Leakage Audit

## Status

Completed in lightweight mode because the full Parquet scan caused Kaggle memory restart.

## Leakage Controls

- Label-like columns are blocked.
- Raw account, bank, entity, name, ID, and source fields are blocked.
- Temporal split is inherited from Phase 1.
- Starter baselines use conservative transaction-level features only.

## Suspicious Label-Like Columns

- `is_laundering`
- `_label`

## Blocked Identifier Columns

- `from_bank`
- `account`
- `to_bank`
- `account_duplicated_0`
- `amount_paid`
- `_source_file`

## Safe Starter Feature Candidates

- `amount_received`
- `receiving_currency`
- `payment_currency`
- `payment_format`

## Dataset Summary From Phase 1

- Total rows: **430,920,901**
- Time range: **{'min_timestamp': '2022-08-01 00:00:00', 'max_timestamp': '2023-01-12 16:49:00'}**

## Interpretation

This audit establishes a conservative leakage-safe baseline feature set. Deeper leakage checks will be added later when rolling, graph, and account-history features are engineered.