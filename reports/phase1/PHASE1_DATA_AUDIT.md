# Phase 1 Data Processing Report

## 1. Dataset Inventory

- CSV files found: **12**
- Total raw CSV size: **38.7334 GB**

## 2. Full Dataset Size

- Total transactions processed: **430,920,901**
- Number of columns: **14**
- Raw size: **38.751 GB**
- Parquet size: **8.88 GB**
- Available system RAM during profiling: **31.3506 GB**

## 3. Laundering Label Distribution

- Label `0`: **430,534,738** rows (99.910387%)
- Label `1`: **386,163** rows (0.089613%)

## 4. Time Range

- Minimum timestamp: **2022-08-01 00:00:00**
- Maximum timestamp: **2023-01-12 16:49:00**

## 5. Transaction Type Column

- Detected transaction type column: **payment_format**

## 6. Sender / Receiver / Account Fields

- possible_sender_fields: `['from_bank', 'account']`
- possible_receiver_fields: `['to_bank', 'account_duplicated_0']`
- possible_bank_fields: `['from_bank', 'to_bank']`
- possible_account_fields: `['account', 'account_duplicated_0']`

## 7. Duplicate Check

```json
{
  "duplicate_check": "skipped"
}
```

## 8. Temporal Split

- Split type: **temporal**
- Train cutoff: **2022-09-30T14:06:00**
- Validation cutoff: **2022-10-19T18:24:00**

### Split Counts

- test | label 0: **64,547,333** rows
- test | label 1: **90,105** rows
- train | label 0: **301,417,307** rows
- train | label 1: **229,094** rows
- valid | label 0: **64,570,098** rows
- valid | label 1: **66,964** rows

## 9. Phase 1 Status

Phase 1 is complete if these files exist:
- `dataset_inventory.csv`
- `dataset_inventory.json`
- `conversion_log.json`
- `label_counts.csv`
- `missing_values.csv`
- `transaction_type_counts.csv`
- `temporal_split_counts.csv`
- `temporal_split_ranges.csv`
- `temporal_split_manifest.json`
- `PHASE1_DATA_AUDIT.md`