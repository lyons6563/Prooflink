# Canonical Schema Documentation

This document describes the canonical field names, data types, and normalization assumptions used by the ProofLink reconciliation engine for payroll and recordkeeper data ingestion.

**Important**: This documentation reflects the current implementation as of the codebase inspection. It does not propose changes or new fields—only documents what the code actually expects today.

---

## Table of Contents

1. [Payroll Schema](#payroll-schema)
2. [Recordkeeper Schema](#recordkeeper-schema)
3. [Derived Fields](#derived-fields)
4. [Normalization Assumptions](#normalization-assumptions)
5. [Vendor-Specific Mappings](#vendor-specific-mappings)

---

## Payroll Schema

### Required Fields

| Canonical Name | Data Type | Description | Notes |
|---------------|-----------|-------------|-------|
| `employee_id` | string | Unique employee identifier | Must be present on both payroll and recordkeeper. Normalized to string and stripped of whitespace. |

### Optional Fields (Core Reconciliation)

| Canonical Name | Data Type | Required For | Default Behavior | Notes |
|---------------|-----------|--------------|------------------|-------|
| `pay_date` | datetime | Timing analysis, late contribution detection | If missing, timing analysis is limited | Parsed with `pd.to_datetime()`, coerced to `NaT` on parse errors |
| `EE Deferral $` | float | Deferral reconciliation | Defaults to 0.0 if missing | Also normalized as `payroll_pretax`. Parsed as numeric, coerced to 0.0 on errors |
| `EE Roth $` | float | Deferral reconciliation | Defaults to 0.0 if missing | Also normalized as `payroll_roth`. Parsed as numeric, coerced to 0.0 on errors |
| `loan_amount` | float | Loan reconciliation | Defaults to 0.0 if missing | Also normalized as `payroll_loan`. Parsed as numeric, coerced to 0.0 on errors |

### Optional Fields (Secure 2.0 Compliance)

| Canonical Name | Data Type | Required For | Default Behavior | Notes |
|---------------|-----------|--------------|------------------|-------|
| `is_hce` | boolean | Secure 2.0 catch-up analysis | Defaults to `False` if missing | Normalized from string values: "true"/"1"/"y"/"yes" → `True`, "false"/"0"/"n"/"no" → `False` |
| `catchup_pretax` | float | Secure 2.0 catch-up analysis | Defaults to 0.0 if missing | Parsed as numeric, coerced to 0.0 on errors |
| `catchup_roth` | float | Secure 2.0 catch-up analysis | Defaults to 0.0 if missing | Parsed as numeric, coerced to 0.0 on errors |

### Optional Fields (Eligibility Drift Detection)

| Canonical Name | Data Type | Required For | Default Behavior | Notes |
|---------------|-----------|--------------|------------------|-------|
| `employment_status` | string | Eligibility drift detection | If missing, drift detection is skipped | Used to identify terminated employees |
| `termination_date` | datetime | Eligibility drift detection | If missing, drift detection is skipped | Parsed with `pd.to_datetime()`, coerced to `NaT` on parse errors |

**Note**: Eligibility drift detection requires all of: `employee_id`, `pay_date`, `def_amount`, `roth_amount`, `employment_status`, `termination_date`. If any are missing, the detection gracefully skips (returns empty DataFrame).

---

## Recordkeeper Schema

### Required Fields

| Canonical Name | Data Type | Description | Notes |
|---------------|-----------|-------------|-------|
| `employee_id` | string | Unique employee/participant identifier | Must be present on both payroll and recordkeeper. Normalized to string and stripped of whitespace. |

### Optional Fields (Core Reconciliation)

| Canonical Name | Data Type | Required For | Default Behavior | Notes |
|---------------|-----------|--------------|------------------|-------|
| `deposit_date` | datetime | Timing analysis, late contribution detection | If missing, timing analysis is limited | Parsed with `pd.to_datetime()`, coerced to `NaT` on parse errors |
| `EE Deferral $` | float | Deferral reconciliation | Defaults to 0.0 if missing | Also normalized as `rk_pretax`. Parsed as numeric, coerced to 0.0 on errors |
| `EE Roth $` | float | Deferral reconciliation | Defaults to 0.0 if missing | Also normalized as `rk_roth`. Parsed as numeric, coerced to 0.0 on errors |
| `loan_amount` | float | Loan reconciliation | Defaults to 0.0 if missing | Also normalized as `rk_loan`. Parsed as numeric, coerced to 0.0 on errors |

---

## Derived Fields

These fields are computed during processing and are not expected in input files:

| Canonical Name | Data Type | Computation | Notes |
|---------------|-----------|-------------|-------|
| `deferral_amount` | float | `pretax + roth` (sum of EE Deferral $ + EE Roth $) | Computed separately for payroll and recordkeeper sides |
| `loan_amount` | float | Direct mapping if present, otherwise 0.0 | May be derived from `payroll_loan` or `rk_loan` |

---

## Normalization Assumptions

### Column Name Normalization

1. **Case Insensitivity**: All column name matching is case-insensitive.
2. **Whitespace Normalization**: 
   - Leading/trailing whitespace is stripped
   - Internal whitespace (spaces, tabs) is normalized to underscores for matching
   - Example: `"Employee ID"` → `"employee_id"` for matching purposes
3. **Variant Mapping**: The `normalize_column_names()` function maps common variants to canonical names:
   - Deferral variants → `"EE Deferral $"`
   - Roth variants → `"EE Roth $"`
   - Loan variants remain as `loan_amount` (not normalized by this function)

### Data Type Normalization

1. **Numeric Fields**: 
   - Parsed with `pd.to_numeric(..., errors="coerce")`
   - Invalid values become `NaN`, then filled with `0.0`
   - Examples: `"$1,234.56"` → `1234.56`, `"invalid"` → `0.0`

2. **Date Fields**:
   - Parsed with `pd.to_datetime(..., errors="coerce")`
   - Invalid values become `NaT` (Not a Time)
   - Examples: `"2024-01-15"` → datetime, `"invalid"` → `NaT`

3. **Boolean Fields** (`is_hce`):
   - String values normalized: `"true"`, `"1"`, `"y"`, `"yes"` → `True`
   - String values normalized: `"false"`, `"0"`, `"n"`, `"no"` → `False`
   - Converted to boolean dtype

4. **String Fields** (`employee_id`):
   - Converted to string dtype: `.astype(str)`
   - Stripped of whitespace: `.str.strip()`

### Fallback Behavior

- **Missing Required Fields**: Reconciliation will fail if `employee_id` is missing on either side.
- **Missing Optional Fields**: 
  - Amount fields default to `0.0`
  - Date fields default to `NaT` (timing analysis may be limited)
  - Boolean fields default to `False`
- **Missing Secure 2.0 Fields**: Analysis continues with defaults; warnings are printed if fields are missing.

---

## Vendor-Specific Mappings

The system supports vendor-specific column mappings through `vendors.py` and `vendor_detection.py`. The following mappings are defined:

### Payroll Vendors

Vendor signatures are defined in `PAYROLL_VENDOR_SIGNATURES` with mappings for:
- ADP
- Paychex
- Paylocity
- QuickBooks
- Workday
- UKG
- BambooHR
- TriNet

Each vendor mapping includes:
- `signature_keywords`: Keywords used for vendor detection
- `column_map`: Maps canonical names to vendor-specific column name variants

**Example (ADP)**:
```python
"ADP": {
    "signature_keywords": ["emp id", "check date", "ee deferral", "ee roth"],
    "column_map": {
        "employee_id": ["emp id", "employee_id", "emp_id", "employee number"],
        "pay_date": ["check date", "pay date", "payroll date", "check_date"],
        "EE Deferral $": ["ee deferral", "ee deferral $", "deferral", "pretax deferral"],
        "EE Roth $": ["ee roth", "ee roth $", "roth deferral", "roth"],
        "loan_amount": ["loan repay", "loan payment", "loan repay $"],
    }
}
```

### Recordkeeper Vendors

Vendor signatures are defined in `RK_VENDOR_SIGNATURES` with mappings for:
- VENDOR_RK_1
- Fidelity
- Vanguard

**Example (VENDOR_RK_1)**:
```python
"VENDOR_RK_1": {
    "signature_keywords": ["part_id", "post_date", "ee_pretax", "ee_roth"],
    "column_map": {
        "employee_id": ["part_id", "participant id", "employee_id"],
        "deposit_date": ["post_date", "post date", "deposit date"],
        "EE Deferral $": ["ee_pretax", "ee pretax", "pretax contribution"],
        "EE Roth $": ["ee_roth", "ee roth", "roth contribution"],
        "loan_amount": ["loan_contr", "loan contribution", "loan payment"],
    }
}
```

### Generic Fallback

If vendor detection fails or confidence is low (< 0.65), the system falls back to generic column mapping using `COLUMN_MAP` in `main.py`. This map includes extensive variants for each canonical field name.

**Generic Column Map Examples**:
- `employee_id`: `["employee_id", "employee id", "ee id", "emp id", "empid", "empnumber", "employee_number", "participant id", "participant_id", "participant", "part_id"]`
- `pay_date`: `["pay_date", "pay date", "payroll date", "check date", "checkdt", "payroll_run_date", "pay period end date", "pay period date", "date"]`
- `deposit_date`: `["deposit_date", "deposit date", "recordkeeper date", "trade date", "post_date", "posting date", "funding date", "contribution date", "transaction_effective_date"]`

---

## Reconciliation Logic Requirements

### Deferral Reconciliation

**Required on Payroll**:
- `employee_id` (required)
- At least one of: `EE Deferral $` or `EE Roth $` (or both, summed as `deferral_amount`)

**Required on Recordkeeper**:
- `employee_id` (required)
- At least one of: `EE Deferral $` or `EE Roth $` (or both, summed as `deferral_amount`)

**Logic**: 
- Total deferral = `pretax + roth` (if both present) or single amount column
- Reconciliation aggregates by `employee_id` before comparing

### Loan Reconciliation

**Required on Payroll**:
- `employee_id` (required)
- `loan_amount` (optional, defaults to 0.0)

**Required on Recordkeeper**:
- `employee_id` (required)
- `loan_amount` (optional, defaults to 0.0)

**Logic**: 
- If payroll has loans but recordkeeper has no loan column, recordkeeper loans are treated as 0.0
- Reconciliation aggregates by `employee_id` before comparing

### Timing Analysis

**Required on Payroll**:
- `employee_id` (required)
- `pay_date` (optional but recommended)

**Required on Recordkeeper**:
- `employee_id` (required)
- `deposit_date` (optional but recommended)

**Logic**: 
- Compares `pay_date` vs `deposit_date` to detect late contributions
- Default threshold: 5 business days (configurable via `MAX_BUSINESS_DAYS_LAG`)
- If dates are missing, timing analysis is limited

---

## Field Name Variants (Generic Mapping)

The following variants are recognized for each canonical field (case-insensitive, whitespace-normalized):

### `employee_id` Variants
- `employee_id`, `employee id`, `ee id`, `emp id`, `empid`, `empnumber`, `employee_number`
- `participant id`, `participant_id`, `participant`, `part_id`

### `pay_date` Variants
- `pay_date`, `pay date`, `payroll date`, `check date`, `checkdt`
- `payroll_run_date`, `pay period end date`, `pay period date`, `date`

### `deposit_date` Variants
- `deposit_date`, `deposit date`, `recordkeeper date`, `trade date`
- `post_date`, `posting date`, `funding date`, `contribution date`, `transaction_effective_date`

### `EE Deferral $` / `payroll_pretax` Variants
- `pretax_defl`, `ee deferral $`, `ee deferral`, `EE Deferral $`
- `employee contribution`, `pre-tax`, `pre tax`, `amount`
- `457b_ee_pretax_amt`, `ee_pretax_def`, `payroll_pretax`

### `EE Roth $` / `payroll_roth` Variants
- `roth_defl`, `ee roth $`, `ee roth`, `EE Roth $`
- `roth contribution`, `roth`
- `457b_ee_roth_amt`, `ee_roth_def`, `payroll_roth`

### `loan_amount` Variants (Payroll)
- `loan_pmt`, `loan repay $`, `loan repayment`
- `457b_loan_repay_amt`, `loan_repayment`

### `loan_amount` Variants (Recordkeeper)
- `loan_contr`, `loan repayment`, `loan_repayment`

### `is_hce` Variants
- `is_hce`, `is hce`, `hce_flag`, `hce flag`, `hce`
- `highly_compensated`, `highly compensated`, `highly compensated employee`

### `catchup_pretax` Variants
- `catchup_pretax`, `catchup pretax`, `catch_up_pretax`, `catch up pretax`
- `pretax catchup`, `pretax catch-up`, `catchup_contribution_pretax`

### `catchup_roth` Variants
- `catchup_roth`, `catchup roth`, `catch_up_roth`, `catch up roth`
- `roth catchup`, `roth catch-up`, `catchup_contribution_roth`

---

## Notes on Implementation

1. **Vendor Detection**: The system attempts to detect vendor format automatically using signature keywords and column patterns. Confidence scores are calculated (0.0 to 1.0). If confidence < 0.65, the system falls back to generic mapping.

2. **Strict vs. Flexible Mapping**: 
   - Known vendors with high confidence (≥ 0.65) may enforce strict column requirements
   - Generic/unknown vendors use flexible mapping with extensive variant matching

3. **Single-Amount Fallback**: If separate pretax/roth columns are not found, the system can use a single `amount` column as a fallback (treated as pretax, with roth = 0.0).

4. **Column Name Normalization Order**:
   1. `normalize_column_names()` maps common variants to `"EE Deferral $"` and `"EE Roth $"`
   2. Vendor-specific mapping (if vendor detected with high confidence)
   3. Generic `COLUMN_MAP` fallback for remaining fields
   4. Direct column name matching (case-insensitive)

5. **Error Handling**: The system is designed to be resilient:
   - Missing optional fields default to safe values (0.0, False, NaT)
   - Invalid data types are coerced rather than causing failures
   - Warnings are printed for missing fields that affect functionality

---

## Summary

**Minimum Required Fields**:
- Payroll: `employee_id`
- Recordkeeper: `employee_id`

**Recommended Fields for Full Functionality**:
- Payroll: `employee_id`, `pay_date`, `EE Deferral $` (or `EE Roth $`), `loan_amount` (if applicable)
- Recordkeeper: `employee_id`, `deposit_date`, `EE Deferral $` (or `EE Roth $`), `loan_amount` (if applicable)

**Optional Fields for Advanced Features**:
- Secure 2.0: `is_hce`, `catchup_pretax`, `catchup_roth`
- Eligibility Drift: `employment_status`, `termination_date`

All fields support extensive variant naming through vendor-specific mappings and generic fallback logic.
