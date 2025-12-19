# ProofLink: Buyer Quick Start Guide

## Overview

This is a **reference implementation** of a 401(k) reconciliation and compliance verification engine. The codebase provides the core logic, analyzers, and preflight validation system. Buyers must build adapters to integrate with their data sources and platforms.

## Preflight Contract

The `preflight.py` module enforces a strict contract before reconciliation runs:

1. **Explicit Mapping Required**: The `MAPPING_YAML_PATH` environment variable must be set, or passed as a parameter. No implicit magic.
2. **Canonical Field Validation**: Input CSVs must map to canonical fields defined in `CANONICAL_SCHEMA.md`.
3. **Required Fields Check**: Preflight validates that required fields are present and mappable.
4. **Join Key Coverage**: Ensures `employee_id` exists and has sufficient coverage for reconciliation.

**Preflight must pass before reconciliation can execute.**

## Required Canonical Fields

See `CANONICAL_SCHEMA.md` for complete documentation. Minimum required fields:

### Payroll
- `employee_id` (required) - Unique employee identifier
- `pay_date` (optional) - Payroll run date
- `def_amount` (optional) - Pre-tax deferral amount
- `roth_amount` (optional) - Roth deferral amount
- `loan_amount` (optional) - Loan payment amount

### Recordkeeper
- `employee_id` (required) - Must match payroll
- `record_date` (optional) - Transaction posting date
- `def_amount` (optional) - Pre-tax contribution amount
- `roth_amount` (optional) - Roth contribution amount
- `loan_amount` (optional) - Loan contribution amount

## Mapping Rules

The mapping system uses **exact match + normalization only**:

1. **Vendor Detection**: Automatically detects vendor format from column signatures
2. **Column Mapping**: Maps vendor-specific columns to canonical fields via `mapping_example.yaml`
3. **Normalization**: Converts data types, strips whitespace, handles missing values
4. **No Transformation Logic**: Mapping is purely structural—no business logic in mapping layer

### Mapping File Format

See `mapping_example.yaml` for the exact structure. The file defines:
- Payroll vendor column mappings
- Recordkeeper vendor column mappings
- Required vs optional fields
- Data type expectations

## What Buyer Must Build

### 1. Data Adapters
- **Payroll Adapter**: Extract data from your payroll system and output CSV with canonical headers (or vendor format that maps to canonical)
- **Recordkeeper Adapter**: Extract data from your recordkeeper system and output CSV with canonical headers (or vendor format that maps to canonical)

### 2. Integration Layer
- Connect `run_reconciliation()` to your data pipeline
- Handle file I/O, error handling, and result storage
- Integrate evidence pack generation into your audit workflow

### 3. UI/API (Optional)
- The included `streamlit_app.py` and `api.py` are reference implementations only
- Build your own UI/API using the core engine functions

### 4. Compliance Customization
- Review and customize analyzers for your specific compliance requirements
- Extend `issue_taxonomy.py` for your issue categorization needs
- Add plan-specific rules as needed

## Installation

### Prerequisites
- **Python 3.8 or higher** (Python 3.9+ recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data processing
- `numpy` - Numerical operations
- `openpyxl` - Excel report generation
- `PyYAML` - Mapping file parsing

## Running the Smoke Demo

### Test Preflight Gate (Good Data - Should Succeed)

Test with valid data that includes required `employee_id` field:

```bash
# Windows PowerShell
$env:MAPPING_YAML_PATH="mapping_example.yaml"
python -c "from main import run_reconciliation; run_reconciliation('demo_inputs/payroll_good.csv', 'demo_inputs/rk_good.csv', mapping_yaml_path='mapping_example.yaml', output_dir='../tmp_output', proofs_dir='../tmp_output/proofs')"

# Linux/Mac
export MAPPING_YAML_PATH="mapping_example.yaml"
python -c "from main import run_reconciliation; run_reconciliation('demo_inputs/payroll_good.csv', 'demo_inputs/rk_good.csv', mapping_yaml_path='mapping_example.yaml', output_dir='../tmp_output', proofs_dir='../tmp_output/proofs')"
```

**Expected**: Reconciliation completes successfully (preflight passes).

### Test Preflight Gate (Bad Data - Should Be Blocked)

Test with invalid data missing required `employee_id` field:

```bash
# Windows PowerShell
$env:MAPPING_YAML_PATH="mapping_example.yaml"
python -c "from main import run_reconciliation; run_reconciliation('demo_inputs/payroll_bad.csv', 'demo_inputs/rk_bad.csv', mapping_yaml_path='mapping_example.yaml', output_dir='../tmp_output', proofs_dir='../tmp_output/proofs')"

# Linux/Mac
export MAPPING_YAML_PATH="mapping_example.yaml"
python -c "from main import run_reconciliation; run_reconciliation('demo_inputs/payroll_bad.csv', 'demo_inputs/rk_bad.csv', mapping_yaml_path='mapping_example.yaml', output_dir='../tmp_output', proofs_dir='../tmp_output/proofs')"
```

**Expected**: RuntimeError with message "Reconciliation execution blocked: preflight checks failed" and details about missing `employee_id` field.

### Run Unit Tests
```bash
# Set mapping path before importing
$env:MAPPING_YAML_PATH="mapping_example.yaml"

# Run smoke tests (requires pytest - install separately if needed)
pytest tests/ -v
```

### Sample Input Format

The tests use canonical headers directly. For your adapters, you can either:
1. Output canonical headers directly (recommended)
2. Output vendor format and configure `mapping_example.yaml` to map to canonical

### Example Test Data Structure

See `tests/test_reconciliation_smoke.py` for examples:
- Payroll rows with `employee_id`, `pay_date`, `def_amount`, `roth_amount`, `loan_amount`
- Recordkeeper rows with `employee_id`, `record_date`, `def_amount`, `roth_amount`, `loan_amount`

## Core Functions

### Main Reconciliation
```python
from main import run_reconciliation

results = run_reconciliation(
    payroll_csv="path/to/payroll.csv",
    rk_csv="path/to/rk.csv",
    mapping_yaml_path="mapping_example.yaml",
    output_dir="output/",
    proofs_dir="proofs/"
)
```

### Preflight Check
```python
from preflight import run_preflight

safe, report = run_preflight(
    payroll_csv_path="path/to/payroll.csv",
    recordkeeper_csv_path="path/to/rk.csv",
    mapping_yaml_path="mapping_example.yaml"
)
```

## Key Files

- `main.py` - Core reconciliation engine
- `preflight.py` - Pre-execution validation
- `mapping_example.yaml` - Column mapping template
- `CANONICAL_SCHEMA.md` - Complete field documentation
- `issue_taxonomy.py` - Issue categorization taxonomy
- `tests/` - Smoke test suite

## Next Steps

1. Review `CANONICAL_SCHEMA.md` for field requirements
2. Study `mapping_example.yaml` for mapping structure
3. Run smoke tests to verify installation
4. Build adapters to convert your data to canonical format
5. Integrate `run_reconciliation()` into your platform
6. Customize analyzers for your compliance needs

## Support

This is a reference implementation. Buyer assumes full responsibility for:
- Integration and testing
- Compliance validation
- Security hardening
- Production deployment

Optional handoff support available per deal terms.

