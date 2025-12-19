# ProofLink: Buyer Handoff Guide

## Quick Start

### Prerequisites

- Python 3.8+ installed
- pip package manager
- (Optional) Virtual environment recommended

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import pandas, fastapi, streamlit; print('Dependencies OK')"
   ```

## Running the System

### 1. Run Tests

Execute the smoke test suite to verify core functionality:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_reconciliation_smoke.py -v
```

**Expected**: All tests should pass. Tests verify:
- Basic reconciliation functionality
- Secure 2.0 compliance checks
- Timing analysis
- Eligibility drift detection

### 2. Run Streamlit UI

The Streamlit interface provides a reference UI for running reconciliations:

```bash
streamlit run streamlit_app.py
```

**Access**: Open browser to `http://localhost:8501`

**Features**:
- Upload payroll and recordkeeper CSV files
- Run reconciliation with real-time progress
- View reconciliation results and exception reports
- Download evidence packs

**Environment Variables** (optional):
- None required for basic operation

### 3. Run FastAPI Backend

The REST API provides programmatic access to reconciliation:

```bash
# Start the API server
uvicorn api:app --reload

# Or with custom host/port
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Access**: API available at `http://localhost:8000`

**Endpoints**:
- `POST /api/v1/runs` - Create reconciliation run
- `GET /api/v1/runs` - List runs
- `GET /api/v1/runs/{run_id}` - Get run details
- `GET /api/v1/runs/{run_id}/evidence-pack` - Download evidence pack
- `POST /auth/register` - Register user
- `POST /auth/login` - Login user

**Environment Variables**:
- `JWT_SECRET_KEY` (optional): Secret key for JWT tokens (defaults to insecure placeholder)
- `UPLOADS_DIR` (optional): Directory for API uploads (defaults to "api_uploads")

**API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs

### 4. Run Preflight Checks

Before running reconciliation, validate input files:

```bash
# Using environment variables
$env:PAYROLL_CSV_PATH="data/raw/demo_clean_payroll.csv"
$env:RECORDKEEPER_CSV_PATH="data/raw/demo_clean_rk.csv"
python preflight.py

# Or with explicit paths
python preflight.py data/raw/demo_clean_payroll.csv data/raw/demo_clean_rk.csv mapping_example.yaml
```

**Output**: Preflight report showing:
- Mapped fields
- Missing required fields
- Join-key coverage
- Warnings

## Environment Variables Reference

### Required for API
- None (uses defaults)

### Optional
- `JWT_SECRET_KEY`: Secret key for JWT authentication (set for production)
- `UPLOADS_DIR`: Directory for API file uploads (default: "api_uploads")
- `MAPPING_YAML_PATH`: Path to column mapping YAML (default: "mapping_example.yaml")

### For Preflight CLI
- `PAYROLL_CSV_PATH`: Path to payroll CSV file
- `RECORDKEEPER_CSV_PATH`: Path to recordkeeper CSV file
- `MAPPING_YAML_PATH`: Path to mapping YAML file (default: "mapping_example.yaml")

## Sample Data

Demo files are included in `data/demo/` and `data/raw/`:
- `demo_clean_payroll.csv` / `demo_clean_rk.csv` - Clean format examples
- `demo_messy_payroll.csv` / `demo_messy_rk.csv` - Variant column names
- `demo_secure20_payroll.csv` / `demo_secure20_rk.csv` - Secure 2.0 test data

## Key Files

- `main.py` - Core reconciliation engine
- `preflight.py` - Pre-execution validation
- `mapping_example.yaml` - Column mapping template
- `api.py` - FastAPI REST backend
- `streamlit_app.py` - Reference UI
- `CANONICAL_SCHEMA.md` - Field schema documentation

## Troubleshooting

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.8+)

### Preflight Fails
- Check CSV files exist and are readable
- Verify `mapping_example.yaml` exists
- Review preflight report for missing fields

### API Errors
- Check database file permissions (`prooflink_runs.db`)
- Verify upload directory is writable
- Review API logs for detailed error messages

### Streamlit Issues
- Clear browser cache if UI doesn't update
- Check console for Python errors
- Verify input files are valid CSV format

## Next Steps

1. Review `CANONICAL_SCHEMA.md` for field requirements
2. Customize `mapping_example.yaml` for your data formats
3. Run preflight checks on your data files
4. Integrate reconciliation engine into your platform
5. Customize compliance analyzers for your requirements
