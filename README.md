# Prooflink

Deterministic reconciliation engine for payroll vs. recordkeeper data in 401(k) plans. Produces an integrity-verified Evidence Pack for use in audit workpapers.

## What it does

Prooflink ingests payroll and recordkeeper CSV exports, normalizes them across vendor formats, and runs the following checks:

- **Deferral reconciliation** — compares employee deferral amounts (pre-tax + Roth) between payroll and recordkeeper, flags mismatches and one-sided rows
- **Loan reconciliation** — same comparison for loan repayment transactions
- **Contribution timing** — calculates business-day lag between pay date and deposit date; flags late contributions against a configurable threshold
- **Secure 2.0 catch-up compliance** — identifies HCEs age 50+ with pre-tax catch-up contributions where Roth is required
- **Eligibility drift** — detects employees whose eligibility status changed mid-period
- **402(g) excess deferrals** — checks employee deferrals against annual IRS limits for the inferred plan year
- **Employer match reasonableness** — compares actual match amounts to the plan's formula

Outputs go to a timestamped run directory. All CSVs and the Excel report are bundled into a ZIP with a SHA-256 manifest so the evidence pack can be independently verified without database access.

## Stack

- Python 3.10+
- pandas, openpyxl, PyYAML
- FastAPI (API layer)
- Streamlit (UI)
- pytest

## Layout

```
main.py                         # core reconciliation engine + evidence pack builder
streamlit_app.py                # Streamlit UI
api.py                          # FastAPI endpoints
preflight.py                    # input validation (run before engine)
vendor_detection.py             # signature-based vendor identification
vendors.py                      # vendor column-map definitions
contribution_timing_analyzer_v2.py
secure20_catchup_analyzer.py
eligibility_drift_analyzer.py
comp_402g_analyzer.py
match_reasonableness_analyzer.py
plan_exception_summary.py
tests/                          # pytest test suite
inputs/                         # place payroll + RK CSVs here
data/processed/                 # engine output (overwritten each run)
proofs/                         # proof manifests (append-only)
```

## Running

Validate inputs first:

```bash
python preflight.py inputs/payroll.csv inputs/recordkeeper.csv inputs/mapping_example.yaml
```

Then run via CLI:

```bash
python main.py --payroll_csv inputs/payroll.csv --rk_csv inputs/recordkeeper.csv --plan_name "Acme 401k"
```

Or start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

Windows demo:

```powershell
.\run_demo.ps1
```

## Evidence Pack

Each run produces `prooflink_evidence_pack.zip` under `data/processed/` containing:

- `reconciliation_report.xlsx` (summary + detail sheets)
- `deferral_mismatches.csv`
- `loan_mismatches.csv`
- `late_deferrals_contributions.csv`
- `secure20_exceptions.csv` (if any)
- `proof_manifest_*.json` with SHA-256 hashes of all outputs

The manifest can be verified offline with `verify_proof.py` — no database or network required.

## Tests

```bash
pytest tests/ -v
```
