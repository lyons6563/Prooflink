# SECURE 2.0 Preflight MVP

## Goal
CLI tool that takes a payroll CSV and a YAML config and outputs SAFE / NOT SAFE plus an exception CSV.

## Inputs
- Payroll CSV
- YAML config

## Outputs
- Console summary (SAFE / NOT SAFE, violation count, top employee_ids)
- CSV: preflight_outputs/<timestamp>/secure20_preflight_exceptions.csv

## Phase 1 Checks (ONLY)
- Roth-only catch-up violations for HCEs
- Potential HCE detection via annualized compensation

## Explicit Non-Goals
- Streamlit UI
- Database or run history
- Evidence packs
- Integrations
- Full SECURE 2.0 suite (Phase 2+)

## CLI Contract
python secure20_preflight.py --payroll <path> --config <path>

Exit Codes:
- 0 = SAFE
- 2 = NOT SAFE
