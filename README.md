A deterministic evidence engine for payroll ↔ recordkeeper reconciliation.

This repository corresponds to the "Evidence Pack v2" packaged implementation.

## What this is

Evidence Pack v2 is a deterministic reconciliation engine that compares payroll and recordkeeper data to identify discrepancies, timing violations, and compliance issues. It produces an immutable Evidence Pack ZIP archive containing reconciliation results with cryptographic integrity verification.

## What this is not

- Not a legal opinion, legal advice, or legal interpretation system
- Not a compliance certification or audit endorsement tool
- Not a fiduciary judgment system
- Not a system of record for source data
- Not an audit endorsement or audit approval system

This system performs rule-based analysis only and makes no compliance, legal, or fiduciary assertions.

## Core capabilities

The engine performs the following reconciliation checks:

- Deferral reconciliation: Compares deferral amounts between payroll and recordkeeper data
- Loan reconciliation: Compares loan transactions between payroll and recordkeeper data
- Timing analysis: Identifies late deferrals and contributions based on configured thresholds
- Secure 2.0 catch-up analysis: Analyzes catch-up contribution compliance (if applicable)
- Eligibility drift analysis: Detects changes in employee eligibility status
- Vendor detection: Identifies payroll and recordkeeper vendors from data patterns

All analysis is deterministic and rule-based. Outputs include detailed mismatch reports, violation summaries, and integrity-verified Evidence Pack archives.

## Intended use

Evidence Pack v2 is intended for use by auditors, compliance teams, and internal review teams as supporting documentation in audit and review processes. Evidence Packs can be included in audit workpapers to document:

- What data was analyzed during a specific period
- What reconciliation rules were applied
- What discrepancies were identified
- The integrity and immutability of the analysis record

## How to run

Run preflight validation with payroll, recordkeeper, and mapping files:

```bash
python preflight.py inputs/payroll.csv inputs/recordkeeper.csv inputs/mapping_demo.yaml
```

Note: `mapping_example.yaml` is documentation-only and may not pass preflight.

The command validates inputs and reports whether the run is safe to proceed.

## Example output

The Evidence Pack ZIP (`evidence_pack_{run_id}.zip`) contains:

- `manifest.json`
- `audit_summary.txt`
- `README.txt`
- `results.csv` (if available)
- `violations.csv` (if available)

## IP & licensing

To be verified by buyer.

## Why this exists

Evidence Pack v2 provides a deterministic, verifiable record of reconciliation analysis for audit and compliance purposes. The cryptographic integrity features enable independent verification that Evidence Pack contents have not been modified since creation.

