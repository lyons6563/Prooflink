# ProofLink: Asset Brief

## Overview

ProofLink is a reference implementation for automated reconciliation and compliance verification in 401(k) retirement plans. The system compares payroll contribution data against recordkeeper transaction data to detect discrepancies, timing violations, and regulatory compliance issues.

## Core Capabilities

- **Automated Reconciliation**: Compares payroll vs recordkeeper contributions with vendor-agnostic data normalization
- **Compliance Verification**: Automated checks for Secure 2.0 catch-up rules, IRS 402(g) limits, eligibility drift, and timing violations
- **Evidence Generation**: Creates audit-ready evidence packs with detailed exception reports
- **Vendor Detection**: Automatic identification and normalization for multiple payroll systems (ADP, Paychex, Paylocity, etc.) and recordkeeper formats
- **Preflight Validation**: Safety checks ensure data quality before reconciliation runs

## Technical Stack

- **Language**: Python 3.x
- **Core Libraries**: pandas, numpy, PyYAML
- **API**: FastAPI with SQLite database
- **UI**: Streamlit reference interface
- **Testing**: pytest with smoke tests

## What's Included

- **Source Code**: ~3,000 lines of Python code with modular architecture
- **Core Engine**: Reconciliation logic (`main.py`, `run_reconciliation()`)
- **Compliance Analyzers**: Six specialized modules for regulatory checks
- **Vendor Detection**: Automated format detection and column mapping
- **Integration Components**: FastAPI REST backend, database schema, Streamlit UI
- **Test Suite**: Smoke tests with sample data
- **Documentation**: Schema documentation, mapping examples, handoff guides

## Target Buyers

- Recordkeeping platforms seeking to embed reconciliation capability
- Payroll software companies adding compliance verification
- Retirement plan administration platforms
- Compliance and audit software providers
- PE-backed platforms accelerating development

## Integration Estimate

- **Timeline**: 3-4 months with 2-3 engineers
- **Effort**: Depends on buyer data formats, compliance scope, and security hardening requirements
- **Support**: Optional 2-4 week founder handoff window available

## Deal Structure

Asset sale of IP and reference implementation codebase. Pre-revenue prototype with no commercial usage. Buyer assumes full responsibility for integration, validation, compliance, and security.
