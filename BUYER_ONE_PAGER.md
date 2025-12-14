# ProofLink: Acquisition Brief

## What It Is

ProofLink is a reference implementation for automated reconciliation and compliance verification in 401(k) retirement plans. The system compares payroll contribution data against recordkeeper transaction data to detect discrepancies, timing violations, and regulatory compliance issues. It includes vendor-agnostic data normalization, automated violation detection for Secure 2.0, IRS 402(g) limits, and eligibility rules, and generates audit-ready evidence packs with detailed exception reports.

## What's Included

- **Codebase**: Python reference implementation (~3,000 lines) with modular architecture
- **Core Engine**: Reconciliation logic comparing payroll data against recordkeeper transactions
- **Compliance Analyzers**: Six specialized modules (Secure 2.0, 402(g) limits, eligibility drift, timing violations, match reasonableness, reconciliation mismatches)
- **Vendor Detection**: Automated identification and normalization for multiple payroll systems and recordkeeper formats
- **Evidence Generation**: Automated creation of audit-ready evidence packs with detailed exception reports
- **Integration Components**: FastAPI REST backend, database schema (SQLite), Streamlit reference UI
- **Test Suite**: Smoke tests and sample data demonstrating core functionality
- **Architecture Documentation**: Modular design documentation

## Who Should Buy It

- **Recordkeeping Platforms**: Embed reconciliation capability into existing infrastructure
- **Payroll Software Companies**: Integrate retirement plan compliance verification into payroll processing workflows
- **Retirement Plan Administration Platforms**: Add automated compliance monitoring to existing administration tooling
- **Compliance and Audit Software Providers**: Extend product capabilities with retirement-specific compliance verification logic
- **PE-Backed Platforms**: Acquire reference implementation to accelerate internal development

## Build vs Buy Rationale

**Domain Complexity**: IRS regulations (Secure 2.0, 402(g), EPCRS), vendor format variations, and timing rules require deep retirement plan expertise. Building from scratch requires 12-18 months of development plus ongoing regulatory updates.

**Technical Risk**: Vendor detection, data normalization, and edge case handling represent non-trivial engineering challenges. The reference implementation's vendor-agnostic approach and confidence scoring reduce integration risk.

**Time-to-Market**: Acquiring eliminates development cycle. Estimated 6-12 month acceleration versus internal build, with reduced technical validation risk.

**Cost Efficiency**: Development cost (engineer time, QA, regulatory research) typically exceeds acquisition cost. Maintenance burden (regulatory updates, vendor changes) is ongoing regardless of build approach.

**Validated Approach**: Working prototype demonstrates core reconciliation and compliance logic with sample data. Reduces technical validation risk compared to greenfield development.

## Deal Structure Notes

Asset sale of IP and reference implementation codebase. No customers, revenue, or operational infrastructure included. Pre-revenue prototype with no commercial usage. Estimated 3-4 month integration timeline with 2-3 engineers. Optional 2-4 week founder handoff window for knowledge transfer available. Buyer assumes full responsibility for integration, validation, compliance, and security.

