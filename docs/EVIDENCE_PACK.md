# ProofLink Evidence Pack

## What is an Evidence Pack

An Evidence Pack is a self-contained, immutable record of a single reconciliation run. It captures all inputs, outputs, and metadata from a specific execution of the ProofLink reconciliation engine at a fixed point in time.

Each Evidence Pack is designed to be:
- **Self-contained**: All necessary information is included within the pack
- **Immutable**: Once created, the pack cannot be modified without detection
- **Verifiable**: Cryptographic hashes enable integrity verification of all components
- **Portable**: The pack is distributed as a single ZIP archive

Evidence Packs serve as objective documentation of what data was analyzed, what rules were applied, and what discrepancies were identified during a reconciliation run.

## What an Evidence Pack Contains

Each Evidence Pack includes the following components:

### manifest.json

A structured JSON file containing:
- Run identification (run ID, execution timestamp)
- Engine metadata (engine name, version)
- Input file hashes (payroll data, recordkeeper data, mapping configuration)
- Output file hashes (reconciliation results, violation reports)
- Evidence pack ZIP hash (for verifying the entire archive)

The manifest serves as the index and integrity verification record for all other components.

### audit_summary.txt

A plain-text summary document that includes:
- Run ID and execution timestamp
- System status declaration (read-only, non-opinionated)
- Statement that the system performs rule-based analysis only

This document provides human-readable context for the Evidence Pack.

### results.csv / violations.csv

When present, these CSV files contain:
- **results.csv**: Detailed reconciliation results showing matched and unmatched transactions
- **violations.csv**: Exception reports listing detected discrepancies, timing violations, and compliance issues

These files represent the deterministic outputs of the reconciliation engine for the specific run.

### evidence_pack_<RUN_ID>.zip

The complete Evidence Pack is distributed as a ZIP archive named with the run ID. The ZIP contains all of the above components in a single, portable file.

## Integrity & Verifiability

Evidence Packs use cryptographic hashing to ensure data integrity and enable verification.

### Input Hashing

All input files (payroll data, recordkeeper data, mapping configuration) are hashed using SHA-256. These hashes are recorded in the manifest, allowing verification that:
- The correct input files were used for the run
- Input files have not been modified since the run

### Output Hashing

All output files (results, violations) are hashed using SHA-256. These hashes are recorded in the manifest, allowing verification that:
- Output files have not been altered after generation
- The outputs match what was originally produced

### ZIP Archive Hashing

The complete Evidence Pack ZIP archive is hashed using SHA-256. This hash is recorded in the manifest, allowing verification that:
- The entire Evidence Pack is intact and unmodified
- No files have been added, removed, or altered within the archive

### How Verification Works

To verify an Evidence Pack:
1. Extract the ZIP archive
2. Compute SHA-256 hashes of all files
3. Compare computed hashes against the hashes recorded in manifest.json
4. If all hashes match, the Evidence Pack is verified as unmodified

The manifest.json file ties all components together by providing a single source of truth for what files should exist and what their integrity hashes should be.

## What This Evidence Pack Does NOT Do

It is important to understand the limitations and non-assertions of an Evidence Pack:

### Not a Legal Opinion

An Evidence Pack does not provide legal opinions, legal advice, or legal interpretations. It contains factual data analysis results only.

### Not a Compliance Certification

An Evidence Pack does not certify compliance with any regulations, standards, or requirements. It documents what discrepancies were detected according to configured rules, but does not assert overall compliance status.

### Not a Fiduciary Judgment

An Evidence Pack does not make fiduciary judgments, fiduciary recommendations, or fiduciary determinations. It presents rule-based analysis results without making judgments about fiduciary obligations or responsibilities.

### Not a System of Record for Source Data

An Evidence Pack is a snapshot of analysis performed at a point in time. It is not a system of record for the underlying payroll or recordkeeper data. Source data systems remain the authoritative records.

### Not an Audit Endorsement

An Evidence Pack does not represent an audit endorsement, audit approval, or audit certification. It is documentation of analysis performed, not an audit opinion or audit conclusion.

## How Auditors and Internal Review Teams Use This

Evidence Packs are designed to serve as supporting documentation in audit and review processes.

### As Supporting Workpapers

Evidence Packs can be included in audit workpapers to document:
- What data was analyzed during a specific period
- What reconciliation rules were applied
- What discrepancies were identified
- The integrity and immutability of the analysis record

### As Objective Evidence of What Was Evaluated at a Point in Time

Evidence Packs provide objective, verifiable documentation of:
- The exact inputs used for analysis
- The exact outputs produced by the analysis
- The point in time when the analysis was performed
- The integrity of all components through cryptographic verification

This documentation can support audit trails, compliance reviews, and internal control assessments by providing an immutable record of reconciliation activities.

### Integration with Audit Processes

Auditors and review teams can:
- Verify the integrity of Evidence Pack contents using the manifest hashes
- Review the reconciliation results and violation reports
- Trace discrepancies back to source data using the recorded input hashes
- Confirm that analysis was performed according to documented rules and procedures

The cryptographic verification capabilities enable auditors to independently confirm that Evidence Pack contents have not been modified since creation.

