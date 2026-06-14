# Demo Script: Payroll Compensation Integrity Report

## Demo objective

Show that ProofLink can identify payroll compensation and match-impact errors, quantify exposure, and produce an evidence pack that a retirement plan advisor, sponsor, TPA, auditor, payroll team, or recordkeeper can act on.

## Demo setup

Use synthetic or scrubbed data only.

Demo should include:

- payroll file
- recordkeeper file
- multiple compensation columns
- at least one excluded compensation column
- at least one employee class field
- actual employer match column
- plan match config

## Storyline

### 1. Set the buyer context

Say:

> "Most plan reviews focus on investments, fees, and education. The operational risk is usually hiding in payroll: wrong compensation, wrong class, wrong match, or missing deposits. ProofLink scans for those issues and creates an evidence pack."

### 2. Upload files

Upload:

- payroll export
- recordkeeper export

Explain:

> "The product does not need a live integration for the first use case. A diagnostic scan can start with exported files."

### 3. Enter plan rules

Enter:

- match rate
- match cap
- eligible compensation columns
- excluded compensation columns
- employee class rules
- true-up setting

Explain:

> "The rules are explicit. The system is not guessing based on AI narrative. It calculates against the rules provided."

### 4. Run analysis

Run ProofLink.

Show:

- issue count
- estimated under-match dollars
- estimated over-match dollars
- likely root causes
- affected participants

### 5. Show evidence

Open the issue CSV or Excel sheet.

Point to:

- employee ID
- pay date
- eligible comp
- excluded comp
- employee deferral
- actual match
- expected match
- variance
- root cause
- correction hint

### 6. Download evidence pack

Show the ZIP.

Explain:

> "The point is not a pretty dashboard. The point is to give the advisor or sponsor evidence that survives review."

### 7. Close with ownership question

Ask:

> "If this showed up in your plan, who owns fixing it today: sponsor HR, payroll, recordkeeper, TPA, auditor, or advisor?"

That question exposes the gap ProofLink fills.

## What not to say

Do not claim:

- final legal determination
- EPCRS correction advice
- guaranteed compliance
- automated fiduciary protection
- replacement for TPA/auditor/legal review

Say instead:

> "This is an exception detection and evidence workflow. It helps identify likely problems and organize the data needed for review and correction."

## Best demo ending

End with:

> "This is not another dashboard. It is a control layer for plan operations. It finds payroll-driven retirement plan errors, quantifies the likely dollars, and tells everyone what evidence exists."
