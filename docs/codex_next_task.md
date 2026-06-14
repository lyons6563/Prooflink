# Codex Next Task: Compensation Definition + Match Impact Auditor

## Objective

Implement the Compensation Definition + Match Impact Auditor described in:

```text
docs/compensation_match_auditor_spec.md
```

This is the next strategic build for ProofLink. It should detect compensation-definition and match-impact errors, quantify likely correction exposure, and produce evidence-pack-compatible outputs.

## Build order

Do the work in this order:

1. Add a standalone analyzer module: `compensation_match_auditor.py`.
2. Add deterministic tests: `tests/test_compensation_match_auditor.py`.
3. Make the analyzer pass tests before wiring it into the engine.
4. Wire the feature into `main.py` run summary.
5. Add the CSV to the evidence index and evidence pack.
6. Add an Excel sheet for compensation/match issues.
7. Add plan exception summary integration.
8. Add minimal Streamlit rule inputs and issue summary display.
9. Update README only after the feature is working.

## Hard constraints

Do not add:

- AI narrative
- plan document parsing
- external payroll or recordkeeper integrations
- SaaS billing
- broad compliance dashboard work
- unrelated refactors

Do not break:

- deferral reconciliation
- loan reconciliation
- contribution timing
- Secure 2.0 checks
- eligibility drift
- 402(g) checks
- evidence pack generation
- run history

## Acceptance criteria

Run:

```bash
pytest tests/ -v
```

Feature is acceptable only if:

- tests pass
- existing behavior is preserved
- `summary["compensation_match"]` exists after a run
- `compensation_match_issues.csv` is generated when issues exist
- evidence pack includes the new CSV
- Excel report includes the new sheet
- Streamlit can collect basic match/compensation rules
- no AI/legal conclusion text is generated

## Recommended first commit

First commit should only add:

- `compensation_match_auditor.py`
- `tests/test_compensation_match_auditor.py`
- synthetic test fixtures if needed

Do not wire into UI or engine until standalone analyzer tests are green.
