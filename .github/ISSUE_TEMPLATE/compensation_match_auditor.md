---
name: Compensation Match Auditor
description: Track work for the Compensation Definition + Match Impact Auditor
title: "Build Compensation Definition + Match Impact Auditor"
labels: []
assignees: []
---

## Objective

Implement the Compensation Definition + Match Impact Auditor described in:

- `docs/compensation_match_auditor_spec.md`
- `docs/codex_next_task.md`

## Scope

- Add deterministic analyzer module
- Add pytest coverage
- Add run summary output
- Add evidence pack integration
- Add Excel report sheet
- Add plan exception summary integration
- Add minimal Streamlit rule inputs

## Non-goals

- AI narrative
- plan document parser
- external integrations
- SaaS billing
- unrelated dashboard redesign

## Acceptance criteria

- [ ] `pytest tests/ -v` passes
- [ ] `summary["compensation_match"]` exists after a run
- [ ] `compensation_match_issues.csv` is generated when issues exist
- [ ] evidence pack includes the new CSV
- [ ] Excel report includes compensation/match issue sheet
- [ ] Streamlit can capture basic match/compensation rules
- [ ] no unrelated refactors are included
