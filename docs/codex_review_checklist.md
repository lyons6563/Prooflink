# Codex Review Checklist

Use this after Codex opens a PR for the Compensation Definition + Match Impact Auditor.

## Product fit

- [ ] Feature is focused on compensation definition and match impact.
- [ ] It does not drift into generic AI/compliance assistant functionality.
- [ ] Outputs answer: who is affected, how many dollars, likely cause, and what evidence exists.

## Technical correctness

- [ ] Analyzer works standalone before engine integration.
- [ ] Tests cover correct match, under-match, over-match, excluded class, true-up, and incomplete data.
- [ ] Existing tests still pass.
- [ ] New config defaults do not silently create false confidence.
- [ ] Missing columns produce warnings, not crashes.

## Evidence pack integration

- [ ] `compensation_match_issues.csv` is included when issues exist.
- [ ] Evidence index includes the new artifact.
- [ ] Excel report includes a dedicated sheet.
- [ ] Plan exception summary includes compensation/match issues.
- [ ] Run summary includes `summary["compensation_match"]`.

## UI discipline

- [ ] Streamlit additions are minimal.
- [ ] No dashboard redesign.
- [ ] Inputs are clear enough for a controlled demo.
- [ ] Download path for issue CSV or evidence pack works.

## Commercial discipline

- [ ] Report can support advisor/sponsor conversation.
- [ ] Language does not make legal conclusions.
- [ ] Correction hints are framed as review/action guidance, not legal advice.

## Security discipline

- [ ] No new hardcoded secrets.
- [ ] No unnecessary participant-level logging.
- [ ] No live-data claims in documentation.
