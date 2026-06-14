# Compensation Definition + Match Impact Auditor Spec

## Strategic intent

ProofLink should not become a generic compliance chatbot or dashboard. The highest-value wedge is a deterministic payroll-to-plan compensation control that finds compensation-definition errors, quantifies match impact, and creates evidence a sponsor, advisor, TPA, auditor, payroll team, or recordkeeper can act on.

This feature should answer four buyer questions:

1. Is payroll compensation being interpreted correctly under the plan's rules?
2. Did that compensation treatment cause under-match, over-match, missed eligibility, or contribution issues?
3. How much money is likely at risk?
4. What evidence do we need to fix it and prove the fix?

Working product name: **Payroll Compensation Integrity Report**.

## Primary buyer

Start with retirement plan advisors and consultants, not plan sponsors directly.

Advisor use case:

> "We run a payroll-to-recordkeeper integrity scan to identify contribution, match, eligibility, and compensation issues before they become audit or correction problems."

Sponsor use case:

> "We check whether payroll compensation is being applied correctly under your plan rules and quantify potential match/correction exposure before audit season."

## Target plans

Best first fit:

- 401(k) plans with employer match
- audited plans
- plans with hourly, part-time, seasonal, or multiple employee classes
- plans with bonuses, overtime, commissions, fringe, shift differential, or special pay codes
- plans that recently changed payroll vendors or recordkeepers
- plans with safe harbor match or annual true-up complexity

Lower-priority initial fit:

- governmental/nonprofit 457(b) and 401(a) plans without employer match impact
- plans with no employer contribution
- plans where source data does not include compensation or match columns

## Non-goals for this phase

Do **not** build these yet:

- AI plan document parser
- chatbot over plan documents
- full EPCRS correction engine
- payroll API integrations
- recordkeeper API integrations
- SaaS billing
- broad compliance library
- CRM/advisor workflow system
- legal conclusion generator

This phase must remain deterministic, auditable, and evidence-pack compatible.

## Product requirement

Add a module that detects compensation-definition and employer-match impact issues using payroll, recordkeeper, and plan rule configuration.

The output should be a buyer-facing evidence packet that shows:

- affected participant
- pay period / pay date
- compensation used
- eligible compensation expected under rules
- excluded compensation
- employee deferral amount
- actual employer match
- expected employer match
- variance
- likely root cause
- severity
- correction hint
- source evidence references

## Proposed file additions

Add:

- `compensation_match_auditor.py`
- `tests/test_compensation_match_auditor.py`
- optionally `examples/plan_match_config_example.json`

Modify only where required:

- `main.py`
- `streamlit_app.py`
- `plan_exception_summary.py`
- `issue_taxonomy.py`
- Excel report generation path, wherever currently implemented
- README only if documentation needs to be updated after the feature is working

## Plan match config schema

Introduce a config structure named `plan_match_config`.

Minimum fields:

```json
{
  "match_formula_name": "50% up to 6%",
  "match_type": "percent_of_comp",
  "match_rate": 0.50,
  "match_cap_pct": 0.06,
  "match_frequency": "per_payroll",
  "true_up_enabled": false,
  "eligible_comp_columns": ["Regular Compensation", "Overtime", "Bonus"],
  "excluded_comp_columns": ["Fringe", "Reimbursement"],
  "employee_class_column": "Employee Class",
  "eligible_classes": ["Full-Time", "Part-Time"],
  "excluded_classes": ["Intern", "Union Excluded"],
  "absolute_tolerance": 5.00,
  "relative_tolerance_pct": 0.15
}
```

Supported `match_frequency` values:

- `per_payroll`
- `annual`

Supported `match_type` values for this phase:

- `percent_of_comp`

Future match types can be added later, but do not widen scope now.

## Engine behavior

For each payroll row or grouped participant period:

1. Normalize employee ID and pay date.
2. Calculate `eligible_comp` as the sum of configured eligible compensation columns.
3. Calculate `excluded_comp` as the sum of configured excluded compensation columns.
4. Calculate total employee deferrals:
   - pretax deferral
   - Roth deferral
   - optionally catch-up sources if present and relevant
5. Calculate `deferral_pct = employee_deferrals / eligible_comp`.
6. Calculate `eligible_pct_for_match = min(deferral_pct, match_cap_pct)`.
7. Calculate `expected_match = eligible_comp * eligible_pct_for_match * match_rate`.
8. Compare to actual employer match.
9. Apply tolerance rules.
10. Assign issue type, severity, correction hint, and likely root cause.

## Likely root-cause categories

Use deterministic rules to classify likely root cause.

Required root causes:

- `missing_eligible_compensation`
- `excluded_compensation_included`
- `employee_class_exclusion_issue`
- `match_formula_variance`
- `annual_true_up_timing_difference`
- `recordkeeper_or_payroll_missing_match`
- `source_data_incomplete`
- `unknown_requires_review`

Example logic:

- If eligible comp is zero but deferral or match exists: `source_data_incomplete` or `missing_eligible_compensation`.
- If excluded comp is positive and actual match is higher than expected: `excluded_compensation_included`.
- If employee class is excluded but match exists: `employee_class_exclusion_issue`.
- If under-match variance is present and eligible comp appears lower than gross/total comp: `missing_eligible_compensation`.
- If `match_frequency = annual` or `true_up_enabled = true`, avoid over-flagging per-payroll under-match; classify as `annual_true_up_timing_difference` unless annualized variance remains after aggregation.

## Output CSV

Create:

- `compensation_match_issues.csv`

Required columns:

- `run_id`
- `plan_name`
- `plan_year`
- `employee_id`
- `pay_date`
- `employee_class`
- `eligible_comp`
- `excluded_comp`
- `employee_deferrals`
- `deferral_pct`
- `actual_match`
- `expected_match`
- `match_variance`
- `match_variance_abs`
- `match_variance_pct`
- `issue_type`
- `likely_root_cause`
- `issue_category`
- `severity`
- `correction_hint`

Optional but useful:

- `gross_comp`
- `payroll_file_row_number`
- `recordkeeper_file_row_number`
- `source_comp_columns_used`
- `source_match_column_used`

## Issue taxonomy additions

Add issue types to `issue_taxonomy.py`.

Required issue types:

- `Under-match from compensation definition variance`
- `Over-match from excluded compensation included`
- `Match paid to excluded employee class`
- `Potential annual true-up timing difference`
- `Compensation source data incomplete`

Suggested categories:

- `Compensation/Match`

Suggested severity:

- under-match: High
- over-match: Medium
- excluded class match: High or Medium depending amount
- true-up timing difference: Low or Medium
- incomplete source data: Medium

## Summary metrics

Add a `compensation_match` object to run summary.

Example:

```json
{
  "total_rows_evaluated": 400,
  "participants_evaluated": 145,
  "issue_count": 18,
  "under_match_count": 11,
  "over_match_count": 5,
  "excluded_class_match_count": 2,
  "estimated_under_match_dollars": 4125.18,
  "estimated_over_match_dollars": 783.40,
  "csv_path": ".../compensation_match_issues.csv"
}
```

Also add the CSV to:

- `summary["evidence_index"]`
- evidence pack ZIP
- Excel report
- plan exception summary aggregation

## Streamlit UI requirements

Do not overbuild UI. Add only what is needed to run the feature.

Minimum UI additions:

- checkbox or expander: `Compensation / Match Rules`
- inputs for:
  - match rate
  - match cap percentage
  - match frequency
  - true-up enabled
  - eligible compensation columns
  - excluded compensation columns
  - employee class column
  - eligible/excluded class values
- summary display:
  - issue count
  - estimated under-match dollars
  - estimated over-match dollars
  - top likely root causes
- download button for `compensation_match_issues.csv`

Do not add AI narrative in this phase.

## Testing requirements

Add deterministic pytest tests using synthetic DataFrames.

Required tests:

1. `test_correct_match_no_issue`
   - eligible comp and match are correct
   - no issue CSV or empty issue CSV

2. `test_under_match_from_missing_compensation`
   - eligible compensation should include overtime but actual match appears based only on regular comp
   - issue type should be under-match
   - root cause should be `missing_eligible_compensation`

3. `test_over_match_from_excluded_comp_included`
   - excluded comp is present and actual match is too high
   - root cause should be `excluded_compensation_included`

4. `test_match_paid_to_excluded_class`
   - employee class is excluded but actual match exists
   - issue type should be excluded class match

5. `test_true_up_does_not_overflag_per_payroll_variance`
   - true-up enabled or annual match frequency
   - do not classify ordinary per-payroll variance as high-severity under-match unless annual aggregation still fails

6. `test_source_data_incomplete_warning`
   - missing actual match or eligible comp columns
   - return warning and do not crash engine

## Acceptance criteria

Feature is done when:

- New analyzer works standalone on synthetic DataFrames.
- Existing match analyzer is either extended cleanly or the new analyzer supersedes it without breaking current behavior.
- Engine returns `summary["compensation_match"]`.
- Evidence pack includes `compensation_match_issues.csv` when issues exist.
- Excel report includes a dedicated compensation/match issue sheet.
- Plan exception summary includes compensation/match issues.
- Tests pass with `pytest tests/ -v`.
- Existing evidence pack and run history behavior is not broken.
- No AI-generated legal conclusions are included.

## Codex build prompt

Use this prompt with Codex:

```text
Implement the Compensation Definition + Match Impact Auditor described in docs/compensation_match_auditor_spec.md.

Keep the implementation deterministic and evidence-pack compatible. Do not add AI narrative, plan document parsing, external integrations, SaaS billing, or unrelated dashboard work.

Start by adding a standalone analyzer module with tests. Then wire it into main.py summary output, evidence_index, evidence pack, Excel report, and Streamlit rule inputs only after the analyzer tests pass.

Preserve existing reconciliation, timing, Secure 2.0, eligibility, 402(g), and run history behavior.

Add tests for correct match, under-match from missing compensation, over-match from excluded compensation included, excluded class match, annual true-up handling, and incomplete source data.
```

## Commercial demo script after build

Once implemented, the demo should be:

1. Upload payroll and recordkeeper files.
2. Enter simple match/compensation rules.
3. Run ProofLink.
4. Show the Payroll Compensation Integrity Report.
5. Point to estimated correction exposure.
6. Download the evidence pack.
7. Ask the buyer: "Who owns fixing this today: payroll, recordkeeper, TPA, or advisor?"

That last question is the sale. If no one owns it, ProofLink becomes the control layer.
