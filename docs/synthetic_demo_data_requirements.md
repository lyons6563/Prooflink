# Synthetic Demo Data Requirements

Use synthetic data for demos until production security controls are implemented.

## Payroll file should include

Required:

- employee_id
- pay_date
- pretax deferral
- Roth deferral
- employer match
- regular compensation

Recommended:

- overtime compensation
- bonus compensation
- excluded compensation
- reimbursement/fringe compensation
- employee class
- age or date of birth
- HCE flag
- catch-up pretax
- catch-up Roth

## Recordkeeper file should include

Required:

- employee_id
- deposit_date
- pretax contribution
- Roth contribution

Recommended:

- employer match source
- loan repayment amount
- source code
- transaction type

## Demo scenarios to include

1. Correct match
   - no issue should be flagged

2. Under-match from missing compensation
   - overtime or bonus should count but was excluded from match calculation

3. Over-match from excluded compensation included
   - excluded pay was included in match calculation

4. Match paid to excluded class
   - employee class is ineligible, but match exists

5. Annual true-up timing difference
   - per-payroll match appears low but true-up is enabled, so issue should be lower severity or deferred to annual review

6. Missing recordkeeper deposit
   - payroll has contribution but RK does not

7. Secure 2.0 catch-up issue
   - HCE age 50+ catch-up coded pretax instead of Roth

## Demo principle

The synthetic data must show realistic operational mess, not perfect files.

The buyer should immediately understand:

- this can happen in real payroll files
- it affects dollars
- it creates work for someone
- ProofLink gives the evidence packet
