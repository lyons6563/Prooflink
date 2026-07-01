# ProofLink Polished 100-Employee Demo Guide

## 1. What ProofLink Does

ProofLink compares payroll records against recordkeeper activity, highlights operational exceptions, and packages the results into reviewable evidence files. The polished demo is designed for a five-minute walkthrough with a small, deterministic, synthetic plan population.

## 2. Files Used In The Demo

- `data/demo/polished/prooflink_polished_100_payroll.csv`
- `data/demo/polished/prooflink_polished_100_recordkeeper.csv`
- `data/demo/polished/prooflink_polished_100_ground_truth.csv`
- `data/demo/polished/prooflink_polished_100_metadata.json`
- `data/demo/polished/prooflink_polished_100_mapping.yaml`

## 3. How To Generate The Dataset

```powershell
py scripts\generate_realistic_demo.py --profile polished --employees 100 --periods 8 --year 2025 --seed 20250630 --output-dir data\demo\polished --prefix prooflink_polished_100
```

The generator is deterministic. Running the same command with the same seed produces the same files.

## 4. How To Run ProofLink

Use the public engine entrypoint with the polished input files and persistent local output folders:

```powershell
py -c "from main import EngineConfig, run_prooflink_engine; run_prooflink_engine(payroll_path='data/demo/polished/prooflink_polished_100_payroll.csv', rk_path='data/demo/polished/prooflink_polished_100_recordkeeper.csv', config=EngineConfig(plan_name='ProofLink Polished Demo Plan', payroll_vendor_hint='ADP', rk_vendor_hint='VENDOR_RK_1', output_dir='data/demo/polished_output', proofs_dir='data/demo/polished_proofs'), run_id='prooflink-polished-100-demo')"
```

## 5. The 12 Planted Issues

| Scenario | Count | What It Shows |
|---|---:|---|
| `DEFERRAL_MISMATCH` | 2 | Payroll and recordkeeper contribution amounts differ. |
| `LOAN_MISMATCH` | 1 | Payroll loan repayment is not fully reflected at the recordkeeper. |
| `ONLY_IN_PAYROLL` | 1 | Payroll contains a participant missing from the recordkeeper file. |
| `ONLY_IN_RECORDKEEPER` | 1 | Recordkeeper activity has no matching payroll participant. |
| `LATE_CONTRIBUTION` | 2 | Deposits posted after the five-business-day demo threshold. |
| `EMPLOYMENT_STATUS_CONFLICT` | 1 | Payroll active status conflicts with recordkeeper terminated status. |
| `POST_TERMINATION_COMPENSATION` | 1 | Positive payroll compensation appears after the RK termination date. |
| `SECURE20_HCE_PRETAX_CATCHUP` | 1 | HCE catch-up dollars are coded pretax instead of Roth. |
| `COMP_MATCH_UNDER` | 1 | Employer match is below expected formula output. |
| `COMP_MATCH_OVER` | 1 | Employer match is above expected formula output. |

## 6. Recommended Five-Minute Walkthrough

1. Show the payroll and recordkeeper source files.
2. Explain that ProofLink compares payroll, RK, timing, status, and plan-rule signals.
3. Open `plan_exception_summary.csv` or the Excel report summary.
4. Highlight one contribution amount mismatch.
5. Highlight one late deposit.
6. Highlight one employment-status conflict.
7. Highlight one post-termination compensation item.
8. Show `prooflink_evidence_pack.zip`.
9. Run the verifier and show `PASS`.

## 7. What Each Output File Demonstrates

- `deferral_mismatches.csv`: contribution amount variances.
- `loan_mismatches.csv`: loan repayment variances.
- `only_in_payroll_deferrals.csv`: payroll-only population records.
- `only_in_recordkeeper_deferrals.csv`: RK-only population records.
- `late_contributions.csv`: period-aligned timing findings, including missing deposits and RK-only rows.
- `population_validation_issues.csv`: status conflict and post-termination compensation review flags.
- `secure20_violations.csv`: Secure 2.0 catch-up source review.
- `compensation_match_issues.csv`: employer-match review issues.
- `reconciliation_report.xlsx`: client-facing Excel workbook with the main outputs.
- `prooflink_evidence_pack.zip`: packaged evidence for offline verification.

## 8. How To Verify The Evidence Pack

```powershell
py verify_proof.py data\demo\polished_output\prooflink_evidence_pack.zip
```

Expected result:

- Input verification: `PASS`
- Output verification: `PASS`
- Overall verification: `PASS`

## 9. Questions To Ask A Reviewer

- Are these exception categories clear enough for plan operations staff?
- Which output would your team review first?
- Does the evidence pack contain the right level of detail for audit support?
- Are the correction hints operationally useful?
- What additional plan provisions would you want configured before production use?

## 10. Synthetic Data Disclosure

All data in this demo is synthetic. It does not contain real names, SSNs, addresses, emails, employer information, payroll history, or recordkeeper data.

