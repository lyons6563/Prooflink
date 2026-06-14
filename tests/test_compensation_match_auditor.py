from pathlib import Path

import pandas as pd

from compensation_match_auditor import analyze_compensation_match


BASE_CONFIG = {
    "match_formula_name": "50% up to 6%",
    "match_type": "percent_of_comp",
    "match_rate": 0.50,
    "match_cap_pct": 0.06,
    "match_frequency": "per_payroll",
    "true_up_enabled": False,
    "eligible_comp_columns": ["Regular Compensation", "Overtime", "Bonus"],
    "excluded_comp_columns": ["Fringe", "Reimbursement"],
    "employee_class_column": "Employee Class",
    "eligible_classes": ["Full-Time", "Part-Time"],
    "excluded_classes": ["Intern", "Union Excluded"],
    "absolute_tolerance": 5.00,
    "relative_tolerance_pct": 0.15,
}


def _base_row(**overrides):
    row = {
        "employee_id": "1001",
        "pay_date": "2025-01-15",
        "Employee Class": "Full-Time",
        "Regular Compensation": 1000.0,
        "Overtime": 0.0,
        "Bonus": 0.0,
        "Fringe": 0.0,
        "Reimbursement": 0.0,
        "Gross Compensation": 1000.0,
        "EE Deferral $": 60.0,
        "EE Roth $": 0.0,
        "ER Match $": 30.0,
    }
    row.update(overrides)
    return row


def test_correct_match_no_issue(tmp_path: Path):
    df = pd.DataFrame([_base_row()])

    summary, csv_path = analyze_compensation_match(
        payroll_df=df,
        output_dir=tmp_path,
        plan_match_config=BASE_CONFIG,
        run_id="run-1",
        plan_name="Demo Plan",
        plan_year=2025,
    )

    assert summary["total_rows_evaluated"] == 1
    assert summary["issue_count"] == 0
    assert summary["csv_path"] is None
    assert csv_path is None
    assert not (tmp_path / "compensation_match_issues.csv").exists()


def test_under_match_from_missing_compensation(tmp_path: Path):
    df = pd.DataFrame(
        [
            _base_row(
                Overtime=500.0,
                **{
                    "Gross Compensation": 1500.0,
                    "EE Deferral $": 90.0,
                    "ER Match $": 30.0,
                },
            )
        ]
    )

    summary, csv_path = analyze_compensation_match(df, tmp_path, BASE_CONFIG)

    assert summary["issue_count"] == 1
    assert summary["under_match_count"] == 1
    issues = pd.read_csv(csv_path)
    assert issues.loc[0, "issue_type"] == "Under-match from compensation definition variance"
    assert issues.loc[0, "likely_root_cause"] == "missing_eligible_compensation"
    assert issues.loc[0, "expected_match"] == 45.0
    assert issues.loc[0, "actual_match"] == 30.0


def test_over_match_from_excluded_comp_included(tmp_path: Path):
    df = pd.DataFrame(
        [
            _base_row(
                Fringe=500.0,
                **{
                    "Gross Compensation": 1500.0,
                    "EE Deferral $": 90.0,
                    "ER Match $": 45.0,
                },
            )
        ]
    )

    summary, csv_path = analyze_compensation_match(df, tmp_path, BASE_CONFIG)

    assert summary["issue_count"] == 1
    assert summary["over_match_count"] == 1
    issues = pd.read_csv(csv_path)
    assert issues.loc[0, "issue_type"] == "Over-match from excluded compensation included"
    assert issues.loc[0, "likely_root_cause"] == "excluded_compensation_included"
    assert issues.loc[0, "excluded_comp"] == 500.0


def test_match_paid_to_excluded_class(tmp_path: Path):
    df = pd.DataFrame(
        [
            _base_row(
                **{
                    "Employee Class": "Intern",
                    "ER Match $": 30.0,
                }
            )
        ]
    )

    summary, csv_path = analyze_compensation_match(df, tmp_path, BASE_CONFIG)

    assert summary["issue_count"] == 1
    assert summary["excluded_class_match_count"] == 1
    issues = pd.read_csv(csv_path)
    assert issues.loc[0, "issue_type"] == "Match paid to excluded employee class"
    assert issues.loc[0, "likely_root_cause"] == "employee_class_exclusion_issue"


def test_true_up_does_not_overflag_per_payroll_variance(tmp_path: Path):
    config = BASE_CONFIG | {"true_up_enabled": True}
    df = pd.DataFrame([_base_row(**{"ER Match $": 0.0})])

    summary, csv_path = analyze_compensation_match(df, tmp_path, config)

    assert summary["issue_count"] == 1
    assert summary["under_match_count"] == 0
    issues = pd.read_csv(csv_path)
    assert issues.loc[0, "issue_type"] == "Potential annual true-up timing difference"
    assert issues.loc[0, "likely_root_cause"] == "annual_true_up_timing_difference"
    assert issues.loc[0, "severity"] in {"Low", "Medium"}


def test_deferral_amount_prevents_component_double_counting(tmp_path: Path):
    df = pd.DataFrame(
        [
            _base_row(
                **{
                    "EE Deferral $": 40.0,
                    "EE Roth $": 0.0,
                    "deferral_amount": 40.0,
                    "ER Match $": 20.0,
                }
            )
        ]
    )

    summary, csv_path = analyze_compensation_match(df, tmp_path, BASE_CONFIG)

    assert summary["issue_count"] == 0
    assert summary["csv_path"] is None
    assert csv_path is None


def test_true_up_timing_difference_excluded_from_under_match_estimate(tmp_path: Path):
    config = BASE_CONFIG | {"match_frequency": "annual"}
    df = pd.DataFrame([_base_row(**{"ER Match $": 0.0})])

    summary, csv_path = analyze_compensation_match(df, tmp_path, config)

    assert summary["issue_count"] == 1
    assert summary["estimated_under_match_dollars"] == 0.0
    issues = pd.read_csv(csv_path)
    assert issues.loc[0, "issue_type"] == "Potential annual true-up timing difference"


def test_source_data_incomplete_warning(tmp_path: Path):
    incomplete_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-01-15",
                "Regular Compensation": 1000.0,
                "EE Deferral $": 60.0,
            }
        ]
    )

    summary, csv_path = analyze_compensation_match(
        incomplete_df,
        tmp_path,
        BASE_CONFIG,
    )

    assert summary["issue_count"] == 0
    assert summary["csv_path"] is None
    assert csv_path is None
    assert "warning" in summary
    assert "missing" in summary["warning"].lower()
