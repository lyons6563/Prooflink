from pathlib import Path

import pandas as pd

from population_validation_analyzer import (
    EMPLOYMENT_STATUS_CONFLICT,
    POST_TERMINATION_COMPENSATION,
    analyze_population_validation,
)


def _read_issues(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def test_active_payroll_terminated_recordkeeper_status_conflict(tmp_path: Path):
    payroll = pd.DataFrame(
        [{"employee_id": "1001", "employment_status": "active", "payroll_file_row_number": 2}]
    )
    rk = pd.DataFrame(
        [{"employee_id": "1001", "employment_status": "terminated", "recordkeeper_file_row_number": 9}]
    )

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path, run_id="run-1")

    assert summary["issue_count"] == 1
    assert summary["employment_status_conflict_count"] == 1
    issues = _read_issues(csv_path)
    row = issues.iloc[0]
    assert row["exception_type"] == EMPLOYMENT_STATUS_CONFLICT
    assert row["issue_category"] == "Population Validation"
    assert row["severity"] == "High"
    assert row["authoritative_source"] == "undetermined"
    assert row["suspected_origin"] == "undetermined"
    assert row["resolution_owner"] == "plan_sponsor_operations"
    assert row["investigation_status"] == "needs_review"
    assert row["resolution_status"] == "open"
    assert row["payroll_file_row_number"] == 2
    assert row["recordkeeper_file_row_number"] == 9


def test_both_statuses_active_no_exception(tmp_path: Path):
    payroll = pd.DataFrame([{"employee_id": "1001", "employment_status": "employed"}])
    rk = pd.DataFrame([{"employee_id": "1001", "employment_status": "active"}])

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["issue_count"] == 0
    assert summary["employment_status_conflict_count"] == 0
    assert csv_path is None


def test_unknown_or_missing_statuses_do_not_false_positive(tmp_path: Path):
    payroll = pd.DataFrame([{"employee_id": "1001", "employment_status": "leave"}])
    rk = pd.DataFrame([{"employee_id": "1001"}])

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["issue_count"] == 0
    assert csv_path is None
    assert EMPLOYMENT_STATUS_CONFLICT in summary["skipped_rules"]
    assert "warning" in summary


def test_positive_compensation_more_than_30_days_after_termination(tmp_path: Path):
    payroll = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-03-05",
                "Compensation": 1200.50,
                "payroll_file_row_number": 4,
            }
        ]
    )
    rk = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "termination_date": "2025-01-31",
                "recordkeeper_file_row_number": 12,
            }
        ]
    )

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["issue_count"] == 1
    assert summary["post_termination_compensation_count"] == 1
    issues = _read_issues(csv_path)
    row = issues.iloc[0]
    assert row["exception_type"] == POST_TERMINATION_COMPENSATION
    assert row["recordkeeper_termination_date"] == "2025-01-31"
    assert row["later_payroll_date"] == "2025-03-05"
    assert row["post_termination_compensation_amount"] == 1200.50
    assert row["days_after_termination"] == 33
    assert row["reference_source"] == "recordkeeper_termination_date"


def test_compensation_within_30_days_after_termination_no_exception(tmp_path: Path):
    payroll = pd.DataFrame([{"employee_id": "1001", "pay_date": "2025-02-20", "Compensation": 900.0}])
    rk = pd.DataFrame([{"employee_id": "1001", "termination_date": "2025-01-31"}])

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["issue_count"] == 0
    assert csv_path is None


def test_compensation_before_termination_no_exception(tmp_path: Path):
    payroll = pd.DataFrame([{"employee_id": "1001", "pay_date": "2025-01-15", "Compensation": 900.0}])
    rk = pd.DataFrame([{"employee_id": "1001", "termination_date": "2025-01-31"}])

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["issue_count"] == 0
    assert csv_path is None


def test_invalid_or_missing_dates_no_failure_no_unsupported_exception(tmp_path: Path):
    payroll = pd.DataFrame([{"employee_id": "1001", "pay_date": "not-a-date", "Compensation": 900.0}])
    rk = pd.DataFrame([{"employee_id": "1001", "termination_date": None}])

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["issue_count"] == 0
    assert summary["post_termination_compensation_count"] == 0
    assert csv_path is None


def test_one_sided_population_rows_are_not_population_validation_exceptions(tmp_path: Path):
    payroll = pd.DataFrame(
        [
            {"employee_id": "1001", "employment_status": "active", "pay_date": "2025-01-15", "Compensation": 1000.0},
            {"employee_id": "1002", "employment_status": "active", "pay_date": "2025-01-15", "Compensation": 1000.0},
        ]
    )
    rk = pd.DataFrame(
        [
            {"employee_id": "1001", "employment_status": "active", "termination_date": ""},
            {"employee_id": "1003", "employment_status": "terminated", "termination_date": "2025-01-01"},
        ]
    )

    summary, csv_path = analyze_population_validation(payroll, rk, tmp_path)

    assert summary["payroll_employee_count"] == 2
    assert summary["recordkeeper_employee_count"] == 2
    assert summary["common_employee_count"] == 1
    assert summary["issue_count"] == 0
    assert csv_path is None
