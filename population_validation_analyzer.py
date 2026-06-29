from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from issue_taxonomy import get_issue_metadata


EMPLOYMENT_STATUS_CONFLICT = "EMPLOYMENT_STATUS_CONFLICT"
POST_TERMINATION_COMPENSATION = "POST_TERMINATION_COMPENSATION"
ISSUE_CATEGORY = "Population Validation"

ACTIVE_STATUSES = {"active", "employed", "current", "a"}
TERMINATED_STATUSES = {"terminated", "inactive", "separated", "term", "t"}

PAYROLL_STATUS_CANDIDATES = ["payroll_employment_status", "employment_status", "employee_status", "Employee Status", "Employment Status", "status", "Status"]
RK_STATUS_CANDIDATES = ["recordkeeper_employment_status", "rk_employment_status", "participant_status", "employment_status", "employee_status", "Employee Status", "Employment Status", "status", "Status"]
TERMINATION_DATE_CANDIDATES = ["termination_date", "Termination Date", "term_date", "Term Date", "date_of_termination", "separation_date"]
PAYROLL_DATE_CANDIDATES = ["pay_date", "payroll_date", "Payroll Date", "check_date", "Check Date"]
COMPENSATION_CANDIDATES = ["compensation", "Compensation", "gross_compensation", "Gross Compensation", "gross_comp", "Gross Comp", "gross_wages", "Gross Wages", "Gross_Wages", "total_compensation", "Total Compensation", "eligible_compensation", "Eligibility Compensation", "Plan Compensation", "Regular Compensation"]

OUTPUT_COLUMNS = [
    "run_id", "plan_name", "plan_year", "employee_id", "exception_type", "issue_type",
    "issue_category", "severity", "compared_sources", "mismatch_location", "reference_source",
    "authoritative_source", "suspected_origin", "resolution_owner", "investigation_status",
    "first_detected_timestamp", "correction_submitted_timestamp", "resolution_verification",
    "business_outcome_confirmation", "resolution_status", "correction_hint",
    "payroll_employment_status", "recordkeeper_employment_status", "recordkeeper_termination_date",
    "later_payroll_date", "post_termination_compensation_amount", "days_after_termination",
    "payroll_file_row_number", "recordkeeper_file_row_number", "source_payroll_status_column",
    "source_recordkeeper_status_column", "source_termination_date_column", "source_payroll_date_column",
    "source_compensation_column", "details",
]


def _base_summary(warning: Optional[str] = None) -> Dict[str, Any]:
    summary = {
        "payroll_employee_count": 0,
        "recordkeeper_employee_count": 0,
        "common_employee_count": 0,
        "issue_count": 0,
        "employment_status_conflict_count": 0,
        "post_termination_compensation_count": 0,
        "skipped_rules": {},
        "csv_path": None,
    }
    if warning:
        summary["warning"] = warning
    return summary


def _employee_id_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or "employee_id" not in df.columns:
        return None
    ids = df["employee_id"].astype(str).str.strip()
    return ids.mask(ids.str.lower().isin({"", "nan", "none", "nat"}))


def _first_present_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_to_actual = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        actual = lower_to_actual.get(str(candidate).lower())
        if actual is not None:
            return actual
    return None


def _normalized_status(value: Any, allowed: set[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"nan", "none", "nat"}:
        return None
    return normalized if normalized in allowed else None


def _first_row_lookup(df: pd.DataFrame, ids: pd.Series) -> Dict[str, pd.Series]:
    work = df.copy()
    work["_population_employee_id"] = ids
    work = work.dropna(subset=["_population_employee_id"])
    return {str(row["_population_employee_id"]): row for _, row in work.drop_duplicates("_population_employee_id", keep="first").iterrows()}


def _rows_by_employee(df: pd.DataFrame, ids: pd.Series) -> Dict[str, pd.DataFrame]:
    work = df.copy()
    work["_population_employee_id"] = ids
    work = work.dropna(subset=["_population_employee_id"])
    return {str(employee_id): group.copy() for employee_id, group in work.groupby("_population_employee_id")}


def _row_value(row: Optional[pd.Series], column: Optional[str]) -> Optional[Any]:
    if row is None or not column or column not in row.index:
        return None
    value = row.get(column)
    if pd.isna(value):
        return None
    return value


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _common_exception_fields(employee_id: str, exception_type: str, mismatch_location: str, reference_source: str, timestamp: str, run_id: Optional[str], plan_name: Optional[str], plan_year: Optional[int]) -> Dict[str, Any]:
    metadata = get_issue_metadata(exception_type)
    return {
        "run_id": run_id,
        "plan_name": plan_name,
        "plan_year": plan_year,
        "employee_id": employee_id,
        "exception_type": exception_type,
        "issue_type": exception_type,
        "issue_category": metadata.get("issue_category", ISSUE_CATEGORY),
        "severity": metadata.get("severity", "High"),
        "compared_sources": "payroll,recordkeeper",
        "mismatch_location": mismatch_location,
        "reference_source": reference_source,
        "authoritative_source": "undetermined",
        "suspected_origin": "undetermined",
        "resolution_owner": "plan_sponsor_operations",
        "investigation_status": "needs_review",
        "first_detected_timestamp": timestamp,
        "correction_submitted_timestamp": None,
        "resolution_verification": "pending",
        "business_outcome_confirmation": "pending",
        "resolution_status": "open",
        "correction_hint": metadata.get("correction_hint", "Review this item with plan operations."),
    }


def _summary(payroll_count: int, rk_count: int, common_count: int, issues_df: Optional[pd.DataFrame], skipped_rules: Dict[str, str], csv_path: Optional[Path]) -> Dict[str, Any]:
    if issues_df is None or issues_df.empty:
        issue_count = status_count = post_term_count = 0
    else:
        issue_count = int(len(issues_df))
        status_count = int((issues_df["exception_type"] == EMPLOYMENT_STATUS_CONFLICT).sum())
        post_term_count = int((issues_df["exception_type"] == POST_TERMINATION_COMPENSATION).sum())
    result = {
        "payroll_employee_count": int(payroll_count),
        "recordkeeper_employee_count": int(rk_count),
        "common_employee_count": int(common_count),
        "issue_count": issue_count,
        "employment_status_conflict_count": status_count,
        "post_termination_compensation_count": post_term_count,
        "skipped_rules": skipped_rules,
        "csv_path": str(csv_path) if csv_path else None,
    }
    if skipped_rules:
        result["warning"] = "Population validation skipped one or more optional-field rules."
    return result


def analyze_population_validation(payroll_df: pd.DataFrame, recordkeeper_df: pd.DataFrame, output_dir: Path, *, run_id: Optional[str] = None, plan_name: Optional[str] = None, plan_year: Optional[int] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Run population-validation controls for employees present in both sources."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if payroll_df is None or payroll_df.empty:
            return _base_summary("Population validation skipped because payroll data is empty."), None
        if recordkeeper_df is None or recordkeeper_df.empty:
            return _base_summary("Population validation skipped because recordkeeper data is empty."), None

        payroll_ids = _employee_id_series(payroll_df)
        rk_ids = _employee_id_series(recordkeeper_df)
        if payroll_ids is None or rk_ids is None:
            missing = []
            if payroll_ids is None:
                missing.append("payroll.employee_id")
            if rk_ids is None:
                missing.append("recordkeeper.employee_id")
            return _base_summary("Population validation skipped because source data is missing: " + ", ".join(missing)), None

        payroll_set = set(payroll_ids.dropna().astype(str))
        rk_set = set(rk_ids.dropna().astype(str))
        common_ids = sorted(payroll_set & rk_set)
        payroll_first = _first_row_lookup(payroll_df, payroll_ids)
        rk_first = _first_row_lookup(recordkeeper_df, rk_ids)
        payroll_rows = _rows_by_employee(payroll_df, payroll_ids)

        payroll_status_col = _first_present_column(payroll_df, PAYROLL_STATUS_CANDIDATES)
        rk_status_col = _first_present_column(recordkeeper_df, RK_STATUS_CANDIDATES)
        term_date_col = _first_present_column(recordkeeper_df, TERMINATION_DATE_CANDIDATES)
        payroll_date_col = _first_present_column(payroll_df, PAYROLL_DATE_CANDIDATES)
        comp_col = _first_present_column(payroll_df, COMPENSATION_CANDIDATES)

        skipped_rules: Dict[str, str] = {}
        issues: List[Dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not payroll_status_col or not rk_status_col:
            skipped_rules[EMPLOYMENT_STATUS_CONFLICT] = "Missing payroll or recordkeeper employment status column."
        else:
            for employee_id in common_ids:
                payroll_row = payroll_first.get(employee_id)
                rk_row = rk_first.get(employee_id)
                payroll_status_raw = _row_value(payroll_row, payroll_status_col)
                rk_status_raw = _row_value(rk_row, rk_status_col)
                if not _normalized_status(payroll_status_raw, ACTIVE_STATUSES) or not _normalized_status(rk_status_raw, TERMINATED_STATUSES):
                    continue
                row = _common_exception_fields(employee_id, EMPLOYMENT_STATUS_CONFLICT, "employment_status", "recordkeeper_status", timestamp, run_id, plan_name, plan_year)
                row.update({
                    "payroll_employment_status": payroll_status_raw,
                    "recordkeeper_employment_status": rk_status_raw,
                    "recordkeeper_termination_date": None,
                    "later_payroll_date": None,
                    "post_termination_compensation_amount": None,
                    "days_after_termination": None,
                    "payroll_file_row_number": _row_value(payroll_row, "payroll_file_row_number"),
                    "recordkeeper_file_row_number": _row_value(rk_row, "recordkeeper_file_row_number"),
                    "source_payroll_status_column": payroll_status_col,
                    "source_recordkeeper_status_column": rk_status_col,
                    "source_termination_date_column": term_date_col,
                    "source_payroll_date_column": payroll_date_col,
                    "source_compensation_column": comp_col,
                    "details": f"Payroll status is recognized as active ({payroll_status_raw}) while recordkeeper status is recognized as terminated ({rk_status_raw}).",
                })
                issues.append(row)

        if not term_date_col or not payroll_date_col or not comp_col:
            missing = []
            if not term_date_col:
                missing.append("recordkeeper termination date")
            if not payroll_date_col:
                missing.append("payroll date")
            if not comp_col:
                missing.append("payroll compensation")
            skipped_rules[POST_TERMINATION_COMPENSATION] = "Missing optional field(s): " + ", ".join(missing) + "."
        else:
            for employee_id in common_ids:
                rk_row = rk_first.get(employee_id)
                term_date = pd.to_datetime(_row_value(rk_row, term_date_col), errors="coerce")
                if pd.isna(term_date):
                    continue
                employee_payroll_rows = payroll_rows.get(employee_id)
                if employee_payroll_rows is None or employee_payroll_rows.empty:
                    continue
                dates = pd.to_datetime(employee_payroll_rows[payroll_date_col], errors="coerce")
                compensation = _numeric_series(employee_payroll_rows, comp_col)
                for idx, payroll_row in employee_payroll_rows.iterrows():
                    payroll_date = dates.loc[idx]
                    if pd.isna(payroll_date):
                        continue
                    days_after = int((payroll_date.normalize() - term_date.normalize()).days)
                    comp_amount = float(compensation.loc[idx])
                    if days_after <= 30 or comp_amount <= 0:
                        continue
                    row = _common_exception_fields(employee_id, POST_TERMINATION_COMPENSATION, "post_termination_compensation", "recordkeeper_termination_date", timestamp, run_id, plan_name, plan_year)
                    row.update({
                        "payroll_employment_status": _row_value(payroll_row, payroll_status_col),
                        "recordkeeper_employment_status": _row_value(rk_row, rk_status_col),
                        "recordkeeper_termination_date": term_date.date().isoformat(),
                        "later_payroll_date": payroll_date.date().isoformat(),
                        "post_termination_compensation_amount": round(comp_amount, 2),
                        "days_after_termination": days_after,
                        "payroll_file_row_number": _row_value(payroll_row, "payroll_file_row_number"),
                        "recordkeeper_file_row_number": _row_value(rk_row, "recordkeeper_file_row_number"),
                        "source_payroll_status_column": payroll_status_col,
                        "source_recordkeeper_status_column": rk_status_col,
                        "source_termination_date_column": term_date_col,
                        "source_payroll_date_column": payroll_date_col,
                        "source_compensation_column": comp_col,
                        "details": f"Payroll compensation amount {comp_amount:.2f} appears {days_after} days after recordkeeper termination date {term_date.date().isoformat()}.",
                    })
                    issues.append(row)

        if not issues:
            return _summary(len(payroll_set), len(rk_set), len(common_ids), None, skipped_rules, None), None
        issues_df = pd.DataFrame(issues, columns=OUTPUT_COLUMNS)
        csv_path = output_dir / "population_validation_issues.csv"
        issues_df.to_csv(csv_path, index=False)
        return _summary(len(payroll_set), len(rk_set), len(common_ids), issues_df, skipped_rules, csv_path), csv_path
    except Exception as exc:
        return _base_summary(f"Population validation failed: {exc}"), None
