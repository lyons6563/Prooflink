from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from issue_taxonomy import get_issue_metadata


ISSUE_UNDER_MATCH = "Under-match from compensation definition variance"
ISSUE_OVER_MATCH = "Over-match from excluded compensation included"
ISSUE_EXCLUDED_CLASS = "Match paid to excluded employee class"
ISSUE_TRUE_UP = "Potential annual true-up timing difference"
ISSUE_INCOMPLETE = "Compensation source data incomplete"

ROOT_MISSING_ELIGIBLE_COMP = "missing_eligible_compensation"
ROOT_EXCLUDED_COMP_INCLUDED = "excluded_compensation_included"
ROOT_CLASS_EXCLUSION = "employee_class_exclusion_issue"
ROOT_FORMULA_VARIANCE = "match_formula_variance"
ROOT_TRUE_UP_TIMING = "annual_true_up_timing_difference"
ROOT_MISSING_MATCH = "recordkeeper_or_payroll_missing_match"
ROOT_SOURCE_INCOMPLETE = "source_data_incomplete"
ROOT_UNKNOWN = "unknown_requires_review"


DEFAULT_CONFIG: Dict[str, Any] = {
    "match_formula_name": "50% up to 6%",
    "match_type": "percent_of_comp",
    "match_rate": 0.50,
    "match_cap_pct": 0.06,
    "match_frequency": "per_payroll",
    "true_up_enabled": False,
    "eligible_comp_columns": ["Eligibility Compensation", "Plan Compensation", "Compensation"],
    "excluded_comp_columns": [],
    "employee_class_column": "Employee Class",
    "eligible_classes": [],
    "excluded_classes": [],
    "actual_match_column": "ER Match $",
    "gross_comp_columns": ["Gross Compensation", "Gross Comp", "Total Compensation"],
    "absolute_tolerance": 5.00,
    "relative_tolerance_pct": 0.15,
}

DEFERRAL_COLUMNS = [
    "EE Deferral $",
    "EE Roth $",
    "deferral_amount",
    "roth_amount",
    "catchup_pretax",
    "catchup_roth",
]

OUTPUT_COLUMNS = [
    "run_id",
    "plan_name",
    "plan_year",
    "employee_id",
    "pay_date",
    "employee_class",
    "eligible_comp",
    "excluded_comp",
    "employee_deferrals",
    "deferral_pct",
    "actual_match",
    "expected_match",
    "match_variance",
    "match_variance_abs",
    "match_variance_pct",
    "issue_type",
    "likely_root_cause",
    "issue_category",
    "severity",
    "correction_hint",
    "gross_comp",
    "payroll_file_row_number",
    "recordkeeper_file_row_number",
    "source_comp_columns_used",
    "source_match_column_used",
]


def _merge_config(plan_match_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if plan_match_config:
        config.update(plan_match_config)
    return config


def _present_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _numeric_sum(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    present = _present_columns(df, columns)
    if not present:
        return pd.Series([0.0] * len(df), index=df.index)
    total = pd.Series([0.0] * len(df), index=df.index)
    for column in present:
        total = total + _numeric_series(df, column)
    return total


def _first_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    present = _present_columns(df, columns)
    if not present:
        return pd.Series([0.0] * len(df), index=df.index)
    return _numeric_series(df, present[0])


def _is_class_excluded(employee_class: Any, config: Dict[str, Any]) -> bool:
    if employee_class is None or pd.isna(employee_class):
        return False
    value = str(employee_class).strip()
    excluded = {str(item).strip() for item in config.get("excluded_classes", [])}
    eligible = {str(item).strip() for item in config.get("eligible_classes", [])}
    if value in excluded:
        return True
    return bool(eligible) and value not in eligible


def _variance_pct(variance_abs: float, expected_match: float, actual_match: float) -> float:
    denominator = abs(expected_match)
    if denominator <= 0:
        denominator = max(abs(actual_match), 1.0)
    return variance_abs / denominator


def _outside_tolerance(
    variance_abs: float,
    variance_pct: float,
    absolute_tolerance: float,
    relative_tolerance_pct: float,
) -> bool:
    return variance_abs > absolute_tolerance and variance_pct > relative_tolerance_pct


def _metadata(issue_type: str) -> Dict[str, str]:
    meta = get_issue_metadata(issue_type)
    return {
        "issue_category": meta["issue_category"],
        "severity": meta["severity"],
        "correction_hint": meta["correction_hint"],
    }


def _summary(
    total_rows: int,
    participants: int,
    csv_path: Optional[Path],
    issues_df: Optional[pd.DataFrame] = None,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    if issues_df is None or issues_df.empty:
        result = {
            "total_rows_evaluated": total_rows,
            "participants_evaluated": participants,
            "issue_count": 0,
            "under_match_count": 0,
            "over_match_count": 0,
            "excluded_class_match_count": 0,
            "estimated_under_match_dollars": 0.0,
            "estimated_over_match_dollars": 0.0,
            "csv_path": None,
        }
    else:
        under = issues_df[issues_df["match_variance"] < 0]
        over = issues_df[issues_df["match_variance"] > 0]
        result = {
            "total_rows_evaluated": total_rows,
            "participants_evaluated": participants,
            "issue_count": int(len(issues_df)),
            "under_match_count": int((issues_df["issue_type"] == ISSUE_UNDER_MATCH).sum()),
            "over_match_count": int((issues_df["issue_type"] == ISSUE_OVER_MATCH).sum()),
            "excluded_class_match_count": int((issues_df["issue_type"] == ISSUE_EXCLUDED_CLASS).sum()),
            "estimated_under_match_dollars": round(float(under["match_variance_abs"].sum()), 2),
            "estimated_over_match_dollars": round(float(over["match_variance_abs"].sum()), 2),
            "csv_path": str(csv_path) if csv_path else None,
        }
    if warning:
        result["warning"] = warning
    return result


def analyze_compensation_match(
    payroll_df: pd.DataFrame,
    output_dir: Path,
    plan_match_config: Optional[Dict[str, Any]] = None,
    *,
    run_id: Optional[str] = None,
    plan_name: Optional[str] = None,
    plan_year: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Detect compensation-definition and employer-match impact issues.

    The analyzer is deterministic and standalone: it uses the supplied payroll
    dataframe plus explicit plan_match_config, writes compensation_match_issues.csv
    only when issues exist, and returns warning metadata instead of raising on
    incomplete source data.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if payroll_df is None or payroll_df.empty:
            return _summary(0, 0, None, warning="Payroll dataframe is empty"), None

        config = _merge_config(plan_match_config)
        if config.get("match_type") != "percent_of_comp":
            warning = "Compensation match analysis supports only match_type=percent_of_comp."
            return _summary(0, 0, None, warning=warning), None

        required_columns = ["employee_id", "pay_date"]
        missing_required = [col for col in required_columns if col not in payroll_df.columns]

        eligible_columns = _present_columns(payroll_df, config.get("eligible_comp_columns", []))
        match_column = config.get("actual_match_column", "ER Match $")
        if match_column not in payroll_df.columns:
            fallback_match_cols = _present_columns(
                payroll_df,
                ["ER Match $", "Employer Match", "Employer Match $", "employer_match", "match_amount"],
            )
            match_column = fallback_match_cols[0] if fallback_match_cols else None

        if not eligible_columns:
            missing_required.append("eligible compensation columns")
        if match_column is None:
            missing_required.append("actual match column")

        total_rows = int(len(payroll_df))
        participants = int(payroll_df["employee_id"].nunique()) if "employee_id" in payroll_df.columns else 0

        if missing_required:
            warning = "Compensation match analysis skipped because source data is missing: "
            warning += ", ".join(missing_required)
            return _summary(total_rows, participants, None, warning=warning), None

        df = payroll_df.copy()
        df["pay_date"] = pd.to_datetime(df["pay_date"], errors="coerce")
        df = df.dropna(subset=["pay_date"]).copy()
        if df.empty:
            return _summary(0, 0, None, warning="No valid pay_date values found."), None

        total_rows = int(len(df))
        participants = int(df["employee_id"].nunique())

        excluded_columns = _present_columns(df, config.get("excluded_comp_columns", []))
        class_column = config.get("employee_class_column")
        gross_comp_columns = config.get("gross_comp_columns", [])
        gross_comp_present = _present_columns(df, gross_comp_columns)
        deferral_columns = _present_columns(df, DEFERRAL_COLUMNS)

        df["eligible_comp"] = _numeric_sum(df, eligible_columns)
        df["excluded_comp"] = _numeric_sum(df, excluded_columns)
        df["employee_deferrals"] = _numeric_sum(df, deferral_columns)
        df["actual_match"] = _numeric_series(df, match_column)
        df["gross_comp"] = _first_numeric(df, gross_comp_present)

        match_rate = float(config.get("match_rate", 0.50))
        match_cap_pct = float(config.get("match_cap_pct", 0.06))
        absolute_tolerance = float(config.get("absolute_tolerance", 5.00))
        relative_tolerance_pct = float(config.get("relative_tolerance_pct", 0.15))
        true_up_mode = bool(config.get("true_up_enabled")) or config.get("match_frequency") == "annual"

        issues: List[Dict[str, Any]] = []

        for index, row in df.iterrows():
            eligible_comp = float(row["eligible_comp"])
            actual_match = float(row["actual_match"])
            employee_deferrals = float(row["employee_deferrals"])
            excluded_comp = float(row["excluded_comp"])
            gross_comp = float(row["gross_comp"])
            employee_class = row.get(class_column) if class_column in df.columns else None
            class_excluded = _is_class_excluded(employee_class, config)

            if class_excluded:
                expected_match = 0.0
            elif eligible_comp > 0:
                deferral_pct = employee_deferrals / eligible_comp
                eligible_pct_for_match = min(deferral_pct, match_cap_pct)
                expected_match = eligible_comp * eligible_pct_for_match * match_rate
            else:
                deferral_pct = 0.0
                expected_match = 0.0

            if eligible_comp > 0:
                deferral_pct = employee_deferrals / eligible_comp
            else:
                deferral_pct = 0.0

            match_variance = actual_match - expected_match
            match_variance_abs = abs(match_variance)
            match_variance_pct = _variance_pct(match_variance_abs, expected_match, actual_match)

            issue_type: Optional[str] = None
            likely_root_cause: Optional[str] = None

            if class_excluded and actual_match > absolute_tolerance:
                issue_type = ISSUE_EXCLUDED_CLASS
                likely_root_cause = ROOT_CLASS_EXCLUSION
            elif eligible_comp <= 0 and (employee_deferrals > 0 or actual_match > 0):
                issue_type = ISSUE_INCOMPLETE
                likely_root_cause = ROOT_SOURCE_INCOMPLETE
            elif _outside_tolerance(
                match_variance_abs,
                match_variance_pct,
                absolute_tolerance,
                relative_tolerance_pct,
            ):
                if match_variance < 0:
                    if true_up_mode:
                        issue_type = ISSUE_TRUE_UP
                        likely_root_cause = ROOT_TRUE_UP_TIMING
                    else:
                        issue_type = ISSUE_UNDER_MATCH
                        if gross_comp > eligible_comp or excluded_comp == 0:
                            likely_root_cause = ROOT_MISSING_ELIGIBLE_COMP
                        else:
                            likely_root_cause = ROOT_FORMULA_VARIANCE
                elif match_variance > 0:
                    if excluded_comp > 0:
                        issue_type = ISSUE_OVER_MATCH
                        likely_root_cause = ROOT_EXCLUDED_COMP_INCLUDED
                    elif expected_match == 0:
                        issue_type = ISSUE_INCOMPLETE
                        likely_root_cause = ROOT_MISSING_MATCH
                    else:
                        issue_type = ISSUE_OVER_MATCH
                        likely_root_cause = ROOT_FORMULA_VARIANCE

            if issue_type is None:
                continue

            meta = _metadata(issue_type)
            issues.append(
                {
                    "run_id": run_id,
                    "plan_name": plan_name,
                    "plan_year": plan_year,
                    "employee_id": row["employee_id"],
                    "pay_date": row["pay_date"].date().isoformat(),
                    "employee_class": employee_class,
                    "eligible_comp": round(eligible_comp, 2),
                    "excluded_comp": round(excluded_comp, 2),
                    "employee_deferrals": round(employee_deferrals, 2),
                    "deferral_pct": round(deferral_pct, 6),
                    "actual_match": round(actual_match, 2),
                    "expected_match": round(expected_match, 2),
                    "match_variance": round(match_variance, 2),
                    "match_variance_abs": round(match_variance_abs, 2),
                    "match_variance_pct": round(match_variance_pct, 6),
                    "issue_type": issue_type,
                    "likely_root_cause": likely_root_cause or ROOT_UNKNOWN,
                    "issue_category": meta["issue_category"],
                    "severity": meta["severity"],
                    "correction_hint": meta["correction_hint"],
                    "gross_comp": round(gross_comp, 2),
                    "payroll_file_row_number": row.get("payroll_file_row_number"),
                    "recordkeeper_file_row_number": row.get("recordkeeper_file_row_number"),
                    "source_comp_columns_used": ",".join(eligible_columns),
                    "source_match_column_used": match_column,
                }
            )

        if not issues:
            return _summary(total_rows, participants, None), None

        issues_df = pd.DataFrame(issues, columns=OUTPUT_COLUMNS)
        csv_path = output_dir / "compensation_match_issues.csv"
        issues_df.to_csv(csv_path, index=False)

        return _summary(total_rows, participants, csv_path, issues_df), csv_path

    except Exception as exc:
        warning = f"Compensation match analysis failed: {exc}"
        return _summary(0, 0, None, warning=warning), None
