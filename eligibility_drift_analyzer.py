# eligibility_drift_analyzer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import date, timedelta

import pandas as pd

from issue_taxonomy import get_issue_metadata


@dataclass
class EligibilityDriftRecord:
    employee_id: str
    hire_date: Optional[date]
    dob: Optional[date]
    age_at_eligibility: Optional[int]
    computed_eligibility_date: Optional[date]
    first_deferral_date: Optional[date]
    drift_days: Optional[int]
    issue_type: str
    recommended_fix: str


def _safe_date(value: Any) -> Optional[date]:
    """Safely convert value to date, returning None if conversion fails."""
    try:
        if pd.isna(value):
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, pd.Timestamp):
            return value.date()
        # Try parsing as date string
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default if conversion fails."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    """Safely convert value to string, returning default if conversion fails."""
    try:
        if pd.isna(value):
            return default
        return str(value)
    except (ValueError, TypeError):
        return default


def _align_first_of_month(dt: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """Align a date to the first of the following month."""
    if dt is None or pd.isna(dt):
        return None
    # If already first of month, keep it
    if dt.day == 1:
        return dt.normalize()
    year = dt.year
    month = dt.month + 1
    if month > 12:
        month = 1
        year += 1
    return pd.Timestamp(year=year, month=month, day=1)


def _compute_eligibility_date_for_row(row: pd.Series, plan_rules: Optional[Dict[str, Any]]) -> Optional[pd.Timestamp]:
    """
    Compute eligibility date for a single row based on plan_rules.
    
    Args:
        row: pandas Series with hire_date, dob columns (may be None/NaT)
        plan_rules: dict with eligibility_rule, service_days_required, age_required, align_first_month
        
    Returns:
        pd.Timestamp for eligibility date, or None if cannot be computed
    """
    rules = plan_rules or {}
    rule_type = rules.get("eligibility_rule", "age21_and_1year")

    hire_ts = pd.to_datetime(row.get("hire_date"), errors="coerce")
    dob_ts = pd.to_datetime(row.get("dob"), errors="coerce")

    service_days_required = rules.get("service_days_required")
    age_required = rules.get("age_required")
    align_first_month = bool(rules.get("align_first_month", False))

    elig_dt = None

    if rule_type == "immediate":
        elig_dt = hire_ts

    elif rule_type == "age21_and_1year":
        service_days = service_days_required if isinstance(service_days_required, int) and service_days_required > 0 else 365
        age_req = age_required if isinstance(age_required, int) and age_required > 0 else 21
        service_dt = hire_ts + pd.Timedelta(days=service_days) if hire_ts is not None and not pd.isna(hire_ts) else None
        age_dt = dob_ts + pd.DateOffset(years=age_req) if dob_ts is not None and not pd.isna(dob_ts) else None
        candidates = [d for d in [service_dt, age_dt] if d is not None]
        elig_dt = max(candidates) if candidates else None

    elif rule_type == "service_only":
        service_days = service_days_required if isinstance(service_days_required, int) and service_days_required > 0 else 0
        elig_dt = hire_ts + pd.Timedelta(days=service_days) if hire_ts is not None and not pd.isna(hire_ts) else None

    elif rule_type == "age_only":
        age_req = age_required if isinstance(age_required, int) and age_required > 0 else 21
        elig_dt = dob_ts + pd.DateOffset(years=age_req) if dob_ts is not None and not pd.isna(dob_ts) else None

    else:
        # fallback to original
        service_dt = hire_ts + pd.Timedelta(days=365) if hire_ts is not None and not pd.isna(hire_ts) else None
        age_dt = dob_ts + pd.DateOffset(years=21) if dob_ts is not None and not pd.isna(dob_ts) else None
        candidates = [d for d in [service_dt, age_dt] if d is not None]
        elig_dt = max(candidates) if candidates else None

    if elig_dt is not None and not pd.isna(elig_dt) and align_first_month:
        elig_dt = _align_first_of_month(elig_dt)

    return elig_dt


def _compute_age_at_eligibility(dob_ts: Optional[pd.Timestamp], elig_ts: Optional[pd.Timestamp]) -> Optional[float]:
    """Compute age at eligibility date given date of birth and eligibility date."""
    if dob_ts is None or pd.isna(dob_ts) or elig_ts is None or pd.isna(elig_ts):
        return None
    delta_days = (elig_ts - dob_ts).days
    return round(delta_days / 365.25, 1)


def analyze_eligibility_drift(
    payroll_df: pd.DataFrame,
    output_dir: Path,
    plan_rules: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Analyze eligibility drift issues and write a CSV of drift records.

    Args:
        payroll_df: fully processed payroll dataframe from reconciliation (one row per contribution).
            Expected columns (where available):
            - employee_id: employee identifier
            - hire_date: hire date
            - dob: date of birth
            - pay_date: pay date
            - "EE Deferral $": regular pre-tax deferrals
            - "EE Roth $": regular Roth deferrals
        output_dir: directory where CSV outputs should be written.
        plan_rules: Optional dict with eligibility rules configuration:
            - eligibility_rule: "immediate", "age21_and_1year", "service_only", "age_only"
            - service_days_required: int (for service_only or age21_and_1year)
            - age_required: int (for age_only or age21_and_1year)
            - align_first_month: bool (align eligibility to first of following month)

    Returns:
        Tuple of (summary dict, csv_path):
        - summary dict with:
            - total_rows: total number of rows in payroll_df
            - eligibility_drift_count: number of drift records found
            - csv_path: string path to the drift records CSV
        - csv_path: Path to the CSV file (or None if no records)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if payroll_df.empty:
        # Return empty summary for empty dataframe
        csv_path = output_dir / "eligibility_drift.csv"
        pd.DataFrame(columns=[
            "employee_id", "hire_date", "dob", "age_at_eligibility",
            "computed_eligibility_date", "first_deferral_date", "drift_days",
            "issue_type", "recommended_fix",
            "issue_category", "severity", "correction_hint"
        ]).to_csv(csv_path, index=False)
        return {
            "total_rows": 0,
            "eligibility_drift_count": 0,
            "csv_path": str(csv_path),
        }, csv_path
    
    # Configurable grace period (hard-coded for now)
    grace_days = 30
    
    drift_records: list[EligibilityDriftRecord] = []
    
    # Work on a copy and parse date columns early
    df = payroll_df.copy()
    
    if "hire_date" in df.columns:
        df["hire_date"] = pd.to_datetime(df["hire_date"], errors="coerce")
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    if "pay_date" in df.columns:
        df["pay_date"] = pd.to_datetime(df["pay_date"], errors="coerce")
    
    # Safely extract required columns with defaults
    # Employee ID: use "employee_id" (mapped column from processed dataframe)
    if "employee_id" not in df.columns:
        # No employee_id column - cannot process
        drift_records = []
    else:
        # Group by employee to process each employee's records
        employee_groups = df.groupby("employee_id")
        
        for employee_id, emp_df in employee_groups:
            employee_id_str = _safe_str(employee_id, "")
            if not employee_id_str:
                continue
            
            # Extract dates safely (take first non-null value per employee)
            hire_date = None
            if "hire_date" in emp_df.columns:
                hire_date_series = emp_df["hire_date"].dropna()
                if not hire_date_series.empty:
                    hire_date_val = hire_date_series.iloc[0]
                    hire_date = _safe_date(hire_date_val)
            
            dob = None
            if "dob" in emp_df.columns:
                dob_series = emp_df["dob"].dropna()
                if not dob_series.empty:
                    dob_val = dob_series.iloc[0]
                    dob = _safe_date(dob_val)
            
            # Skip if neither hire_date nor dob exists
            if hire_date is None and dob is None:
                continue
            
            # Build a representative row for eligibility computation
            # Use the first row of the employee group
            representative_row = emp_df.iloc[0].copy()
            representative_row["hire_date"] = pd.to_datetime(hire_date, errors="coerce") if hire_date else None
            representative_row["dob"] = pd.to_datetime(dob, errors="coerce") if dob else None
            
            # Compute eligibility date using plan_rules
            computed_eligibility_ts = _compute_eligibility_date_for_row(representative_row, plan_rules)
            
            if computed_eligibility_ts is None or pd.isna(computed_eligibility_ts):
                continue
            
            # Convert to date for compatibility with existing logic
            computed_eligibility_date = computed_eligibility_ts.date()
            
            # Compute age at eligibility if dob exists
            dob_ts = pd.to_datetime(dob, errors="coerce") if dob else None
            age_at_eligibility_val = _compute_age_at_eligibility(dob_ts, computed_eligibility_ts)
            age_at_eligibility = int(age_at_eligibility_val) if age_at_eligibility_val is not None else None
            
            # Compute first contribution date per employee
            # Look for rows where ("EE Deferral $" + "EE Roth $") > 0
            first_contribution_date = None
            
            if "pay_date" in emp_df.columns:
                # Parse pay dates
                pay_dates = pd.to_datetime(emp_df["pay_date"], errors="coerce")
                
                # Get deferral amounts
                deferral_pretax = emp_df.get("EE Deferral $", pd.Series([0.0] * len(emp_df), index=emp_df.index))
                deferral_roth = emp_df.get("EE Roth $", pd.Series([0.0] * len(emp_df), index=emp_df.index))
                
                # Convert to numeric
                deferral_pretax = pd.to_numeric(deferral_pretax, errors="coerce").fillna(0.0)
                deferral_roth = pd.to_numeric(deferral_roth, errors="coerce").fillna(0.0)
                
                # Total contribution
                total_contribution = deferral_pretax + deferral_roth
                
                # Find rows with contributions > 0
                contribution_mask = total_contribution > 0
                contribution_dates = pay_dates[contribution_mask].dropna()
                
                if not contribution_dates.empty:
                    # Get earliest contribution date
                    first_contribution_date = contribution_dates.min().date()
            
            # Check for drift issues
            issue_type = None
            recommended_fix = None
            drift_days_val = None
            
            # Case 1: Eligibility date known but no contributions
            if computed_eligibility_date and first_contribution_date is None:
                issue_type = "No contributions after eligibility"
                recommended_fix = "Review participant for missed eligibility; consider correction under plan rules and EPCRS."
                drift_days_val = None
            
            # Case 2: Both dates exist and first_contribution_date > computed_eligibility_date + grace_days
            elif computed_eligibility_date and first_contribution_date:
                grace_threshold = computed_eligibility_date + timedelta(days=grace_days)
                if first_contribution_date > grace_threshold:
                    issue_type = "Late start after eligibility"
                    drift_days_val = (first_contribution_date - computed_eligibility_date).days
                    recommended_fix = "Review delayed start against plan provisions; consider eligibility correction if required."
            
            # Create record if issue detected
            if issue_type:
                drift_records.append(EligibilityDriftRecord(
                    employee_id=employee_id_str,
                    hire_date=hire_date,
                    dob=dob,
                    age_at_eligibility=age_at_eligibility,
                    computed_eligibility_date=computed_eligibility_date,
                    first_deferral_date=first_contribution_date,
                    drift_days=drift_days_val,
                    issue_type=issue_type,
                    recommended_fix=recommended_fix,
                ))
    
    # Build DataFrame from drift records
    drift_df = pd.DataFrame(
        [
            {
                "employee_id": r.employee_id,
                "hire_date": r.hire_date.isoformat() if r.hire_date else None,
                "dob": r.dob.isoformat() if r.dob else None,
                "age_at_eligibility": r.age_at_eligibility,
                "computed_eligibility_date": r.computed_eligibility_date.isoformat() if r.computed_eligibility_date else None,
                "first_deferral_date": r.first_deferral_date.isoformat() if r.first_deferral_date else None,
                "drift_days": r.drift_days,
                "issue_type": r.issue_type,
                "recommended_fix": r.recommended_fix,
                **get_issue_metadata(r.issue_type),
            }
            for r in drift_records
        ]
    )
    
    # Write CSV (even if empty)
    csv_path = output_dir / "eligibility_drift.csv"
    drift_df.to_csv(csv_path, index=False)
    
    summary = {
        "total_rows": int(len(payroll_df)),
        "eligibility_drift_count": int(len(drift_records)),
        "csv_path": str(csv_path),
    }
    
    return summary, csv_path

