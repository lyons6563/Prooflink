# eligibility_drift_analyzer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import date, timedelta

import pandas as pd


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


def analyze_eligibility_drift(
    payroll_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
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

    Returns:
        summary dict with:
        - total_rows: total number of rows in payroll_df
        - eligibility_drift_count: number of drift records found
        - csv_path: string path to the drift records CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if payroll_df.empty:
        # Return empty summary for empty dataframe
        csv_path = output_dir / "eligibility_drift.csv"
        pd.DataFrame(columns=[
            "employee_id", "hire_date", "dob", "age_at_eligibility",
            "computed_eligibility_date", "first_deferral_date", "drift_days",
            "issue_type", "recommended_fix"
        ]).to_csv(csv_path, index=False)
        return {
            "total_rows": 0,
            "eligibility_drift_count": 0,
            "csv_path": str(csv_path),
        }
    
    # Configurable grace period (hard-coded for now)
    grace_days = 30
    
    drift_records: list[EligibilityDriftRecord] = []
    
    # Safely extract required columns with defaults
    # Employee ID: use "employee_id" (mapped column from processed dataframe)
    if "employee_id" not in payroll_df.columns:
        # No employee_id column - cannot process
        drift_records = []
    else:
        # Group by employee to process each employee's records
        employee_groups = payroll_df.groupby("employee_id")
        
        for employee_id, emp_df in employee_groups:
            employee_id_str = _safe_str(employee_id, "")
            if not employee_id_str:
                continue
            
            # Extract dates safely (take first non-null value per employee)
            hire_date = None
            if "hire_date" in emp_df.columns:
                hire_date_series = emp_df["hire_date"].dropna()
                if not hire_date_series.empty:
                    hire_date = _safe_date(hire_date_series.iloc[0])
            
            dob = None
            if "dob" in emp_df.columns:
                dob_series = emp_df["dob"].dropna()
                if not dob_series.empty:
                    dob = _safe_date(dob_series.iloc[0])
            
            # Skip if neither hire_date nor dob exists
            if hire_date is None and dob is None:
                continue
            
            # Compute eligibility date
            computed_eligibility_date = None
            hire_eligibility_date = None
            age21_eligibility_date = None
            
            if hire_date:
                # hire_date + 1 year (365 days)
                hire_eligibility_date = hire_date + timedelta(days=365)
            
            if dob:
                # dob + 21 years
                age21_eligibility_date = date(dob.year + 21, dob.month, dob.day)
            
            # Final eligibility date = max(hire_date+1Y, age-21 date) when both exist
            if hire_eligibility_date and age21_eligibility_date:
                computed_eligibility_date = max(hire_eligibility_date, age21_eligibility_date)
            elif hire_eligibility_date:
                computed_eligibility_date = hire_eligibility_date
            elif age21_eligibility_date:
                computed_eligibility_date = age21_eligibility_date
            
            if computed_eligibility_date is None:
                continue
            
            # Compute age at eligibility if dob exists
            age_at_eligibility = None
            if dob and computed_eligibility_date:
                try:
                    age_at_eligibility = (computed_eligibility_date - dob).days // 365
                except Exception:
                    age_at_eligibility = None
            
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
    
    return summary

