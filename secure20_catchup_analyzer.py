# secure20_catchup_analyzer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from datetime import datetime


@dataclass
class Secure20Violation:
    employee_id: str
    age: int
    is_hce: bool
    deferral_pretax: float
    deferral_roth: float
    catchup_amount: float
    violation_type: str
    recommended_fix: str


def _safe_get_column(df: pd.DataFrame, col_name: str, default_value: Any = None) -> pd.Series:
    """
    Safely get a column from dataframe, returning a Series with default_value if column doesn't exist.
    """
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([default_value] * len(df), index=df.index)


def _compute_age_from_dob(dob_series: pd.Series, pay_date_series: pd.Series) -> pd.Series:
    """
    Compute age from date of birth and pay date.
    Returns Series with integer ages, or -1 where calculation is not possible.
    """
    age_series = pd.Series([-1] * len(dob_series), index=dob_series.index, dtype=int)
    
    # Try to parse dates
    dob_parsed = pd.to_datetime(dob_series, errors="coerce")
    pay_date_parsed = pd.to_datetime(pay_date_series, errors="coerce")
    
    # Calculate age where both dates are valid
    valid_mask = dob_parsed.notna() & pay_date_parsed.notna()
    if valid_mask.any():
        age_series.loc[valid_mask] = (
            (pay_date_parsed.loc[valid_mask] - dob_parsed.loc[valid_mask]).dt.days / 365.25
        ).astype(int)
    
    return age_series


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default if conversion fails."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool, returning default if conversion fails."""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "y")
        return default
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


def analyze_secure20_catchup(
    payroll_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Analyze Secure 2.0 catch-up issues and write a CSV of violations.

    Args:
        payroll_df: fully processed payroll dataframe from reconciliation (one row per contribution).
            Expected columns from processed dataframe:
            - employee_id: employee identifier (mapped column)
            - pay_date: pay date (mapped column)
            - age: age (may exist, or computed from dob and pay_date)
            - is_hce: HCE indicator (boolean, added during processing)
            - "EE Deferral $": regular pre-tax deferrals (normalized column name)
            - "EE Roth $": regular Roth deferrals (normalized column name)
            - catchup_pretax: catch-up pre-tax amount (added during processing)
            - catchup_roth: catch-up Roth amount (added during processing)
        output_dir: directory where CSV outputs should be written.

    Returns:
        summary dict with:
        - total_rows: total number of rows in payroll_df
        - total_violations: number of violations found
        - hce_violation_count: count of Rule A violations (HCE catch-up not Roth)
        - potential_catchup_miscode_count: count of Rule B violations (catch-up in base deferrals)
        - csv_path: string path to the violations CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if payroll_df.empty:
        # Return empty summary for empty dataframe
        csv_path = output_dir / "secure20_violations.csv"
        pd.DataFrame(columns=[
            "employee_id", "age", "is_hce", "deferral_pretax", "deferral_roth",
            "catchup_amount", "violation_type", "recommended_fix"
        ]).to_csv(csv_path, index=False)
        return {
            "total_rows": 0,
            "total_violations": 0,
            "hce_violation_count": 0,
            "potential_catchup_miscode_count": 0,
            "csv_path": str(csv_path),
        }
    
    violations: List[Secure20Violation] = []
    
    # Compute age series: try age column first, then compute from dob and pay_date
    age_series = None
    if "age" in payroll_df.columns:
        age_series = payroll_df["age"].copy()
        # Convert to numeric, filling invalid values with -1
        age_series = pd.to_numeric(age_series, errors="coerce").fillna(-1).astype(int)
        # If all ages are invalid (-1), try computing from dob
        if (age_series == -1).all():
            age_series = None
    
    # If age column doesn't exist or all values are invalid, try to compute from dob
    if age_series is None:
        # Try to find dob column
        dob_col = None
        for col_name in ["dob", "date_of_birth", "birth_date", "Date of Birth"]:
            if col_name in payroll_df.columns:
                dob_col = col_name
                break
        
        # Try to find pay_date column (should be present in processed dataframe)
        if dob_col and "pay_date" in payroll_df.columns:
            dob_series = payroll_df[dob_col]
            pay_date_series = payroll_df["pay_date"]
            age_series = _compute_age_from_dob(dob_series, pay_date_series)
        else:
            # No age or dob available, create series of -1
            age_series = pd.Series([-1] * len(payroll_df), index=payroll_df.index, dtype=int)
    
    # Process each row using df.get() for robust column access
    for idx, row in payroll_df.iterrows():
        # Extract values safely using df.get() with defaults
        # Employee ID: use "employee_id" (mapped column from processed dataframe)
        employee_id = _safe_str(row.get("employee_id", ""), "")
        
        # Age: get from computed age_series
        age_val = _safe_float(age_series.loc[idx] if idx in age_series.index else -1, -1)
        age_50_plus = age_val >= 50 if age_val >= 0 else False
        
        # HCE indicator: use "is_hce" (boolean column added during processing)
        is_hce = _safe_bool(row.get("is_hce", False), False)
        
        # Regular deferrals: use normalized column names from processed dataframe
        deferral_pretax = _safe_float(row.get("EE Deferral $", 0.0), 0.0)
        deferral_roth = _safe_float(row.get("EE Roth $", 0.0), 0.0)
        
        # Catch-up amounts: use columns added during processing
        catchup_pretax = _safe_float(row.get("catchup_pretax", 0.0), 0.0)
        catchup_roth = _safe_float(row.get("catchup_roth", 0.0), 0.0)
        total_catchup = catchup_pretax + catchup_roth
        
        # Rule A: HCE catch-up not coded as Roth
        # If is_hce is True AND age >= 50 AND catchup_pretax > 0 AND catchup_roth == 0
        if is_hce and age_50_plus and catchup_pretax > 0 and catchup_roth == 0:
            violations.append(Secure20Violation(
                employee_id=employee_id,
                age=int(age_val) if age_val >= 0 else -1,
                is_hce=is_hce,
                deferral_pretax=deferral_pretax,
                deferral_roth=deferral_roth,
                catchup_amount=total_catchup,
                violation_type="HCE catch-up not coded as Roth",
                recommended_fix="Confirm Secure 2.0 HCE Roth requirement; reclassify HCE catch-up to Roth source if needed."
            ))
        
        # Rule B: Potential catch-up mis-coded in base deferrals
        # If age >= 50 AND (catchup_pretax + catchup_roth) == 0
        if age_50_plus and total_catchup == 0:
            violations.append(Secure20Violation(
                employee_id=employee_id,
                age=int(age_val) if age_val >= 0 else -1,
                is_hce=is_hce,
                deferral_pretax=deferral_pretax,
                deferral_roth=deferral_roth,
                catchup_amount=total_catchup,
                violation_type="Potential catch-up coded in base deferral source",
                recommended_fix="Review age 50+ deferrals to confirm whether catch-up amounts are tracked separately from base deferrals."
            ))
    
    # Count violations by type
    hce_violation_count = sum(1 for v in violations if v.violation_type == "HCE catch-up not coded as Roth")
    potential_catchup_miscode_count = sum(
        1 for v in violations if v.violation_type == "Potential catch-up coded in base deferral source"
    )
    
    # Build DataFrame from violations
    violations_df = pd.DataFrame(
        [
            {
                "employee_id": v.employee_id,
                "age": v.age,
                "is_hce": v.is_hce,
                "deferral_pretax": v.deferral_pretax,
                "deferral_roth": v.deferral_roth,
                "catchup_amount": v.catchup_amount,
                "violation_type": v.violation_type,
                "recommended_fix": v.recommended_fix,
            }
            for v in violations
        ]
    )
    
    # Write CSV (even if empty)
    csv_path = output_dir / "secure20_violations.csv"
    violations_df.to_csv(csv_path, index=False)
    
    summary = {
        "total_rows": int(len(payroll_df)),
        "total_violations": int(len(violations)),
        "hce_violation_count": hce_violation_count,
        "potential_catchup_miscode_count": potential_catchup_miscode_count,
        "csv_path": str(csv_path),
    }
    
    return summary
