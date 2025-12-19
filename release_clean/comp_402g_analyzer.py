# comp_402g_analyzer.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from issue_taxonomy import get_issue_metadata


# IRS 402(g) elective deferral limits by plan year
BASE_402G_LIMITS = {
    2023: 22500,
    2024: 23000,
}

CATCHUP_402G_LIMITS = {
    2023: 7500,
    2024: 7500,
}


def _get_base_limit(year: int) -> int:
    """Get base 402(g) limit for a given year, falling back to latest known year if not found."""
    if year in BASE_402G_LIMITS:
        return BASE_402G_LIMITS[year]
    return BASE_402G_LIMITS[max(BASE_402G_LIMITS.keys())]


def _get_catchup_limit(year: int) -> int:
    """Get catch-up 402(g) limit for a given year, falling back to latest known year if not found."""
    if year in CATCHUP_402G_LIMITS:
        return CATCHUP_402G_LIMITS[year]
    return CATCHUP_402G_LIMITS[max(CATCHUP_402G_LIMITS.keys())]


def analyze_comp_402g_limits(
    payroll_df: pd.DataFrame,
    output_dir: Path,
    plan_year: int,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Analyze participant elective deferrals against IRS 402(g) limits for a given plan year.

    Args:
        payroll_df: fully processed payroll dataframe from reconciliation (one row per contribution).
            Expected columns (where available):
            - employee_id: employee identifier (required)
            - pay_date: pay date (required, will be converted to datetime)
            - "EE Deferral $": regular pre-tax deferrals (defaults to 0 if missing)
            - "EE Roth $": regular Roth deferrals (defaults to 0 if missing)
            - catchup_pretax: catch-up pre-tax amount (defaults to 0 if missing)
            - catchup_roth: catch-up Roth amount (defaults to 0 if missing)
            - age: age (optional, used to determine catch-up eligibility)
        output_dir: directory where CSV outputs should be written.
        plan_year: integer plan year (e.g., 2024).

    Returns:
        Tuple of (summary_dict, csv_path):
        - summary_dict contains:
          - total_participants: count of distinct employee_id evaluated for the plan_year
          - excess_violation_count: number of violating participants
          - csv_path: string path to violations CSV, or None if no violations
          - warning: optional warning message if analysis could not be performed
        - csv_path: Path object to violations CSV, or None if no violations
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for required columns
        if payroll_df.empty:
            return (
                {
                    "total_participants": 0,
                    "excess_violation_count": 0,
                    "csv_path": None,
                    "warning": "Payroll dataframe is empty",
                },
                None,
            )
        
        missing_cols = []
        if "employee_id" not in payroll_df.columns:
            missing_cols.append("employee_id")
        if "pay_date" not in payroll_df.columns:
            missing_cols.append("pay_date")
        
        if missing_cols:
            warning_msg = f"Cannot perform 402(g) analysis because required columns are missing: {', '.join(missing_cols)}"
            print(f"[WARN] comp_402g_analyzer: {warning_msg}")
            return (
                {
                    "total_participants": 0,
                    "excess_violation_count": 0,
                    "csv_path": None,
                    "warning": warning_msg,
                },
                None,
            )
        
        # Make a copy to avoid modifying the input
        df = payroll_df.copy()
        
        # Convert pay_date to datetime, dropping rows where conversion fails
        original_len = len(df)
        df["pay_date"] = pd.to_datetime(df["pay_date"], errors="coerce")
        df = df.dropna(subset=["pay_date"])
        
        if len(df) == 0:
            warning_msg = "No valid pay_date values found after conversion"
            print(f"[WARN] comp_402g_analyzer: {warning_msg}")
            return (
                {
                    "total_participants": 0,
                    "excess_violation_count": 0,
                    "csv_path": None,
                    "warning": warning_msg,
                },
                None,
            )
        
        if len(df) < original_len:
            print(f"[INFO] comp_402g_analyzer: Dropped {original_len - len(df)} rows with invalid pay_date")
        
        # Add plan_year column from pay_date
        df["plan_year"] = df["pay_date"].dt.year
        
        # Filter to rows matching the requested plan_year
        df = df[df["plan_year"] == plan_year].copy()
        
        if len(df) == 0:
            warning_msg = f"No rows found for plan_year {plan_year}"
            print(f"[INFO] comp_402g_analyzer: {warning_msg}")
            return (
                {
                    "total_participants": 0,
                    "excess_violation_count": 0,
                    "csv_path": None,
                    "warning": warning_msg,
                },
                None,
            )
        
        # Extract deferral amounts safely with defaults
        ee_deferral = df.get("EE Deferral $", pd.Series([0.0] * len(df), index=df.index))
        ee_roth = df.get("EE Roth $", pd.Series([0.0] * len(df), index=df.index))
        catchup_pretax = df.get("catchup_pretax", pd.Series([0.0] * len(df), index=df.index))
        catchup_roth = df.get("catchup_roth", pd.Series([0.0] * len(df), index=df.index))
        
        # Convert to numeric, filling invalid values with 0
        ee_deferral = pd.to_numeric(ee_deferral, errors="coerce").fillna(0.0)
        ee_roth = pd.to_numeric(ee_roth, errors="coerce").fillna(0.0)
        catchup_pretax = pd.to_numeric(catchup_pretax, errors="coerce").fillna(0.0)
        catchup_roth = pd.to_numeric(catchup_roth, errors="coerce").fillna(0.0)
        
        # Compute per-row amounts
        df["base_deferrals"] = ee_deferral + ee_roth
        df["catchup_deferrals"] = catchup_pretax + catchup_roth
        
        # Group by employee_id and plan_year, then aggregate
        agg_dict = {
            "base_deferrals": "sum",
            "catchup_deferrals": "sum",
        }
        
        # Add age aggregation if age column exists
        if "age" in df.columns:
            age_series = pd.to_numeric(df["age"], errors="coerce")
            agg_dict["age"] = "max"
        
        grouped = df.groupby(["employee_id", "plan_year"], as_index=False).agg(agg_dict)
        
        # Rename aggregated columns
        grouped = grouped.rename(columns={
            "base_deferrals": "total_base_deferrals",
            "catchup_deferrals": "total_catchup_deferrals",
        })
        
        # Determine catch-up eligibility per employee-year
        grouped["has_catchup"] = grouped["total_catchup_deferrals"] > 0
        
        if "age" in grouped.columns:
            grouped["max_age"] = grouped["age"]
            grouped["is_catchup_eligible"] = (
                (grouped["max_age"].notna() & (grouped["max_age"] >= 50)) | grouped["has_catchup"]
            )
        else:
            grouped["max_age"] = None
            grouped["is_catchup_eligible"] = grouped["has_catchup"]
        
        # Get limits for plan_year
        base_limit = _get_base_limit(plan_year)
        catchup_limit = _get_catchup_limit(plan_year)
        
        # Calculate allowed limit and check for violations
        grouped["base_limit"] = base_limit
        grouped["catchup_limit"] = catchup_limit
        grouped["allowed_limit"] = grouped["base_limit"] + (
            grouped["catchup_limit"].where(grouped["is_catchup_eligible"], 0)
        )
        grouped["total_deferrals"] = grouped["total_base_deferrals"] + grouped["total_catchup_deferrals"]
        grouped["excess_amount"] = (grouped["total_deferrals"] - grouped["allowed_limit"]).clip(lower=0)
        
        # Filter to violations only
        violations = grouped[grouped["excess_amount"] > 0].copy()
        
        # Prepare summary
        total_participants = len(grouped)
        excess_violation_count = len(violations)
        
        # If no violations, return summary with csv_path = None
        if excess_violation_count == 0:
            return (
                {
                    "total_participants": total_participants,
                    "excess_violation_count": 0,
                    "csv_path": None,
                },
                None,
            )
        
        # Get issue metadata from taxonomy
        issue_type = "402(g) excess deferrals"
        meta = get_issue_metadata(issue_type)
        
        # Build violations dataframe with required columns including taxonomy fields
        violations_list = []
        for _, row in violations.iterrows():
            violation_row = {
                "employee_id": row["employee_id"],
                "plan_year": row["plan_year"],
                "total_deferrals": row["total_deferrals"],
                "total_base_deferrals": row["total_base_deferrals"],
                "total_catchup_deferrals": row["total_catchup_deferrals"],
                "is_catchup_eligible": row["is_catchup_eligible"],
                "base_limit": row["base_limit"],
                "catchup_limit": row["catchup_limit"],
                "allowed_limit": row["allowed_limit"],
                "excess_amount": row["excess_amount"],
                "issue_type": issue_type,
                "issue_category": meta["issue_category"],
                "severity": meta["severity"],
                "correction_hint": meta["correction_hint"],
            }
            violations_list.append(violation_row)
        
        violations_df = pd.DataFrame(violations_list)
        
        # Write CSV
        csv_path = output_dir / "comp_402g_violations.csv"
        violations_df.to_csv(csv_path, index=False)
        
        return (
            {
                "total_participants": total_participants,
                "excess_violation_count": excess_violation_count,
                "csv_path": str(csv_path),
            },
            csv_path,
        )
    
    except Exception as e:
        warning_msg = f"Unexpected error in comp_402g_analyzer: {str(e)}"
        print(f"[WARN] comp_402g_analyzer: {warning_msg}")
        return (
            {
                "total_participants": 0,
                "excess_violation_count": 0,
                "csv_path": None,
                "warning": warning_msg,
            },
            None,
        )
