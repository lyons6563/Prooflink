# match_reasonableness_analyzer.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from issue_taxonomy import get_issue_metadata


def analyze_match_reasonableness(
    payroll_df: pd.DataFrame,
    output_dir: Path,
    plan_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Analyze employer match amounts versus a simple match formula and flag
    rows where the employer match appears materially under or over the expected amount.

    Args:
        payroll_df: fully processed payroll dataframe from reconciliation (one row per contribution).
            Expected columns (where available):
            - employee_id: employee identifier (required)
            - pay_date: pay date (required, will be converted to datetime)
            - "EE Deferral $": regular pre-tax deferrals (defaults to 0 if missing)
            - "EE Roth $": regular Roth deferrals (defaults to 0 if missing)
            - "ER Match $": employer match amount (required for meaningful analysis)
            - Compensation column: tries "Eligibility Compensation", "Plan Compensation", "Compensation" in order
        output_dir: directory where CSV outputs should be written.
        plan_config: optional dict with match formula parameters. If None, uses defaults:
            - match_type: "percent_of_comp"
            - match_rate: 0.50 (50% match)
            - match_cap_pct: 0.06 (on first 6% of eligible comp)
            - absolute_tolerance: 5.00 ($5)
            - relative_tolerance_pct: 0.15 (15%)

    Returns:
        Tuple of (summary_dict, csv_path):
        - summary_dict contains:
          - total_rows: number of rows considered after basic cleaning
          - under_match_count: count of rows flagged as under-match
          - over_match_count: count of rows flagged as over-match
          - csv_path: string path to match issues CSV, or None if no issues
          - warning: optional warning message if analysis could not be performed
        - csv_path: Path object to match issues CSV, or None if no issues
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for required columns
        if payroll_df.empty:
            return (
                {
                    "total_rows": 0,
                    "under_match_count": 0,
                    "over_match_count": 0,
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
            warning_msg = f"Cannot perform match analysis because required columns are missing: {', '.join(missing_cols)}"
            print(f"[WARN] match_reasonableness_analyzer: {warning_msg}")
            return (
                {
                    "total_rows": 0,
                    "under_match_count": 0,
                    "over_match_count": 0,
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
            print(f"[WARN] match_reasonableness_analyzer: {warning_msg}")
            return (
                {
                    "total_rows": 0,
                    "under_match_count": 0,
                    "over_match_count": 0,
                    "csv_path": None,
                    "warning": warning_msg,
                },
                None,
            )
        
        if len(df) < original_len:
            print(f"[INFO] match_reasonableness_analyzer: Dropped {original_len - len(df)} rows with invalid pay_date")
        
        # Find compensation column (try in order of preference)
        comp_col = None
        for col_name in ["Eligibility Compensation", "Plan Compensation", "Compensation"]:
            if col_name in df.columns:
                comp_col = col_name
                break
        
        # Check for match column
        match_col = "ER Match $" if "ER Match $" in df.columns else None
        
        if comp_col is None or match_col is None:
            missing = []
            if comp_col is None:
                missing.append("compensation column (tried: Eligibility Compensation, Plan Compensation, Compensation)")
            if match_col is None:
                missing.append("ER Match $")
            warning_msg = f"Cannot perform match analysis because required columns are missing: {', '.join(missing)}"
            print(f"[WARN] match_reasonableness_analyzer: {warning_msg}")
            return (
                {
                    "total_rows": 0,
                    "under_match_count": 0,
                    "over_match_count": 0,
                    "csv_path": None,
                    "warning": warning_msg,
                },
                None,
            )
        
        # Set up plan config with defaults
        default_config = {
            "match_type": "percent_of_comp",
            "match_rate": 0.50,
            "match_cap_pct": 0.06,
            "absolute_tolerance": 5.00,
            "relative_tolerance_pct": 0.15,
        }
        
        if plan_config is None:
            plan_config = default_config
        else:
            # Update defaults with provided config
            config = default_config.copy()
            config.update(plan_config)
            plan_config = config
        
        # Extract match_rate and tolerances
        match_rate = plan_config.get("match_rate", 0.50)
        match_cap_pct = plan_config.get("match_cap_pct", 0.06)
        absolute_tolerance = plan_config.get("absolute_tolerance", 5.00)
        relative_tolerance_pct = plan_config.get("relative_tolerance_pct", 0.15)
        
        # Extract deferral amounts safely with defaults
        ee_deferral = df.get("EE Deferral $", pd.Series([0.0] * len(df), index=df.index))
        ee_roth = df.get("EE Roth $", pd.Series([0.0] * len(df), index=df.index))
        
        # Convert to numeric, filling invalid values with 0
        ee_deferral = pd.to_numeric(ee_deferral, errors="coerce").fillna(0.0)
        ee_roth = pd.to_numeric(ee_roth, errors="coerce").fillna(0.0)
        
        # Extract compensation and match amounts
        eligible_comp = pd.to_numeric(df[comp_col], errors="coerce").fillna(0.0)
        er_match_amount = pd.to_numeric(df[match_col], errors="coerce").fillna(0.0)
        
        # Compute per-row amounts
        df["ee_deferral_amount"] = ee_deferral + ee_roth
        df["eligible_comp"] = eligible_comp
        df["er_match_amount"] = er_match_amount
        
        # Filter to rows with eligible_comp > 0 (skip rows with no eligible compensation)
        df = df[df["eligible_comp"] > 0].copy()
        
        if len(df) == 0:
            warning_msg = "No rows with eligible compensation > 0 found"
            print(f"[INFO] match_reasonableness_analyzer: {warning_msg}")
            return (
                {
                    "total_rows": 0,
                    "under_match_count": 0,
                    "over_match_count": 0,
                    "csv_path": None,
                    "warning": warning_msg,
                },
                None,
            )
        
        # Compute match reasonableness metrics per row
        df["deferral_pct"] = df["ee_deferral_amount"] / df["eligible_comp"]
        # eligible_pct_for_match = min(deferral_pct, match_cap_pct)
        df["eligible_pct_for_match"] = df["deferral_pct"].clip(upper=match_cap_pct)
        df["expected_match"] = df["eligible_comp"] * df["eligible_pct_for_match"] * match_rate
        df["variance"] = df["er_match_amount"] - df["expected_match"]
        df["variance_abs"] = df["variance"].abs()
        df["variance_pct"] = df["variance_abs"] / df["expected_match"].replace(0.0, 1.0)  # Avoid division by zero
        
        # Determine if variance is within tolerance
        df["within_tolerance"] = (
            (df["variance_abs"] <= absolute_tolerance) & 
            (df["variance_pct"] <= relative_tolerance_pct)
        )
        
        # Flag issues
        df["issue_type"] = None
        df.loc[(~df["within_tolerance"]) & (df["er_match_amount"] < df["expected_match"]), "issue_type"] = "Under-match"
        df.loc[(~df["within_tolerance"]) & (df["er_match_amount"] > df["expected_match"]), "issue_type"] = "Over-match"
        
        # Filter to rows with issues
        issues = df[df["issue_type"].notna()].copy()
        
        # Prepare summary
        total_rows = len(df)
        under_match_count = len(issues[issues["issue_type"] == "Under-match"])
        over_match_count = len(issues[issues["issue_type"] == "Over-match"])
        
        # If no issues, return summary with csv_path = None
        if len(issues) == 0:
            return (
                {
                    "total_rows": total_rows,
                    "under_match_count": 0,
                    "over_match_count": 0,
                    "csv_path": None,
                },
                None,
            )
        
        # Get issue metadata from taxonomy for each issue type
        issues_list = []
        for _, row in issues.iterrows():
            issue_type = row["issue_type"]
            meta = get_issue_metadata(issue_type)
            
            issue_row = {
                "employee_id": row["employee_id"],
                "pay_date": row["pay_date"],
                "eligible_comp": row["eligible_comp"],
                "ee_deferral_amount": row["ee_deferral_amount"],
                "deferral_pct": row["deferral_pct"],
                "er_match_amount": row["er_match_amount"],
                "expected_match": row["expected_match"],
                "variance": row["variance"],
                "variance_pct": row["variance_pct"],
                "issue_type": issue_type,
                "issue_category": meta["issue_category"],
                "severity": meta["severity"],
                "correction_hint": meta["correction_hint"],
            }
            issues_list.append(issue_row)
        
        # Build DataFrame from issues
        issues_df = pd.DataFrame(issues_list)
        
        # Write CSV
        csv_path = output_dir / "match_issues.csv"
        issues_df.to_csv(csv_path, index=False)
        
        return (
            {
                "total_rows": total_rows,
                "under_match_count": under_match_count,
                "over_match_count": over_match_count,
                "csv_path": str(csv_path),
            },
            csv_path,
        )
    
    except Exception as e:
        warning_msg = f"Unexpected error in match_reasonableness_analyzer: {str(e)}"
        print(f"[WARN] match_reasonableness_analyzer: {warning_msg}")
        return (
            {
                "total_rows": 0,
                "under_match_count": 0,
                "over_match_count": 0,
                "csv_path": None,
                "warning": warning_msg,
            },
            None,
        )

