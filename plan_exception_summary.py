# plan_exception_summary.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


def _guess_category_from_source(source_key: str) -> str:
    """
    Guess issue category from source key.
    
    Args:
        source_key: Logical source identifier (e.g., "secure20", "eligibility", "comp_402g", "match")
        
    Returns:
        Default issue category string
    """
    category_map = {
        "secure20": "Secure 2.0",
        "eligibility": "Eligibility",
        "comp_402g": "Comp/402(g)",
        "match": "Match",
    }
    return category_map.get(source_key, "Other")


def _build_details_string(source_key: str, row: pd.Series) -> str:
    """
    Build a human-readable details string for an issue row based on source and available columns.
    
    Args:
        source_key: Logical source identifier
        row: Pandas Series representing one row from the source CSV
        
    Returns:
        Human-readable details string
    """
    issue_type = row.get("issue_type", "Unknown issue")
    
    if source_key == "secure20":
        # Secure 2.0 details
        deferral_pretax = row.get("deferral_pretax", 0.0)
        deferral_roth = row.get("deferral_roth", 0.0)
        catchup_amount = row.get("catchup_amount", 0.0)
        total_deferral = deferral_pretax + deferral_roth
        
        details_parts = [f"Secure 2.0 issue_type={issue_type}"]
        if total_deferral > 0:
            details_parts.append(f"total_deferral=${total_deferral:.2f}")
        if catchup_amount > 0:
            details_parts.append(f"catchup_amount=${catchup_amount:.2f}")
        return ", ".join(details_parts) + "."
    
    elif source_key == "eligibility":
        # Eligibility details
        computed_eligibility_date = row.get("computed_eligibility_date")
        first_deferral_date = row.get("first_deferral_date")
        drift_days = row.get("drift_days")
        
        details_parts = [f"Eligibility issue_type={issue_type}"]
        if computed_eligibility_date:
            details_parts.append(f"eligibility={computed_eligibility_date}")
        if first_deferral_date:
            details_parts.append(f"first_deferral={first_deferral_date}")
        if drift_days is not None and not pd.isna(drift_days):
            details_parts.append(f"drift_days={drift_days}")
        return ", ".join(details_parts) + "."
    
    elif source_key == "comp_402g":
        # 402(g) details
        total_deferrals = row.get("total_deferrals")
        allowed_limit = row.get("allowed_limit")
        excess_amount = row.get("excess_amount")
        
        details_parts = [f"402(g) issue_type={issue_type}"]
        if total_deferrals is not None and not pd.isna(total_deferrals):
            details_parts.append(f"total_deferrals=${total_deferrals:.2f}")
        if allowed_limit is not None and not pd.isna(allowed_limit):
            details_parts.append(f"allowed_limit=${allowed_limit:.2f}")
        if excess_amount is not None and not pd.isna(excess_amount) and excess_amount > 0:
            details_parts.append(f"excess_amount=${excess_amount:.2f}")
        return ", ".join(details_parts) + "."
    
    elif source_key == "match":
        # Match details
        eligible_comp = row.get("eligible_comp")
        expected_match = row.get("expected_match")
        er_match_amount = row.get("er_match_amount")
        variance = row.get("variance")
        
        details_parts = [f"Match issue_type={issue_type}"]
        if eligible_comp is not None and not pd.isna(eligible_comp):
            details_parts.append(f"eligible_comp=${eligible_comp:.2f}")
        if expected_match is not None and not pd.isna(expected_match):
            details_parts.append(f"expected_match=${expected_match:.2f}")
        if er_match_amount is not None and not pd.isna(er_match_amount):
            details_parts.append(f"er_match_amount=${er_match_amount:.2f}")
        if variance is not None and not pd.isna(variance):
            details_parts.append(f"variance=${variance:.2f}")
        return ", ".join(details_parts) + "."
    
    else:
        # Generic details for unknown sources
        return f"{source_key} issue_type={issue_type}."


def build_plan_exception_summary(
    output_dir: Path,
    run_id: str,
    plan_name: Optional[str],
    plan_year: Optional[int],
    issue_csv_paths: Dict[str, Optional[Union[str, Path]]],
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Aggregate per-analyzer issue CSVs into a single plan_exception_summary.csv.

    Args:
        output_dir: Directory for writing the aggregated CSV.
        run_id: Unique identifier for this engine run.
        plan_name: Optional plan name.
        plan_year: Optional plan year.
        issue_csv_paths: Mapping from a logical source key to a CSV path, e.g.:

            {
                "secure20": secure20_csv_path,
                "eligibility": eligibility_csv_path,
                "comp_402g": comp_402g_csv_path,
                "match": match_csv_path,
            }

            Values may be None if the corresponding analyzer did not produce a CSV.

    Returns:
        Tuple of (summary_dict, csv_path):
        - summary_dict contains:
          - total_issues: total number of issues aggregated
          - by_category: dict mapping issue_category to count
          - csv_path: string path to aggregated CSV, or None if no issues
          - warning: optional warning message if aggregation failed
        - csv_path: Path object to aggregated CSV, or None if no issues
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        normalized_rows: List[Dict[str, Any]] = []
        
        # Process each source CSV
        for source_key, csv_path in issue_csv_paths.items():
            # Skip if path is None
            if csv_path is None:
                continue
            
            # Normalize to Path instance
            csv_path_obj = Path(csv_path)
            
            # Check if file exists
            if not csv_path_obj.exists():
                print(f"[WARN] plan_exception_summary: Source CSV not found for {source_key}: {csv_path_obj}")
                continue
            
            # Read CSV
            try:
                df_source = pd.read_csv(csv_path_obj)
            except Exception as e:
                print(f"[WARN] plan_exception_summary: Failed to read CSV for {source_key} at {csv_path_obj}: {e}")
                continue
            
            # Skip if empty
            if df_source.empty:
                continue
            
            # Process each row
            for _, row in df_source.iterrows():
                # Extract values with defaults
                employee_id = row.get("employee_id")
                if pd.isna(employee_id):
                    employee_id = None
                
                issue_type = row.get("issue_type")
                if pd.isna(issue_type) or issue_type is None:
                    issue_type = source_key
                else:
                    issue_type = str(issue_type)
                
                issue_category = row.get("issue_category")
                if pd.isna(issue_category) or issue_category is None:
                    issue_category = _guess_category_from_source(source_key)
                else:
                    issue_category = str(issue_category)
                
                severity = row.get("severity")
                if pd.isna(severity) or severity is None:
                    severity = "Medium"
                else:
                    severity = str(severity)
                
                correction_hint = row.get("correction_hint")
                if pd.isna(correction_hint):
                    correction_hint = None
                else:
                    correction_hint = str(correction_hint)
                
                # Build details string
                details = _build_details_string(source_key, row)
                
                # Build normalized row
                normalized_row = {
                    "run_id": run_id,
                    "plan_name": plan_name,
                    "plan_year": plan_year,
                    "employee_id": employee_id,
                    "issue_category": issue_category,
                    "issue_type": issue_type,
                    "severity": severity,
                    "source": source_key,
                    "source_csv_path": str(csv_path_obj),
                    "details": details,
                    "correction_hint": correction_hint,
                }
                
                normalized_rows.append(normalized_row)
        
        # If no issues, return empty summary
        if len(normalized_rows) == 0:
            return (
                {
                    "total_issues": 0,
                    "by_category": {},
                    "csv_path": None,
                },
                None,
            )
        
        # Build DataFrame from normalized rows
        df_summary = pd.DataFrame(normalized_rows)
        
        # Compute summary statistics
        total_issues = len(df_summary)
        by_category = df_summary["issue_category"].value_counts().to_dict()
        
        # Write CSV
        csv_path = output_dir / "plan_exception_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        
        return (
            {
                "total_issues": total_issues,
                "by_category": by_category,
                "csv_path": str(csv_path),
            },
            csv_path,
        )
    
    except Exception as e:
        warning_msg = f"Plan exception summary failed: {e}"
        print(f"[WARN] plan_exception_summary: {warning_msg}")
        return (
            {
                "total_issues": 0,
                "by_category": {},
                "csv_path": None,
                "warning": warning_msg,
            },
            None,
        )

