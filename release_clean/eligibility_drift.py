"""
Eligibility Drift Detection v1

Detects contribution rows that occur after employee termination.
"""

import pandas as pd


def detect_eligibility_drift(
    df_payroll: pd.DataFrame,
    grace_days: int = 0,
) -> pd.DataFrame:
    """
    Detects contribution rows AFTER termination.
    
    Requires columns: employee_id, pay_date, def_amount, roth_amount,
                      employment_status, termination_date.
    Returns a DataFrame with drift rows + helper columns.
    If required columns are missing, return an empty DataFrame.
    
    Args:
        df_payroll: Payroll dataframe with normalized columns
        grace_days: Number of grace days after termination to allow (default: 0)
    
    Returns:
        DataFrame with drift rows containing:
        - employee_id
        - employment_status
        - termination_date
        - pay_date
        - days_after_termination
        - def_amount
        - roth_amount
        - total_contribution
        
        Returns empty DataFrame if required columns are missing.
    """
    # Required columns
    required_cols = [
        "employee_id",
        "pay_date",
        "def_amount",
        "roth_amount",
        "employment_status",
        "termination_date",
    ]
    
    # Check if all required columns are present
    missing_cols = [col for col in required_cols if col not in df_payroll.columns]
    if missing_cols:
        # Gracefully no-op if required columns are missing
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the input
    df = df_payroll.copy()
    
    # Coerce dates to datetime with errors="coerce"
    df["pay_date"] = pd.to_datetime(df["pay_date"], errors="coerce")
    df["termination_date"] = pd.to_datetime(df["termination_date"], errors="coerce")
    
    # Compute total_contribution = def_amount + roth_amount
    # Handle missing values by treating as 0.0
    def_amount = pd.to_numeric(df["def_amount"], errors="coerce").fillna(0.0)
    roth_amount = pd.to_numeric(df["roth_amount"], errors="coerce").fillna(0.0)
    df["total_contribution"] = def_amount + roth_amount
    
    # Compute days_after_termination = (pay_date - termination_date).dt.days
    df["days_after_termination"] = (df["pay_date"] - df["termination_date"]).dt.days
    
    # Filter where:
    # - termination_date.notna() (has a termination date)
    # - days_after_termination > grace_days (contribution after grace period)
    # - total_contribution > 0 (actual contribution amount)
    mask = (
        df["termination_date"].notna()
        & (df["days_after_termination"] > grace_days)
        & (df["total_contribution"] > 0)
    )
    
    drift_df = df[mask].copy()
    
    # Return only the specified columns
    output_cols = [
        "employee_id",
        "employment_status",
        "termination_date",
        "pay_date",
        "days_after_termination",
        "def_amount",
        "roth_amount",
        "total_contribution",
    ]
    
    # Only select columns that exist (in case some are missing)
    available_cols = [col for col in output_cols if col in drift_df.columns]
    if available_cols:
        return drift_df[available_cols]
    else:
        return pd.DataFrame()

