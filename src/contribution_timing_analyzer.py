import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from datetime import datetime


# =========================
# Config – adjust as needed
# =========================

BUSINESS_DAY_THRESHOLD = 5  # > 5 business days = late


def load_payroll(path: Path) -> pd.DataFrame:
    """
    Load and normalize payroll data for ADP-style synthetic file.

    Raw columns expected:
      - EmpNumber
      - Payroll_Run_Date
      - PreTax_Defl
      - Roth_Defl
      - Loan_Pmt
    """
    df = pd.read_csv(path)

    # Optional debug so you can see columns while you're learning
    

    df = df.rename(
        columns={
            "EmpNumber": "employee_id",
            "Payroll_Run_Date": "pay_date",
            "PreTax_Defl": "payroll_pretax",
            "Roth_Defl": "payroll_roth",
            "Loan_Pmt": "payroll_loan",
        }
    )

    # Parse dates
    df["pay_date"] = pd.to_datetime(df["pay_date"])

    # Compute total deferral for reconciliation
    df["payroll_total_deferral"] = (
        df[["payroll_pretax", "payroll_roth"]].fillna(0).sum(axis=1)
    )

    return df




def load_recordkeeper(path: Path) -> pd.DataFrame:
    """
    Load and normalize recordkeeper data.

    You need to align these names to whatever you actually have in your
    synthetic RK file. These are example placeholders:

      - 'EmpNumber' or 'Part_ID'  -> employee_id
      - 'Post_Date'               -> deposit_date
      - 'EE_PreTax'               -> rk_pretax
      - 'EE_Roth'                 -> rk_roth
      - 'Loan_Contr'              -> rk_loan
    """
    df = pd.read_csv(path)

    # TODO: update these mappings to match your actual RK CSV headers
    df = df.rename(
        columns={
            "EmpNumber": "employee_id",
            "Part_ID": "employee_id",          # if you have Part_ID instead
            "Post_Date": "deposit_date",
            "EE_PreTax": "rk_pretax",
            "EE_Roth": "rk_roth",
            "Loan_Contr": "rk_loan",
        }
    )

    # If both EmpNumber and Part_ID exist, prefer one
    if "EmpNumber" in df.columns and "employee_id" not in df.columns:
        df["employee_id"] = df["EmpNumber"]

    df["deposit_date"] = pd.to_datetime(df["deposit_date"])

    df["rk_total_deferral"] = df[["rk_pretax", "rk_roth"]].fillna(0).sum(axis=1)

    return df


from datetime import datetime, timedelta

def business_days_between(start: datetime, end: datetime) -> int:
    """
    Rough business day count between two dates (Mon-Fri, ignores holidays).
    If end < start, returns 0.
    """
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0

    start_date = start.date()
    end_date = end.date()

    day_count = 0
    current = start_date
    while current < end_date:
        if current.weekday() < 5:  # Monday=0, Sunday=6
            day_count += 1
        current += timedelta(days=1)

    return day_count



def join_payroll_rk(
    payroll: pd.DataFrame, rk: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join payroll and RK on employee + date proxies and return:
      - merged DataFrame
      - rows only in payroll
      - rows only in rk
    """

    # For now, we assume we match on employee_id + pay_date ≈ deposit_date
    # In real life you'd have more robust mapping (pay period IDs, etc.)
    payroll_key = payroll[["employee_id", "pay_date", "payroll_total_deferral", "payroll_loan"]].copy()
    rk_key = rk[["employee_id", "deposit_date", "rk_total_deferral", "rk_loan"]].copy()

    merged = payroll_key.merge(
        rk_key,
        on="employee_id",
        how="outer",
        suffixes=("_payroll", "_rk"),
    )

    # Simple detection of "only on one side" based on missing totals
    only_payroll = merged[merged["rk_total_deferral"].isna()].copy()
    only_rk = merged[merged["payroll_total_deferral"].isna()].copy()

    return merged, only_payroll, only_rk


def analyze_contributions(merged: pd.DataFrame) -> dict:
    """
    Compute high-level contribution reconciliation metrics.
    """

    # Fill NaNs with 0 so math works
    merged["payroll_total_deferral"] = merged["payroll_total_deferral"].fillna(0)
    merged["rk_total_deferral"] = merged["rk_total_deferral"].fillna(0)

    total_payroll_deferrals = merged["payroll_total_deferral"].sum()
    total_rk_deferrals = merged["rk_total_deferral"].sum()

    # Amount mismatches where both sides exist but differ
    both_sides = merged[
        (merged["payroll_total_deferral"] > 0) & (merged["rk_total_deferral"] > 0)
    ].copy()
    both_sides["deferral_diff"] = both_sides["rk_total_deferral"] - both_sides["payroll_total_deferral"]
    amount_mismatch_rows = both_sides[both_sides["deferral_diff"].round(2) != 0]

    summary = {
        "total_payroll_deferrals": float(total_payroll_deferrals),
        "total_rk_deferrals": float(total_rk_deferrals),
        "net_deferral_diff": float(total_rk_deferrals - total_payroll_deferrals),
        "num_amount_mismatches": int(len(amount_mismatch_rows)),
        "amount_mismatches_total_abs": float(amount_mismatch_rows["deferral_diff"].abs().sum()),
    }

    return summary


def analyze_late_contributions(merged: pd.DataFrame) -> dict:
    """
    Flag late contributions based on business days between pay_date and deposit_date.
    """

    merged = merged.copy()
    merged["days_to_deposit"] = merged.apply(
        lambda row: business_days_between(row.get("pay_date"), row.get("deposit_date")),
        axis=1,
    )

    late_rows = merged[merged["days_to_deposit"] > BUSINESS_DAY_THRESHOLD]

    return {
        "num_late_contributions": int(len(late_rows)),
        "max_days_to_deposit": int(late_rows["days_to_deposit"].max()) if len(late_rows) > 0 else 0,
        "late_rows_sample": late_rows.head(5),
    }


def print_summary(
    summary: dict,
    late_summary: dict,
    only_payroll: pd.DataFrame,
    only_rk: pd.DataFrame,
) -> None:
    """
    Print a clean, audit-style summary to stdout.
    """

    print("\n==============================")
    print("  CONTRIBUTION TIMING SUMMARY ")
    print("==============================\n")

    print("Deferral Totals")
    print("---------------")
    print(f"Total payroll deferrals:       {summary['total_payroll_deferrals']:,.2f}")
    print(f"Total RK posted deferrals:     {summary['total_rk_deferrals']:,.2f}")
    print(f"Net difference (RK - payroll): {summary['net_deferral_diff']:,.2f}")
    print()

    print("Reconciliation Exceptions")
    print("-------------------------")
    print(f"Rows only in payroll:          {len(only_payroll)}")
    print(f"Rows only in RK:               {len(only_rk)}")
    print(f"Amount mismatches (rows):      {summary['num_amount_mismatches']}")
    print(f"Amount mismatches (abs total): {summary['amount_mismatches_total_abs']:,.2f}")
    print()

    print("Contribution Timing")
    print("-------------------")
    print(f"Late contributions (> {BUSINESS_DAY_THRESHOLD} business days): {late_summary['num_late_contributions']}")
    print(f"Max days to deposit (late set):                               {late_summary['max_days_to_deposit']}")
    print()

    if late_summary["num_late_contributions"] > 0:
        print("Sample late contributions (up to 5 rows):")
        print(late_summary["late_rows_sample"][["employee_id", "pay_date", "deposit_date", "days_to_deposit"]])
        print()

    print("Note: This is a synthetic analysis for engine development, not production output.")


def main(payroll_path: str, rk_path: str) -> None:
    payroll_file = Path(payroll_path)
    rk_file = Path(rk_path)

    if not payroll_file.exists():
        raise FileNotFoundError(f"Payroll file not found: {payroll_file}")
    if not rk_file.exists():
        raise FileNotFoundError(f"Recordkeeper file not found: {rk_file}")

    payroll = load_payroll(payroll_file)
    rk = load_recordkeeper(rk_file)

    merged, only_payroll, only_rk = join_payroll_rk(payroll, rk)

    summary = analyze_contributions(merged)
    late_summary = analyze_late_contributions(merged)

    print_summary(summary, late_summary, only_payroll, only_rk)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python contribution_timing_analyzer.py <payroll_csv> <recordkeeper_csv>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
