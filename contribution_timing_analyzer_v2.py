import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd

# Import normalization from main.py
from main import normalize_column_names


def classify_timing_risk(
    total_rows: int,
    num_late: int,
    num_missing: int,
    late_threshold_days: int,
    max_days_late: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """
    Classify overall timing risk for Secure 2.0 purposes.

    This is:
      - Vendor-agnostic: only uses counts/thresholds, not vendor-specific fields.
      - Audit-friendly: returns explicit drivers explaining *why* a level was chosen.

    Returns:
      timing_risk: "High", "Medium", "Low", or "N/A"
      drivers: list of human-readable reasons for the classification
    """
    drivers: List[str] = []

    # Guardrail: no data
    if total_rows == 0:
        timing_risk = "N/A"
        drivers.append(
            "No payroll rows were analyzed. Check input files, column mapping, "
            "and filtered rows (e.g., all-zero deferrals)."
        )
        return timing_risk, drivers

    # 1) Missing deposits are always High risk
    if num_missing > 0:
        timing_risk = "High"
        drivers.append(
            f"{num_missing} payroll row(s) have missing deposits "
            "(contributions present in payroll but not in recordkeeper data)."
        )
        drivers.append(
            "Missing deposits must be resolved before the plan can be considered "
            "compliant with timely deposit requirements."
        )
        return timing_risk, drivers

    # 2) No late rows and no missing deposits → Low risk
    if num_late == 0:
        timing_risk = "Low"
        drivers.append(
            f"No late contributions detected above the {late_threshold_days}-day "
            "Secure 2.0 threshold."
        )
        drivers.append("No missing deposits detected between payroll and recordkeeper.")
        return timing_risk, drivers

    # 3) Late rows present, but no missing deposits → Medium or High
    late_ratio = num_late / total_rows
    drivers.append(
        f"{num_late} late contribution row(s) detected above the "
        f"{late_threshold_days}-day threshold out of {total_rows} payroll row(s)."
    )
    drivers.append(f"Late row percentage: {late_ratio:.2%}.")

    if max_days_late is not None:
        drivers.append(f"Maximum days late: {max_days_late} day(s).")

        # Medium: slightly late + small footprint
        if max_days_late <= late_threshold_days + 3 and late_ratio < 0.05:
            timing_risk = "Medium"
            drivers.append(
                "Delays are only slightly above the threshold and impact fewer than "
                "5% of rows. Escalated review recommended, but not systemic."
            )
        else:
            timing_risk = "High"
            if max_days_late > late_threshold_days + 10:
                drivers.append(
                    "Some contributions are significantly late (more than 10 days "
                    "beyond the threshold)."
                )
            if late_ratio >= 0.05:
                drivers.append(
                    "Late contributions affect 5% or more of payroll rows, indicating "
                    "a potential systemic timing issue."
                )
    else:
        # Fallback if we don't have day-level lag; use ratio only
        if late_ratio < 0.05:
            timing_risk = "Medium"
            drivers.append(
                "Exact days-late data is not available, but fewer than 5% of rows "
                "are late. Treated as Medium risk."
            )
        else:
            timing_risk = "High"
            drivers.append(
                "Exact days-late data is not available and 5% or more of rows are "
                "late. Treated as High risk."
            )

    return timing_risk, drivers


# ==============================
# File loading
# ==============================

def load_table(path: Path) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    return df


# ==============================
# Vendor detection
# ==============================

def detect_payroll_vendor(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}

    # ADP-style heuristic
    if {"emp id", "check date", "ee deferral $"}.issubset(cols):
        return "ADP"

    # Our synthetic payroll schema
    if {"empnumber", "payroll_run_date", "pretax_defl"}.issubset(cols):
        return "GenericPayroll"

    # 457(b) / municipal-style payroll schema (City of Lakes example)
    if {"employee_id", "payroll_date", "457b_ee_pretax_amt", "457b_ee_roth_amt"}.issubset(
        cols
    ):
        return "City457Payroll"

    return "UnknownPayroll"


def detect_rk_vendor(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}

    # Empower-style heuristic
    if {"part_id", "post_date", "ee_pretax"}.issubset(cols):
        return "Empower"

    # Our synthetic RK schema
    if {"part_id", "post_date", "ee_pretax", "ee_roth", "loan_contr"}.issubset(cols):
        return "GenericRK"

    # 457(b) RK export schema (City of Lakes)
    if {"plan_id", "recordkeeper_client_id", "money_type", "contribution_amount"}.issubset(
        cols
    ):
        return "City457RK"

    return "UnknownRK"


# ==============================
# Normalization / mapping
# ==============================

def normalize_payroll(df: pd.DataFrame, vendor: str, vendor_confidence: float = 0.0) -> pd.DataFrame:
    """
    Normalize payroll dataframe to standard column names.
    
    For unknown/low-confidence vendors, uses generic fallback that expects
    columns to already be normalized by normalize_column_names().
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    
    # Normalize vendor string for comparison
    vendor_str = str(vendor).strip() if vendor else None
    
    # Check if this is a generic/fallback case
    # Treat as generic if: None, "Unknown", "UnknownPayroll", or low confidence
    is_generic = (
        vendor_str is None
        or vendor_str in ("Unknown", "UnknownPayroll", "")
        or vendor_confidence < 0.65
    )
    
    # Known vendors that require strict schema (only if confidence >= 0.65)
    known_vendors = {"ADP", "GenericPayroll", "City457Payroll"}
    is_known_high_confidence = (vendor_str in known_vendors and vendor_confidence >= 0.65)
    
    if not is_known_high_confidence:
        # Generic fallback path - do NOT enforce vendor-specific columns
        # Generic fallback: assume columns already normalized by normalize_column_names()
        # Use standard column names: "EE Deferral $", "EE Roth $", loan_amount
        out = pd.DataFrame()
        
        # employee_id - try common variants
        emp_id_col = None
        for col_name in ["employee_id", "emp_id", "employee id", "emp id", "employee number"]:
            if col_name in df.columns:
                emp_id_col = col_name
                break
            # Case-insensitive search
            for col in df.columns:
                if col.lower().strip() == col_name.lower():
                    emp_id_col = col
                    break
            if emp_id_col:
                break
        
        if emp_id_col:
            out["employee_id"] = df[emp_id_col]
        else:
            print("[WARN] employee_id column not found. Analysis may fail.")
            # Try to use index or first column as fallback
            if len(df) > 0:
                out["employee_id"] = df.index if hasattr(df.index, 'values') else range(len(df))
        
        # pay_date - try common variants
        pay_date_col = None
        for col_name in ["pay_date", "pay date", "payroll_date", "payroll date", "check_date", "check date"]:
            if col_name in df.columns:
                pay_date_col = col_name
                break
            # Case-insensitive search
            for col in df.columns:
                if col.lower().strip() == col_name.lower():
                    pay_date_col = col
                    break
            if pay_date_col:
                break
        
        if pay_date_col:
            out["pay_date"] = pd.to_datetime(df[pay_date_col], errors="coerce")
        else:
            print("[WARN] pay_date column not found. Late contribution detection may be limited.")
            out["pay_date"] = pd.NaT
        
        # Use normalized column names from normalize_column_names()
        if "EE Deferral $" in df.columns:
            out["payroll_pretax"] = pd.to_numeric(df["EE Deferral $"], errors="coerce").fillna(0.0)
        else:
            print("[WARN] 'EE Deferral $' column not found after normalization. Using 0.0.")
            out["payroll_pretax"] = 0.0
        
        if "EE Roth $" in df.columns:
            out["payroll_roth"] = pd.to_numeric(df["EE Roth $"], errors="coerce").fillna(0.0)
        else:
            print("[WARN] 'EE Roth $' column not found after normalization. Using 0.0.")
            out["payroll_roth"] = 0.0
        
        if "loan_amount" in df.columns:
            out["payroll_loan"] = pd.to_numeric(df["loan_amount"], errors="coerce").fillna(0.0)
        else:
            print("[WARN] 'loan_amount' column not found. Using 0.0.")
            out["payroll_loan"] = 0.0
        
        # Normalize employee_id to string dtype for consistent merging
        if "employee_id" in out.columns:
            out["employee_id"] = out["employee_id"].astype(str).str.strip()
        
        return out
    
    # Only reach here if we have a known vendor with high confidence
    # Enforce strict requirements for known vendors
    if vendor_str == "ADP":
        mapping = {
            "employee_id": "Emp ID",
            "pay_date": "Check Date",
            "payroll_pretax": "EE Deferral $",
            "payroll_roth": "EE Roth $",
            "payroll_loan": "Loan Repay $",
        }

    elif vendor_str == "GenericPayroll":
        mapping = {
            "employee_id": "EmpNumber",
            "pay_date": "Payroll_Run_Date",
            "payroll_pretax": "PreTax_Defl",
            "payroll_roth": "Roth_Defl",
            "payroll_loan": "Loan_Pmt",
        }

    elif vendor_str == "City457Payroll":
        # City of Lakes 457(b) payroll schema
        mapping = {
            "employee_id": cols_lower["employee_id"],           # Employee_ID
            "pay_date": cols_lower["payroll_date"],             # Payroll_Date
            "payroll_pretax": cols_lower["457b_ee_pretax_amt"], # 457b_EE_Pretax_Amt
            "payroll_roth": cols_lower["457b_ee_roth_amt"],     # 457b_EE_Roth_Amt
            "payroll_loan": cols_lower["457b_loan_repay_amt"],  # 457b_Loan_Repay_Amt
        }

    else:
        # Should not reach here due to is_known_high_confidence check, but handle gracefully
        print(f"[WARN] Unexpected vendor '{vendor_str}' with high confidence. Using generic fallback.")
        return normalize_payroll(df, "UnknownPayroll", 0.0)

    # For known vendors, enforce strict requirements
    out = pd.DataFrame()
    for target, source in mapping.items():
        if source not in df.columns:
            raise KeyError(
                f"Expected payroll column '{source}' not found for vendor '{vendor}'"
            )
        out[target] = df[source]

    for col in ["payroll_pretax", "payroll_roth", "payroll_loan"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["pay_date"] = pd.to_datetime(out["pay_date"], errors="coerce")
    
    # Normalize employee_id to string dtype for consistent merging
    if "employee_id" in out.columns:
        out["employee_id"] = out["employee_id"].astype(str).str.strip()

    return out


def normalize_rk(df: pd.DataFrame, vendor: str, vendor_confidence: float = 0.0) -> pd.DataFrame:
    """
    Normalize recordkeeper dataframe to standard column names.
    
    For unknown/low-confidence vendors, uses generic fallback that expects
    columns to already be normalized by normalize_column_names().
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    
    # Normalize vendor string for comparison
    vendor_str = str(vendor).strip() if vendor else None
    
    # Check if this is a generic/fallback case
    # Treat as generic if: None, "Unknown", "UnknownRK", or low confidence
    is_generic = (
        vendor_str is None
        or vendor_str in ("Unknown", "UnknownRK", "")
        or vendor_confidence < 0.65
    )
    
    # Known vendors that require strict schema (only if confidence >= 0.65)
    known_vendors = {"Empower", "GenericRK", "City457RK"}
    is_known_high_confidence = (vendor_str in known_vendors and vendor_confidence >= 0.65)
    
    if not is_known_high_confidence:
        # Generic fallback path - do NOT enforce vendor-specific columns
        # Generic fallback: assume columns already normalized by normalize_column_names()
        out = pd.DataFrame()
        
        # employee_id - try common variants
        emp_id_col = None
        for col_name in ["employee_id", "emp_id", "employee id", "emp id", "part_id", "participant_id", "participant id"]:
            if col_name in df.columns:
                emp_id_col = col_name
                break
            # Case-insensitive search
            for col in df.columns:
                if col.lower().strip() == col_name.lower():
                    emp_id_col = col
                    break
            if emp_id_col:
                break
        
        if emp_id_col:
            out["employee_id"] = df[emp_id_col]
        else:
            print("[WARN] employee_id column not found in RK. Analysis may fail.")
            if len(df) > 0:
                out["employee_id"] = df.index if hasattr(df.index, 'values') else range(len(df))
        
        # deposit_date - try common variants
        deposit_date_col = None
        for col_name in ["deposit_date", "deposit date", "post_date", "post date", "transaction_date", "transaction date"]:
            if col_name in df.columns:
                deposit_date_col = col_name
                break
            # Case-insensitive search
            for col in df.columns:
                if col.lower().strip() == col_name.lower():
                    deposit_date_col = col
                    break
            if deposit_date_col:
                break
        
        if deposit_date_col:
            out["deposit_date"] = pd.to_datetime(df[deposit_date_col], errors="coerce")
        else:
            print("[WARN] deposit_date column not found in RK. Late contribution detection may be limited.")
            out["deposit_date"] = pd.NaT
        
        # Use normalized column names from normalize_column_names()
        if "EE Deferral $" in df.columns:
            out["rk_pretax"] = pd.to_numeric(df["EE Deferral $"], errors="coerce").fillna(0.0)
        else:
            print("[WARN] 'EE Deferral $' column not found in RK after normalization. Using 0.0.")
            out["rk_pretax"] = 0.0
        
        if "EE Roth $" in df.columns:
            out["rk_roth"] = pd.to_numeric(df["EE Roth $"], errors="coerce").fillna(0.0)
        else:
            print("[WARN] 'EE Roth $' column not found in RK after normalization. Using 0.0.")
            out["rk_roth"] = 0.0
        
        if "loan_amount" in df.columns:
            out["rk_loan"] = pd.to_numeric(df["loan_amount"], errors="coerce").fillna(0.0)
        else:
            print("[WARN] 'loan_amount' column not found in RK. Using 0.0.")
            out["rk_loan"] = 0.0
        
        # Normalize employee_id to string dtype for consistent merging
        if "employee_id" in out.columns:
            out["employee_id"] = out["employee_id"].astype(str).str.strip()
        
        return out

    # Only reach here if we have a known vendor with high confidence
    # Enforce strict requirements for known vendors
    if vendor_str in ("Empower", "GenericRK"):
        mapping = {
            "employee_id": "Part_ID",
            "deposit_date": "Post_Date",
            "rk_pretax": "EE_PreTax",
            "rk_roth": "EE_Roth",
            "rk_loan": "Loan_Contr",
        }

        out = pd.DataFrame()
        for target, source in mapping.items():
            if source not in df.columns:
                raise KeyError(
                    f"Expected RK column '{source}' not found for vendor '{vendor}'"
                )
            out[target] = df[source]

        for col in ["rk_pretax", "rk_roth", "rk_loan"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        out["deposit_date"] = pd.to_datetime(out["deposit_date"], errors="coerce")
        
        # Normalize employee_id to string dtype for consistent merging
        if "employee_id" in out.columns:
            out["employee_id"] = out["employee_id"].astype(str).str.strip()
        
        return out

    elif vendor_str == "City457RK":
        # City of Lakes 457(b) RK schema
        emp_col = cols_lower["employee_id"]  # Employee_ID
        date_col = cols_lower.get(
            "transaction_effective_date", "Transaction_Effective_Date"
        )
        money_type_col = cols_lower["money_type"]         # Money_Type
        amt_col = cols_lower["contribution_amount"]       # Contribution_Amount

        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

        grouped = (
            df.groupby([emp_col, date_col, money_type_col], as_index=False)[amt_col]
            .sum()
        )

        pivot = (
            grouped.pivot(
                index=[emp_col, date_col],
                columns=money_type_col,
                values=amt_col,
            )
            .fillna(0.0)
        ).reset_index()

        out = pd.DataFrame()
        out["employee_id"] = pivot[emp_col]
        out["deposit_date"] = pd.to_datetime(pivot[date_col], errors="coerce")

        pretax_series = pivot.get("457b_Pretax", 0.0)
        roth_series = pivot.get("457b_Roth", 0.0)

        out["rk_pretax"] = pd.to_numeric(pretax_series, errors="coerce").fillna(0.0)
        out["rk_roth"] = pd.to_numeric(roth_series, errors="coerce").fillna(0.0)
        out["rk_loan"] = 0.0  # no explicit loan field in this RK schema
        
        # Normalize employee_id to string dtype for consistent merging
        if "employee_id" in out.columns:
            out["employee_id"] = out["employee_id"].astype(str).str.strip()

        return out

    else:
        # Should not reach here due to is_known_high_confidence check, but handle gracefully
        print(f"[WARN] Unexpected RK vendor '{vendor_str}' with high confidence. Using generic fallback.")
        return normalize_rk(df, "UnknownRK", 0.0)


# ==============================
# Timing / late-contribution logic
# ==============================

def compute_timing_risk(total_rows: int, late_rows: int) -> str:
    """
    Return 'Low', 'Medium', or 'High' timing risk based on the late contribution rate.
    
    - Low: late_rate <= 0.05 (5% or less)
    - Medium: 0.05 < late_rate <= 0.15 (5-15%)
    - High: late_rate > 0.15 (more than 15%)
    
    If total_rows == 0, return 'N/A'.
    """
    if total_rows == 0:
        return "N/A"
    
    late_rate = late_rows / total_rows
    
    if late_rate <= 0.05:
        return "Low"
    elif late_rate <= 0.15:
        return "Medium"
    else:
        return "High"


def business_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    delta_days = (end.dt.normalize() - start.dt.normalize()).dt.days
    delta_days = delta_days.clip(lower=0)
    return delta_days


def compute_late_contributions(payroll_df: pd.DataFrame,
                               rk_df: pd.DataFrame,
                               late_threshold_days: int = 5) -> pd.DataFrame:
    payroll_df = payroll_df.copy()
    rk_df = rk_df.copy()

    payroll_df["pay_total"] = (
        payroll_df["payroll_pretax"]
        + payroll_df["payroll_roth"]
        + payroll_df["payroll_loan"]
    )

    rk_df["rk_total"] = (
        rk_df["rk_pretax"]
        + rk_df["rk_roth"]
        + rk_df["rk_loan"]
    )

    # Defensive: ensure employee_id is normalized before merge
    if "employee_id" in payroll_df.columns:
        payroll_df["employee_id"] = payroll_df["employee_id"].astype(str).str.strip()
    if "employee_id" in rk_df.columns:
        rk_df["employee_id"] = rk_df["employee_id"].astype(str).str.strip()

    merged = payroll_df.merge(
        rk_df,
        on=["employee_id"],
        how="left",
        suffixes=("_payroll", "_rk"),
    )

    merged["days_to_deposit"] = business_days_between(
        merged["pay_date"],
        merged["deposit_date"],
    )

    merged["is_late"] = merged["days_to_deposit"] > late_threshold_days
    merged["missing_deposit"] = merged["deposit_date"].isna()

    return merged


# ==============================
# CLI orchestration
# ==============================

def run_timing_analysis(payroll_path: Path,
                        rk_path: Path,
                        output_dir: Path,
                        late_threshold_days: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_payroll = load_table(payroll_path)
    raw_rk = load_table(rk_path)
    
    # Normalize column names first (handles flexible deferral/roth variants)
    raw_payroll = normalize_column_names(raw_payroll)
    raw_rk = normalize_column_names(raw_rk)

    payroll_vendor = detect_payroll_vendor(raw_payroll)
    rk_vendor = detect_rk_vendor(raw_rk)
    
    # Default confidence to 0.0 for CLI (no confidence scoring in timing analyzer)
    payroll_confidence = 0.0
    rk_confidence = 0.0

    print("=== Vendor Detection ===")
    print(f"Detected payroll vendor:     {payroll_vendor}")
    print(f"Detected recordkeeper:       {rk_vendor}")
    print()

    payroll = normalize_payroll(raw_payroll, payroll_vendor, payroll_confidence)
    rk = normalize_rk(raw_rk, rk_vendor, rk_confidence)

    # Normalize join key types to avoid pandas merge dtype errors
    if "employee_id" in payroll.columns:
        payroll["employee_id"] = (
            payroll["employee_id"]
            .astype(str)
            .str.strip()
        )
    if "employee_id" in rk.columns:
        rk["employee_id"] = (
            rk["employee_id"]
            .astype(str)
            .str.strip()
        )

    result = compute_late_contributions(
        payroll_df=payroll,
        rk_df=rk,
        late_threshold_days=late_threshold_days,
    )

    late_rows = result[(result["is_late"]) | (result["missing_deposit"])].copy()

    late_path = output_dir / "late_contributions.csv"
    late_rows.to_csv(late_path, index=False)

    # Compute core counts for timing risk
    # result is the merged DataFrame from compute_late_contributions (payroll_df_normalized equivalent)
    total_rows = len(result)
    late_contributions_df = result[result["is_late"]].copy() if "is_late" in result.columns else None
    missing_deposits_df = result[result["missing_deposit"]].copy() if "missing_deposit" in result.columns else None
    
    num_late = len(late_contributions_df) if late_contributions_df is not None else 0
    # Compute num_missing directly from result DataFrame to ensure accuracy
    if "missing_deposit" in result.columns:
        num_missing = int(result["missing_deposit"].sum())
    else:
        num_missing = len(missing_deposits_df) if missing_deposits_df is not None else 0

    # Hard business rule:
    # - Any missing deposits -> High risk
    # - Otherwise, any late rows -> Medium risk (for now)
    # - Otherwise -> Low risk
    if num_missing > 0:
        timing_risk = "High"
    elif num_late > 0:
        timing_risk = "Medium"
    else:
        timing_risk = "Low"

    print()
    print("==============================")
    print("  CONTRIBUTION TIMING SUMMARY")
    print("==============================")
    print()
    print(f"Total payroll rows analyzed:  {total_rows}")
    print(f"Late contributions (> {late_threshold_days} days): {num_late}")
    print(f"Missing deposit rows:         {num_missing}")
    print(f"Timing Risk:                  {timing_risk}")
    print()
    print(f"Late-contribution detail CSV written to: {late_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProofLink Contribution Timing Analyzer (v2 with vendor auto-detection)"
    )

    parser.add_argument(
        "payroll",
        nargs="?",
        help="Path to payroll file (CSV or Excel). If omitted, uses default synthetic payroll.",
    )
    parser.add_argument(
        "recordkeeper",
        nargs="?",
        help="Path to recordkeeper file (CSV or Excel). If omitted, uses default synthetic RK.",
    )
    parser.add_argument(
        "--late-threshold",
        type=int,
        default=5,
        help="Late threshold in days (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "processed"),
        help="Directory for output CSV (default: ../data/processed).",
    )

    return parser.parse_args()


def resolve_default_paths(payroll_arg: str | None,
                          rk_arg: str | None) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    default_payroll = raw_dir / "payroll_adp_synthetic_400.csv"
    default_rk = raw_dir / "rk_empower_synthetic_400.csv"

    if payroll_arg is None and rk_arg is None:
        return default_payroll, default_rk
    elif payroll_arg is not None and rk_arg is None:
        return Path(payroll_arg), default_rk
    else:
        return Path(payroll_arg), Path(rk_arg)


def main():
    args = parse_args()
    payroll_path, rk_path = resolve_default_paths(args.payroll, args.recordkeeper)
    output_dir = Path(args.output_dir)

    run_timing_analysis(
        payroll_path=payroll_path,
        rk_path=rk_path,
        output_dir=output_dir,
        late_threshold_days=args.late_threshold,
    )


if __name__ == "__main__":
    main()
