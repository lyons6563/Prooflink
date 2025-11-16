from pathlib import Path
import json
import pandas as pd
import numpy as np



# =========================
# CONFIGURATION SECTION
# =========================

# File names in data/raw
PAYROLL_FILE = "payroll_soc1_challenge.csv"
RECORDKEEPER_FILE = "rk_soc1_challenge.csv"

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

# Compliance threshold: max allowed business days between pay_date and deposit_date
MAX_BUSINESS_DAYS_LAG = 5  # adjust per policy if needed

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CONFIG_NAME = "empower_adp.json"


def load_config(config_name: str = CONFIG_NAME) -> dict:
    path = CONFIG_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

# Logical column names → candidate physical column names
# We separate payroll vs RK and deferrals vs loans
COLUMN_MAP = {
    # IDs
    "employee_id": [
        "employee_id",
        "employee id",
        "ee id",
        "emp id",
        "empid",
        "empnumber",
        "participant id",
        "participant",
        "part_id",
    ],

    # Dates
    "pay_date": [
        "pay_date",
        "pay date",
        "payroll date",
        "check date",
        "payroll_run_date",
        "pay period end date",
        "pay period date",
        "date",
    ],
    "deposit_date": [
        "deposit_date",
        "deposit date",
        "recordkeeper date",
        "trade date",
        "post_date",
        "posting date",
        "funding date",
        "contribution date",
    ],

    # PAYROLL side amounts
    "payroll_pretax": [
        "pretax_defl",
        "ee deferral $",
        "employee contribution",
        "amount",
    ],
    "payroll_roth": [
        "roth_defl",
        "ee roth $",
        "roth contribution",
    ],
    "payroll_loan": [
        "loan_pmt",
        "loan repay $",
        "loan repayment",
    ],

    # RECORDKEEPER side amounts
    "rk_pretax": [
        "ee_pretax",
        "employee contribution",
        "amount",
    ],
    "rk_roth": [
        "ee_roth",
        "roth contribution",
    ],
    "rk_loan": [
        "loan_contr",
        "loan repayment",
    ],

    # Fallback single-amount column (legacy/simple files)
    "amount": [
        "amount",
        "deferral",
        "ee deferral $",
        "deposit amount",
        "employee contribution",
    ],
}


# =========================
# CORE UTILITIES
# =========================

def load_csv(name: str) -> pd.DataFrame:
    path = DATA_RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)


def infer_column_mapping(df: pd.DataFrame, logical_map: dict[str, list[str]]) -> dict[str, str]:
    """
    Given a DataFrame and a dict of logical_name -> list of possible column names,
    return a dict of logical_name -> actual column name in df where possible.
    """
    actual = {}
    normalized_to_actual = {c.lower().strip(): c for c in df.columns}

    for logical, candidates in logical_map.items():
        match = None
        for candidate in candidates:
            key = candidate.lower().strip()
            if key in normalized_to_actual:
                match = normalized_to_actual[key]
                break
        if match:
            actual[logical] = match

    return actual


def parse_amount(series: pd.Series) -> pd.Series:
    """
    Clean a numeric/amount series: strip $, commas, spaces, and coerce to float.
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
        .replace({"": "0", "nan": "0", "None": "0"})
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """
    Read a CSV if it exists, otherwise return None.
    """
    if path.exists():
        return pd.read_csv(path)
    return None

def compute_business_days_lag(df: pd.DataFrame, pay_col: str, dep_col: str) -> pd.Series:
    """
    Compute business day lag between pay_col and dep_col.
    Returns a Series aligned with df index with NaN where invalid.
    """
    pay_parsed = pd.to_datetime(df[pay_col], errors="coerce")
    dep_parsed = pd.to_datetime(df[dep_col], errors="coerce")

    mask_valid = pay_parsed.notna() & dep_parsed.notna()
    result = pd.Series(np.nan, index=df.index, dtype="float")

    if mask_valid.any():
        pay = pay_parsed[mask_valid].dt.date.values.astype("datetime64[D]")
        dep = dep_parsed[mask_valid].dt.date.values.astype("datetime64[D]")
        result.loc[mask_valid] = np.busday_count(pay, dep)

    return result


def reconcile_stream(
    stream_name: str,
    payroll_df: pd.DataFrame,
    rk_df: pd.DataFrame,
    payroll_cols: dict[str, str],
    rk_cols: dict[str, str],
    required_keys: list[str],
    write_outputs: bool = True,
):
    """
    Generic reconciliation function for a given stream, e.g. "deferrals" or "loans".
    required_keys is the list of logical keys needed on both sides (e.g. ["employee_id", "def_amount"])
    """
    # Validate required logical keys exist
    missing_payroll = [k for k in required_keys if k not in payroll_cols]
    missing_rk = [k for k in required_keys if k not in rk_cols]

    if missing_payroll or missing_rk:
        print(
            f"\n[WARN] Stream '{stream_name}': missing columns. "
            f"Payroll missing: {missing_payroll}, RK missing: {missing_rk}. Skipping this stream."
        )
        return None

    p_id = payroll_df[payroll_cols["employee_id"]]
    r_id = rk_df[rk_cols["employee_id"]]

    p_amt = parse_amount(payroll_df[payroll_cols[required_keys[1]]])
    r_amt = parse_amount(rk_df[rk_cols[required_keys[1]]])

    # Attach dates if available
    p_date = payroll_df[payroll_cols["pay_date"]] if "pay_date" in payroll_cols else None
    r_date = rk_df[rk_cols["deposit_date"]] if "deposit_date" in rk_cols else None

    payroll_norm = pd.DataFrame({"employee_id": p_id, "amount": p_amt})
    rk_norm = pd.DataFrame({"employee_id": r_id, "amount": r_amt})

    if p_date is not None:
        payroll_norm["pay_date"] = p_date
    if r_date is not None:
        rk_norm["deposit_date"] = r_date

    merged = payroll_norm.merge(
        rk_norm,
        on="employee_id",
        how="outer",
        suffixes=("_payroll", "_rk"),
        indicator=True,
    )

    only_in_payroll = merged[merged["_merge"] == "left_only"]
    only_in_rk = merged[merged["_merge"] == "right_only"]
    amount_mismatch = merged[
        (merged["_merge"] == "both")
        & (merged["amount_payroll"].fillna(0) != merged["amount_rk"].fillna(0))
    ]

    print(f"\n=== {stream_name.upper()} Reconciliation Summary ===")
    print(f"Total in payroll ({stream_name}):      {len(payroll_norm):>4}")
    print(f"Total in recordkeeper ({stream_name}): {len(rk_norm):>4}")
    print(f"Only in payroll ({stream_name}):       {len(only_in_payroll):>4}")
    print(f"Only in recordkeeper ({stream_name}):  {len(only_in_rk):>4}")
    print(f"Amount mismatches ({stream_name}):     {len(amount_mismatch):>4}")

    # Ensure output folder exists
    DATA_OUT.mkdir(exist_ok=True, parents=True)

    if write_outputs:
        base = stream_name.lower()
        only_in_payroll.to_csv(DATA_OUT / f"only_in_payroll_{base}.csv", index=False)
        only_in_rk.to_csv(DATA_OUT / f"only_in_recordkeeper_{base}.csv", index=False)
        amount_mismatch.to_csv(DATA_OUT / f"{base}_mismatch.csv", index=False)
    # Late funding detection if dates are present
    if "pay_date" in merged.columns and "deposit_date" in merged.columns:
        # We already have pay_date and deposit_date columns on the merged frame
        lag = compute_business_days_lag(merged, "pay_date", "deposit_date")
        merged["business_days_lag"] = lag

        late_mask = lag > MAX_BUSINESS_DAYS_LAG
        late_df = merged[late_mask].copy()

        if not late_df.empty:
            late_path = DATA_OUT / f"late_{base}_contributions.csv"
            late_df.to_csv(late_path, index=False)
            print(
                f"Late {stream_name} contributions: {len(late_df)} rows "
                f"(> {MAX_BUSINESS_DAYS_LAG} business days). Written to: {late_path}"
            )
        else:
            print(
                f"No late {stream_name} contributions detected "
                f"(threshold = {MAX_BUSINESS_DAYS_LAG} business days)."
            )
    else:
        print(f"No usable dates for late {stream_name} detection.")



# =========================
# MAIN ORCHESTRATION
# =========================

def reconcile_payroll_vs_recordkeeper():
    # Load config and override defaults if present
    cfg = load_config()
    global MAX_BUSINESS_DAYS_LAG

    payroll_file = cfg.get("payroll_file", PAYROLL_FILE)
    rk_file = cfg.get("recordkeeper_file", RECORDKEEPER_FILE)
    MAX_BUSINESS_DAYS_LAG = cfg.get("max_business_days_lag", MAX_BUSINESS_DAYS_LAG)

    # Load raw files
    payroll_df = load_csv(payroll_file)
    rk_df = load_csv(rk_file)


    # Infer mappings
    payroll_cols = infer_column_mapping(payroll_df, COLUMN_MAP)
    rk_cols = infer_column_mapping(rk_df, COLUMN_MAP)

    print("\n=== Column Mapping (Payroll) ===")
    for k, v in payroll_cols.items():
        print(f"  {k} -> {v}")
    print("\n=== Column Mapping (Recordkeeper) ===")
    for k, v in rk_cols.items():
        print(f"  {k} -> {v}")

    # Ensure we at least know employee_id
    if "employee_id" not in payroll_cols or "employee_id" not in rk_cols:
        raise ValueError(
            f"Cannot proceed: employee_id not found on both sides. "
            f"Payroll columns: {list(payroll_df.columns)}, RK columns: {list(rk_df.columns)}"
        )

    # Ensure pay/deposit dates map if present
    # Not required, but used for late logic
    if "pay_date" not in payroll_cols:
        print("[WARN] No pay_date mapped on payroll file; late logic will be limited.")
    if "deposit_date" not in rk_cols:
        print("[WARN] No deposit_date mapped on recordkeeper file; late logic will be limited.")

    # Build synthetic logical keys for streams:
    #  - deferrals: sum of pretax + roth
    #  - loans: loan column
    # To leverage reconcile_stream, we materialize these as virtual columns in temporary frames.

    # Copy dataframes so we don't mutate originals
    p = payroll_df.copy()
    r = rk_df.copy()

    # DEFERRALS
    # Payroll side
    p_def = None
    if "payroll_pretax" in payroll_cols or "payroll_roth" in payroll_cols or "amount" in payroll_cols:
        pretax = parse_amount(p[payroll_cols["payroll_pretax"]]) if "payroll_pretax" in payroll_cols else 0.0
        roth = parse_amount(p[payroll_cols["payroll_roth"]]) if "payroll_roth" in payroll_cols else 0.0
        # fallback: single amount column if no pretax/roth split
        if "payroll_pretax" not in payroll_cols and "payroll_roth" not in payroll_cols and "amount" in payroll_cols:
            pretax = parse_amount(p[payroll_cols["amount"]])
            roth = 0.0
        p_def = pretax + roth
        p["deferral_amount"] = p_def

    # RK side
    r_def = None
    if "rk_pretax" in rk_cols or "rk_roth" in rk_cols or "amount" in rk_cols:
        pretax_rk = parse_amount(r[rk_cols["rk_pretax"]]) if "rk_pretax" in rk_cols else 0.0
        roth_rk = parse_amount(r[rk_cols["rk_roth"]]) if "rk_roth" in rk_cols else 0.0
        if "rk_pretax" not in rk_cols and "rk_roth" not in rk_cols and "amount" in rk_cols:
            pretax_rk = parse_amount(r[rk_cols["amount"]])
            roth_rk = 0.0
        r_def = pretax_rk + roth_rk
        r["deferral_amount"] = r_def

    # LOANS
    if "payroll_loan" in payroll_cols:
        p["loan_amount"] = parse_amount(p[payroll_cols["payroll_loan"]])
    if "rk_loan" in rk_cols:
        r["loan_amount"] = parse_amount(r[rk_cols["rk_loan"]])

    # Build extended mapping dicts with virtual keys
    payroll_cols_ext = payroll_cols.copy()
    rk_cols_ext = rk_cols.copy()

    if "deferral_amount" in p.columns:
        payroll_cols_ext["def_amount"] = "deferral_amount"
    if "deferral_amount" in r.columns:
        rk_cols_ext["def_amount"] = "deferral_amount"

    if "loan_amount" in p.columns:
        payroll_cols_ext["loan_amount"] = "loan_amount"
    if "loan_amount" in r.columns:
        rk_cols_ext["loan_amount"] = "loan_amount"

    # Map pay/deposit dates into logical keys if present
    # (For late logic inside reconcile_stream)
    if "pay_date" in payroll_cols_ext:
        pass  # already mapped
    else:
        # try to map if payroll has something date-like but not in COLUMN_MAP (unlikely but defensive)
        pass

    if "deposit_date" in rk_cols_ext:
        pass
    else:
        pass

    # Now run deferrals stream
    reconcile_stream(
        stream_name="deferrals",
        payroll_df=p,
        rk_df=r,
        payroll_cols=payroll_cols_ext,
        rk_cols=rk_cols_ext,
        required_keys=["employee_id", "def_amount"],
    )

    # Now run loans stream
    reconcile_stream(
        stream_name="loans",
        payroll_df=p,
        rk_df=r,
        payroll_cols=payroll_cols_ext,
        rk_cols=rk_cols_ext,
        required_keys=["employee_id", "loan_amount"],
    )

def generate_excel_report():
    """
    Build a consolidated Excel report from whatever CSVs exist in data/processed.
    Sheets:
      - Summary
      - Deferrals: mismatches, only-in-payroll, only-in-RK, late
      - Loans:    mismatches, only-in-payroll, only-in-RK, late
    """
    report_path = DATA_OUT / "reconciliation_report.xlsx"

    # Ensure output dir exists
    DATA_OUT.mkdir(exist_ok=True, parents=True)

    streams = ["deferrals", "loans"]
    summary_rows = []

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        for stream in streams:
            base = stream.lower()

            files = {
                "only_in_payroll": DATA_OUT / f"only_in_payroll_{base}.csv",
                "only_in_recordkeeper": DATA_OUT / f"only_in_recordkeeper_{base}.csv",
                "mismatch": DATA_OUT / f"{base}_mismatch.csv",
                "late": DATA_OUT / f"late_{base}_contributions.csv",
            }

            for label, path in files.items():
                df = safe_read_csv(path)
                sheet_name = f"{stream[:3].title()} - {label.replace('_', ' ').title()}"

                if df is not None and not df.empty:
                    # Write detailed sheet
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                    # Add summary row
                    summary_rows.append(
                        {
                            "stream": stream,
                            "category": label,
                            "rows": len(df),
                        }
                    )
                else:
                    # Even if file doesn't exist, still write a tiny placeholder so structure is predictable
                    placeholder = pd.DataFrame(
                        [{"info": f"No rows for {stream}/{label} or file missing"}]
                    )
                    placeholder.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    summary_rows.append(
                        {
                            "stream": stream,
                            "category": label,
                            "rows": 0,
                        }
                    )

        # Summary sheet
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\nConsolidated Excel report written to: {report_path}")
    print("You can drag this into Google Sheets or email it as-is.")

def main():
    print("Running Prooflink reconciliation for deferrals + loans (with business-day late checks)...")
    reconcile_payroll_vs_recordkeeper()
    generate_excel_report()


if __name__ == "__main__":
    main()
