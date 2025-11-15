from pathlib import Path
import pandas as pd


# =========================
# CONFIGURATION SECTION
# =========================

# File names in data/raw
PAYROLL_FILE = "payroll.csv"
RECORDKEEPER_FILE = "recordkeeper.csv"

# Logical column names → what they might be called in real files
COLUMN_MAP = {
    "employee_id": [
        "employee_id",
        "employee id",
        "emp id",
        "empid",
        "ee id",
        "participant id",
        "participant",
    ],
    "amount": [
        "amount",
        "deferral",
        "ee deferral $",
        "contribution",
        "deposit amount",
        "employee contribution",
    ],
    # If you later want to include dates, we can turn this on:
    # "date": [
    #     "date",
    #     "trade date",
    #     "contribution date",
    #     "pay date",
    # ],
}

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


# =========================
# CORE LOGIC
# =========================


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)


def infer_column_mapping(df: pd.DataFrame, logical_map: dict[str, list[str]]) -> dict[str, str]:
    """
    Given a DataFrame and a dict of logical_name -> list of possible column names,
    return a dict of logical_name -> actual column name in df.
    """
    actual = {}
    # normalize actual cols once
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


def reconcile_payroll_vs_recordkeeper(
    payroll_file: str = PAYROLL_FILE,
    rk_file: str = RECORDKEEPER_FILE,
):
    # Load raw files
    payroll_df = load_csv(payroll_file)
    rk_df = load_csv(rk_file)

    # Infer column mappings separately for each file (in case headers differ)
    payroll_cols = infer_column_mapping(payroll_df, COLUMN_MAP)
    rk_cols = infer_column_mapping(rk_df, COLUMN_MAP)

    required = {"employee_id", "amount"}

    missing_payroll = required - set(payroll_cols.keys())
    missing_rk = required - set(rk_cols.keys())

    if missing_payroll:
        raise ValueError(f"Payroll file missing logical columns: {missing_payroll}. "
                         f"Available columns: {list(payroll_df.columns)}")

    if missing_rk:
        raise ValueError(f"Recordkeeper file missing logical columns: {missing_rk}. "
                         f"Available columns: {list(rk_df.columns)}")

    # Log what we mapped for transparency
    print("\n=== Column Mapping ===")
    print("Payroll mapping:")
    for k, v in payroll_cols.items():
        print(f"  {k} -> {v}")

    print("\nRecordkeeper mapping:")
    for k, v in rk_cols.items():
        print(f"  {k} -> {v}")

    # Build normalized views so we can merge cleanly
    payroll_norm = pd.DataFrame({
        "employee_id": payroll_df[payroll_cols["employee_id"]],
        "amount": payroll_df[payroll_cols["amount"]],
    })

    rk_norm = pd.DataFrame({
        "employee_id": rk_df[rk_cols["employee_id"]],
        "amount": rk_df[rk_cols["amount"]],
    })

    # Merge on employee_id
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

    print("\n=== Reconciliation Summary ===")
    print(f"Total in payroll file:      {len(payroll_norm):>4}")
    print(f"Total in recordkeeper file: {len(rk_norm):>4}")
    print(f"Only in payroll:            {len(only_in_payroll):>4}")
    print(f"Only in recordkeeper:       {len(only_in_rk):>4}")
    print(f"Amount mismatches:          {len(amount_mismatch):>4}")

    # Ensure output folder exists
    DATA_OUT.mkdir(exist_ok=True, parents=True)

    # Export results
    only_in_payroll.to_csv(DATA_OUT / "only_in_payroll.csv", index=False)
    only_in_rk.to_csv(DATA_OUT / "only_in_recordkeeper.csv", index=False)
    amount_mismatch.to_csv(DATA_OUT / "amount_mismatch.csv", index=False)

    print(f"\nOutput written to: {DATA_OUT}")


def main():
    print("Running Prooflink reconciliation with column mapping...")
    reconcile_payroll_vs_recordkeeper()


if __name__ == "__main__":
    main()
