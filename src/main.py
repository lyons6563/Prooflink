from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)


def reconcile_payroll_vs_recordkeeper(
    payroll_file: str = "payroll.csv",
    rk_file: str = "recordkeeper.csv",
) -> None:
    payroll_df = load_csv(payroll_file)
    rk_df = load_csv(rk_file)

    # Standardize column names (defensive)
    payroll_df.columns = [c.lower().strip() for c in payroll_df.columns]
    rk_df.columns = [c.lower().strip() for c in rk_df.columns]

    # Basic expectation: employee_id + amount in both
    required_cols = {"employee_id", "amount"}
    if not required_cols.issubset(set(payroll_df.columns)):
        raise ValueError(f"Payroll file missing columns: {required_cols - set(payroll_df.columns)}")
    if not required_cols.issubset(set(rk_df.columns)):
        raise ValueError(f"Recordkeeper file missing columns: {required_cols - set(rk_df.columns)}")

    # Key on employee_id
    merged = payroll_df.merge(
        rk_df,
        on="employee_id",
        how="outer",
        suffixes=("_payroll", "_rk"),
        indicator=True,
    )

    # Flag categories
    only_in_payroll = merged[merged["_merge"] == "left_only"]
    only_in_rk = merged[merged["_merge"] == "right_only"]

    amount_mismatch = merged[
        (merged["_merge"] == "both")
        & (merged["amount_payroll"].fillna(0) != merged["amount_rk"].fillna(0))
    ]

    print("\n=== Reconciliation Summary ===")
    print(f"Total in payroll file:      {len(payroll_df):>4}")
    print(f"Total in recordkeeper file: {len(rk_df):>4}")
    print(f"Only in payroll:            {len(only_in_payroll):>4}")
    print(f"Only in recordkeeper:       {len(only_in_rk):>4}")
    print(f"Amount mismatches:          {len(amount_mismatch):>4}")

    if not only_in_payroll.empty:
        print("\n-- Only in payroll --")
        print(only_in_payroll[["employee_id", "amount_payroll"]])

    if not only_in_rk.empty:
        print("\n-- Only in recordkeeper --")
        print(only_in_rk[["employee_id", "amount_rk"]])

    if not amount_mismatch.empty:
        print("\n-- Amount mismatches --")
        print(
            amount_mismatch[
                ["employee_id", "amount_payroll", "amount_rk"]
            ]
        )


def main():
    print("Running basic Prooflink reconciliation test...")
    reconcile_payroll_vs_recordkeeper()


if __name__ == "__main__":
    main()

