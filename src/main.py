from pathlib import Path
import pandas as pd


DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)


def reconcile_payroll_vs_recordkeeper(
    payroll_file: str = "payroll.csv",
    rk_file: str = "recordkeeper.csv",
):
    payroll_df = load_csv(payroll_file)
    rk_df = load_csv(rk_file)

    payroll_df.columns = [c.lower().strip() for c in payroll_df.columns]
    rk_df.columns = [c.lower().strip() for c in rk_df.columns]

    required = {"employee_id", "amount"}
    if not required.issubset(payroll_df.columns):
        raise ValueError("Payroll missing required columns")
    if not required.issubset(rk_df.columns):
        raise ValueError("Recordkeeper missing required columns")

    merged = payroll_df.merge(
        rk_df,
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
    print(f"Only in payroll:            {len(only_in_payroll)}")
    print(f"Only in recordkeeper:       {len(only_in_rk)}")
    print(f"Amount mismatches:          {len(amount_mismatch)}")

    # Write outputs
    DATA_OUT.mkdir(exist_ok=True, parents=True)

    only_in_payroll.to_csv(DATA_OUT / "only_in_payroll.csv", index=False)
    only_in_rk.to_csv(DATA_OUT / "only_in_recordkeeper.csv", index=False)
    amount_mismatch.to_csv(DATA_OUT / "amount_mismatch.csv", index=False)

    print(f"\nOutput written to: {DATA_OUT}")


def main():
    print("Running Prooflink reconciliation with output files...")
    reconcile_payroll_vs_recordkeeper()


if __name__ == "__main__":
    main()
