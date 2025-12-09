import os
import sys
import pandas as pd

# Make sure parent directory (where eligibility_drift.py lives) is on the path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # dev/src
sys.path.insert(0, BASE_DIR)

from eligibility_drift import detect_eligibility_drift

PAYROLL_PATH = os.path.join(BASE_DIR, "data", "raw", "elig_test_payroll.csv")


def main():
    print(f"Loading payroll from: {PAYROLL_PATH}")
    df = pd.read_csv(PAYROLL_PATH)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    drift_df = detect_eligibility_drift(df, grace_days=0)
    print("Drift rows detected:", len(drift_df))

    if not drift_df.empty:
        print("Drift sample:")
        print(drift_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
