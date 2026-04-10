import os
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eligibility_drift import detect_eligibility_drift

# data/demo/ is the canonical location for test fixtures
PAYROLL_PATH = BASE_DIR / "data" / "demo" / "elig_test_payroll.csv"


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


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.skipif(
    not PAYROLL_PATH.exists(),
    reason="elig_test_payroll.csv not present in data/demo/",
)
def test_eligibility_drift_runs():
    """detect_eligibility_drift executes without error and returns a DataFrame."""
    df = pd.read_csv(PAYROLL_PATH)
    drift_df = detect_eligibility_drift(df, grace_days=0)
    assert isinstance(drift_df, pd.DataFrame)


if __name__ == "__main__":
    main()
