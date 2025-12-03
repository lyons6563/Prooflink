from pathlib import Path
import sys
from datetime import datetime, timedelta

import pandas as pd

# Ensure src root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from contribution_timing_analyzer_v2 import compute_late_contributions  # noqa: E402


def test_timing_flags_late_deferral():
    """
    Simple case: one on-time row, one late row (6 days vs 5-day threshold).
    This mimics the normalized column schema the timing analyzer expects.
    """
    from datetime import datetime, timedelta
    import pandas as pd

    # --- Define dates ---
    pay_date = datetime(2025, 1, 2)
    ontime_deposit = pay_date + timedelta(days=3)
    late_deposit = pay_date + timedelta(days=10)

    # --- Build payroll dataframe (normalized column names) ---
    payroll_df = pd.DataFrame(
        [
            {
                "employee_id": 1001,
                "pay_date": pay_date,
                "payroll_pretax": 100,
                "payroll_roth": 0,
                "payroll_loan": 0,
            },
            {
                "employee_id": 1002,
                "pay_date": pay_date,
                "payroll_pretax": 200,
                "payroll_roth": 0,
                "payroll_loan": 0,
            },
        ]
    )

    # --- Build recordkeeper dataframe (normalized column names) ---
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": 1001,
                "deposit_date": ontime_deposit,
                "rk_pretax": 100,
                "rk_roth": 0,
                "rk_loan": 0,
            },
            {
                "employee_id": 1002,
                "deposit_date": late_deposit,
                "rk_pretax": 200,
                "rk_roth": 0,
                "rk_loan": 0,
            },
        ]
    )

    # Ensure proper datetime dtypes for .dt accessor
    payroll_df["pay_date"] = pd.to_datetime(payroll_df["pay_date"])
    rk_df["deposit_date"] = pd.to_datetime(rk_df["deposit_date"])

    # --- Call the timing function ---
    result = compute_late_contributions(
        payroll_df=payroll_df,
        rk_df=rk_df,
        late_threshold_days=5,
    )

    # --- Assertions ---
    assert "is_late" in result.columns
    assert int(result["is_late"].sum()) == 1
