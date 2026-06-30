import os
from pathlib import Path

# -------------------------------------------------------
# Preflight requires explicit mapping — set BEFORE imports
# -------------------------------------------------------
os.environ["MAPPING_YAML_PATH"] = str(
    Path(__file__).resolve().parents[1] / "mapping_example.yaml"
)

import sys
import csv
import pandas as pd

# Ensure repo root (where main.py lives) is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import run_reconciliation  # noqa: E402
from contribution_timing_analyzer_v2 import compute_late_contributions, run_timing_analysis  # noqa: E402


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


def test_multiple_periods_for_one_employee_do_not_cross_join():
    payroll_df = pd.DataFrame(
        [
            {"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 100},
            {"employee_id": "1001", "pay_date": "2025-01-17", "payroll_pretax": 125},
        ]
    )
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-01-03",
                "deposit_date": "2025-01-07",
                "rk_pretax": 100,
            },
            {
                "employee_id": "1001",
                "pay_date": "2025-01-17",
                "deposit_date": "2025-01-21",
                "rk_pretax": 125,
            },
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df)

    assert len(result) == 2
    assert set(result["timing_match_rule"]) == {"employee_id+pay_date"}
    assert int(result["is_late"].sum()) == 0


def test_exact_pay_date_matching_uses_period_reference_date():
    payroll_df = pd.DataFrame(
        [{"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 100}]
    )
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-01-03",
                "deposit_date": "2025-01-16",
                "rk_pretax": 100,
            }
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df, late_threshold_days=5)

    assert len(result) == 1
    assert result.loc[0, "deposit_date"] == pd.Timestamp("2025-01-16")
    assert bool(result.loc[0, "is_late"]) is True
    assert result.loc[0, "timing_match_rule"] == "employee_id+pay_date"


def test_nearest_deposit_fallback_is_deterministic_without_period_keys():
    payroll_df = pd.DataFrame(
        [
            {"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 100},
            {"employee_id": "1001", "pay_date": "2025-01-17", "payroll_pretax": 125},
        ]
    )
    rk_df = pd.DataFrame(
        [
            {"employee_id": "1001", "deposit_date": "2025-01-21", "rk_pretax": 125},
            {"employee_id": "1001", "deposit_date": "2025-01-07", "rk_pretax": 100},
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df)
    ordered = result.sort_values("pay_date").reset_index(drop=True)

    assert len(ordered) == 2
    assert ordered.loc[0, "deposit_date"] == pd.Timestamp("2025-01-07")
    assert ordered.loc[1, "deposit_date"] == pd.Timestamp("2025-01-21")
    assert set(ordered["timing_match_rule"]) == {"employee_id+nearest_deposit_date"}


def test_unmatched_payroll_rows_remain_visible_as_missing_deposits():
    payroll_df = pd.DataFrame(
        [
            {"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 100},
            {"employee_id": "1001", "pay_date": "2025-01-17", "payroll_pretax": 125},
        ]
    )
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-01-03",
                "deposit_date": "2025-01-07",
                "rk_pretax": 100,
            }
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df)

    missing = result[result["missing_deposit"]]
    assert len(missing) == 1
    assert missing.iloc[0]["pay_date"] == pd.Timestamp("2025-01-17")
    assert missing.iloc[0]["unmatched_source"] == "payroll_only"


def test_unmatched_recordkeeper_rows_remain_visible():
    payroll_df = pd.DataFrame(
        [{"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 100}]
    )
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-01-03",
                "deposit_date": "2025-01-07",
                "rk_pretax": 100,
            },
            {
                "employee_id": "1001",
                "pay_date": "2025-01-17",
                "deposit_date": "2025-01-21",
                "rk_pretax": 125,
            },
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df)

    unmatched = result[result["unmatched_recordkeeper"]]
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["unmatched_source"] == "recordkeeper_only"
    assert unmatched.iloc[0]["deposit_date"] == pd.Timestamp("2025-01-21")


def test_invalid_reference_dates_remain_visible_as_unmatched_rows():
    payroll_df = pd.DataFrame(
        [{"employee_id": "1001", "pay_date": "not-a-date", "payroll_pretax": 100}]
    )
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "also-not-a-date",
                "deposit_date": "2025-01-07",
                "rk_pretax": 100,
            }
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df)

    assert len(result) == 2
    assert int(result["missing_deposit"].sum()) == 1
    assert int(result["unmatched_recordkeeper"].sum()) == 1


def test_duplicate_period_keys_are_aggregated_before_matching():
    payroll_df = pd.DataFrame(
        [
            {"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 40},
            {"employee_id": "1001", "pay_date": "2025-01-03", "payroll_pretax": 60},
        ]
    )
    rk_df = pd.DataFrame(
        [
            {
                "employee_id": "1001",
                "pay_date": "2025-01-03",
                "deposit_date": "2025-01-07",
                "rk_pretax": 30,
            },
            {
                "employee_id": "1001",
                "pay_date": "2025-01-03",
                "deposit_date": "2025-01-07",
                "rk_pretax": 70,
            },
        ]
    )

    result = compute_late_contributions(payroll_df, rk_df)

    assert len(result) == 1
    assert result.loc[0, "payroll_pretax"] == 100
    assert result.loc[0, "rk_pretax"] == 100
    assert result.loc[0, "_payroll_row_count"] == 2
    assert result.loc[0, "_rk_row_count"] == 2


def test_realistic_smoke_fixture_detects_expected_late_count(tmp_path: Path):
    smoke_dir = ROOT_DIR / "data" / "demo" / "smoke"
    summary = run_timing_analysis(
        payroll_path=str(smoke_dir / "realistic_smoke_25_payroll.csv"),
        rk_path=str(smoke_dir / "realistic_smoke_25_recordkeeper.csv"),
        output_dir=str(tmp_path),
        late_threshold_days=5,
    )

    late_rows = pd.read_csv(summary["late_contributions_path"])

    assert summary["late_rows"] == 1
    assert summary["total_rows"] == 101
    assert len(late_rows) == 6
    assert int(late_rows["is_late"].sum()) == 1
