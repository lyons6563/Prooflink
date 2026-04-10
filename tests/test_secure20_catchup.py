"""
Pytest-compatible tests for the Secure 2.0 catch-up analyzer.

Covers both the standalone analyzer (analyze_secure20_catchup) and the
full engine path via run_prooflink_engine against the demo data file.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import pytest

from secure20_catchup_analyzer import analyze_secure20_catchup


# ---------------------------------------------------------------------------
# Unit-level: analyzer works on a minimal synthetic dataframe
# ---------------------------------------------------------------------------

def _make_payroll_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "employee_id": "0",
        "pay_date": "2025-01-15",
        "age": -1,
        "is_hce": False,
        "EE Deferral $": 0.0,
        "EE Roth $": 0.0,
        "catchup_pretax": 0.0,
        "catchup_roth": 0.0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_secure20_no_violations(tmp_path):
    """Employee under 50 with pre-tax catchup: not a violation."""
    df = _make_payroll_df([
        {"employee_id": "1001", "age": 45, "is_hce": True,
         "EE Deferral $": 500.0, "catchup_pretax": 50.0},
    ])
    summary = analyze_secure20_catchup(df, tmp_path)
    assert summary["total_violations"] == 0
    assert summary["hce_violation_count"] == 0


def test_secure20_hce_pretax_catchup_flagged(tmp_path):
    """HCE age 50+ with pre-tax catch-up and no Roth catch-up: Rule A violation."""
    df = _make_payroll_df([
        {"employee_id": "1002", "age": 52, "is_hce": True,
         "EE Deferral $": 500.0, "catchup_pretax": 100.0, "catchup_roth": 0.0},
    ])
    summary = analyze_secure20_catchup(df, tmp_path)
    assert summary["hce_violation_count"] >= 1


def test_secure20_age50_no_catchup_flagged(tmp_path):
    """Age 50+ with zero catch-up amounts: Rule B (potential miscode)."""
    df = _make_payroll_df([
        {"employee_id": "1003", "age": 55, "is_hce": False,
         "EE Deferral $": 800.0, "catchup_pretax": 0.0, "catchup_roth": 0.0},
    ])
    summary = analyze_secure20_catchup(df, tmp_path)
    assert summary["potential_catchup_miscode_count"] >= 1


def test_secure20_csv_written(tmp_path):
    """CSV output is always written, even with zero violations."""
    df = _make_payroll_df([
        {"employee_id": "1004", "age": 30, "is_hce": False},
    ])
    summary = analyze_secure20_catchup(df, tmp_path)
    assert Path(summary["csv_path"]).exists()


def test_secure20_empty_dataframe(tmp_path):
    """Empty input returns zeroed summary and still writes the CSV."""
    df = pd.DataFrame()
    summary = analyze_secure20_catchup(df, tmp_path)
    assert summary["total_rows"] == 0
    assert summary["total_violations"] == 0
    assert Path(summary["csv_path"]).exists()


# ---------------------------------------------------------------------------
# Integration: full engine run against demo data
# ---------------------------------------------------------------------------

DEMO_DIR = ROOT_DIR / "data" / "demo"
PAYROLL_FILE = DEMO_DIR / "secure20_test_payroll.csv"
RK_FILE = DEMO_DIR / "demo_clean_rk.csv"


@pytest.mark.skipif(
    not (PAYROLL_FILE.exists() and RK_FILE.exists()),
    reason="Demo data files not present",
)
def test_secure20_engine_integration(tmp_path):
    """Run the full ProofLink engine and verify the Secure 2.0 summary is present."""
    import os
    os.environ.setdefault(
        "MAPPING_YAML_PATH",
        str(ROOT_DIR / "mapping_example.yaml"),
    )
    from main import EngineConfig, run_prooflink_engine

    config = EngineConfig(
        plan_name="Secure 2.0 Integration Test",
        output_dir=str(tmp_path / "output"),
        proofs_dir=str(tmp_path / "proofs"),
    )
    result = run_prooflink_engine(
        payroll_path=str(PAYROLL_FILE),
        rk_path=str(RK_FILE),
        config=config,
    )

    assert result.run_id
    assert Path(result.evidence_pack_path).exists()

    secure20 = result.summary.get("secure20")
    assert secure20 is not None, "secure20 key missing from summary"
    assert "total_violations" in secure20
    assert "hce_violation_count" in secure20
