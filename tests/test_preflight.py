"""
Tests for preflight.py — verifies that run_preflight correctly allows/blocks
reconciliation runs based on the presence of required vs optional fields.

Key rules being tested:
  - employee_id is required on both sides; absence blocks the run
  - At least one amount field must be mappable on each side
  - All other YAML-mapping fields are optional; their absence must NOT block
  - CSVs that use canonical column names directly (e.g. loan_amount, pay_date)
    must be accepted without adding them to every examples list
  - Empty files must be blocked (join-key coverage check)
"""

import csv
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preflight import run_preflight

MAPPING = str(ROOT_DIR / "mapping_example.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _payroll(tmp_path, rows):
    return _write(tmp_path / "payroll.csv", rows)


def _rk(tmp_path, rows):
    return _write(tmp_path / "rk.csv", rows)


# Minimal valid payroll / RK rows using canonical column names
_PAYROLL_ROW = {
    "employee_id": "1001",
    "pay_date": "2025-01-15",
    "def_amount": "100.00",
    "roth_amount": "0.00",
    "loan_amount": "50.00",
}
_RK_ROW = {
    "employee_id": "1001",
    "record_date": "2025-01-16",
    "def_amount": "100.00",
    "roth_amount": "0.00",
    "loan_amount": "50.00",
}


# ---------------------------------------------------------------------------
# Happy-path: minimal canonical-column CSVs must pass
# ---------------------------------------------------------------------------

def test_preflight_passes_canonical_columns(tmp_path):
    """CSVs using canonical column names (def_amount, loan_amount, etc.) must pass."""
    p = _payroll(tmp_path, [_PAYROLL_ROW])
    r = _rk(tmp_path, [_RK_ROW])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert safe, f"Expected safe=True but got False. Report: {report}"


def test_preflight_passes_adp_style_columns(tmp_path):
    """ADP-style column names that appear in the YAML examples must also pass."""
    p = _payroll(tmp_path, [{
        "Emp ID": "1001",
        "Check Date": "2025-01-15",
        "EE Deferral $": "100.00",
        "EE Roth $": "0.00",
        "Loan Repay $": "50.00",
    }])
    r = _rk(tmp_path, [{
        "Participant ID": "1001",
        "Post Date": "2025-01-16",
        "EE Pretax": "100.00",
        "EE Roth": "0.00",
        "Loan Contr": "50.00",
    }])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert safe, f"Expected safe=True but got False. Report: {report}"


def test_preflight_optional_fields_absent_does_not_block(tmp_path):
    """
    Optional fields (is_hce, catchup_pretax, employment_status, etc.) being absent
    must not block the run — they should be silently skipped.
    """
    # Bare-minimum CSV: only the required columns
    p = _payroll(tmp_path, [{"employee_id": "1001", "def_amount": "100.00"}])
    r = _rk(tmp_path, [{"employee_id": "1001", "def_amount": "100.00"}])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert safe, f"Optional-field absence should not block. Report: {report}"


def test_preflight_timing_warning_when_dates_absent(tmp_path):
    """
    Missing pay_date / deposit_date should produce warnings, not a block.
    """
    p = _payroll(tmp_path, [{"employee_id": "1001", "def_amount": "100.00"}])
    r = _rk(tmp_path, [{"employee_id": "1001", "def_amount": "100.00"}])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert safe
    assert any("pay_date" in w for w in report["warnings"])
    assert any("deposit_date" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Blocking conditions: missing employee_id
# ---------------------------------------------------------------------------

def test_preflight_blocks_missing_payroll_employee_id(tmp_path):
    """No employee_id on payroll side must block."""
    p = _payroll(tmp_path, [{"pay_date": "2025-01-15", "def_amount": "100.00"}])
    r = _rk(tmp_path, [_RK_ROW])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert not safe
    assert "employee_id" in report["missing_fields"]["payroll"]


def test_preflight_blocks_missing_rk_employee_id(tmp_path):
    """No employee_id on RK side must block."""
    p = _payroll(tmp_path, [_PAYROLL_ROW])
    r = _rk(tmp_path, [{"record_date": "2025-01-16", "def_amount": "100.00"}])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert not safe
    assert "employee_id" in report["missing_fields"]["recordkeeper"]


# ---------------------------------------------------------------------------
# Blocking conditions: no amount field on either side
# ---------------------------------------------------------------------------

def test_preflight_blocks_no_amount_on_payroll(tmp_path):
    """Payroll with only employee_id (no deferral/roth/loan) must block."""
    p = _payroll(tmp_path, [{"employee_id": "1001"}])
    r = _rk(tmp_path, [_RK_ROW])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert not safe


def test_preflight_blocks_no_amount_on_rk(tmp_path):
    """RK with only employee_id must block."""
    p = _payroll(tmp_path, [_PAYROLL_ROW])
    r = _rk(tmp_path, [{"employee_id": "1001"}])
    safe, report = run_preflight(str(p), str(r), MAPPING)
    assert not safe


# ---------------------------------------------------------------------------
# Blocking conditions: file not found / empty file
# ---------------------------------------------------------------------------

def test_preflight_blocks_missing_payroll_file(tmp_path):
    """Non-existent payroll file must block immediately."""
    r = _rk(tmp_path, [_RK_ROW])
    safe, report = run_preflight(str(tmp_path / "ghost.csv"), str(r), MAPPING)
    assert not safe
    assert "blocked_files" in report


def test_preflight_blocks_missing_rk_file(tmp_path):
    """Non-existent RK file must block immediately."""
    p = _payroll(tmp_path, [_PAYROLL_ROW])
    safe, report = run_preflight(str(p), str(tmp_path / "ghost.csv"), MAPPING)
    assert not safe
    assert "blocked_files" in report


def test_preflight_blocks_empty_payroll(tmp_path):
    """Header-only (0 data rows) payroll file must block."""
    path = tmp_path / "empty_payroll.csv"
    path.write_text("employee_id,def_amount\n")
    r = _rk(tmp_path, [_RK_ROW])
    safe, report = run_preflight(str(path), str(r), MAPPING)
    assert not safe
    assert report["join_key_empty_file"]["payroll"]


def test_preflight_blocks_empty_rk(tmp_path):
    """Header-only (0 data rows) RK file must block."""
    p = _payroll(tmp_path, [_PAYROLL_ROW])
    path = tmp_path / "empty_rk.csv"
    path.write_text("employee_id,def_amount\n")
    safe, report = run_preflight(str(p), str(path), MAPPING)
    assert not safe
    assert report["join_key_empty_file"]["recordkeeper"]


# ---------------------------------------------------------------------------
# Mapped-fields content checks
# ---------------------------------------------------------------------------

def test_preflight_mapped_fields_populated(tmp_path):
    """mapped_fields should contain the canonical names actually found."""
    p = _payroll(tmp_path, [_PAYROLL_ROW])
    r = _rk(tmp_path, [_RK_ROW])
    _, report = run_preflight(str(p), str(r), MAPPING)
    assert "employee_id" in report["mapped_fields"]["payroll"]
    assert "employee_id" in report["mapped_fields"]["recordkeeper"]


def test_preflight_join_key_coverage_100(tmp_path):
    """All rows having employee_id → 100% coverage reported."""
    p = _payroll(tmp_path, [_PAYROLL_ROW, {**_PAYROLL_ROW, "employee_id": "1002"}])
    r = _rk(tmp_path, [_RK_ROW])
    _, report = run_preflight(str(p), str(r), MAPPING)
    assert report["join_key_coverage"]["payroll"] == 100.0
