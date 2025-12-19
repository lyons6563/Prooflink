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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("rows list is empty")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_len_csv(path: Path) -> int:
    if path.exists() and path.is_file():
        try:
            df = pd.read_csv(path)
            return len(df)
        except Exception:
            return 0
    return 0


def test_reconciliation_no_mismatches(tmp_path: Path):
    """
    Happy-path: payroll and RK match exactly → no mismatches, no late rows.
    """
    payroll_csv = tmp_path / "payroll.csv"
    rk_csv = tmp_path / "rk.csv"
    out_dir = tmp_path / "processed"
    proofs_dir = tmp_path / "proofs"

    # Simple 2 employees, matching deferrals + loans
    payroll_rows = [
        {
            "employee_id": 1001,
            "pay_date": "2025-01-15",
            "def_amount": 100,
            "roth_amount": 0,
            "loan_amount": 50,
        },
        {
            "employee_id": 1002,
            "pay_date": "2025-01-15",
            "def_amount": 200,
            "roth_amount": 0,
            "loan_amount": 0,
        },
    ]
    rk_rows = [
        {
            "employee_id": 1001,
            "record_date": "2025-01-16",
            "def_amount": 100,
            "roth_amount": 0,
            "loan_amount": 50,
        },
        {
            "employee_id": 1002,
            "record_date": "2025-01-16",
            "def_amount": 200,
            "roth_amount": 0,
            "loan_amount": 0,
        },
    ]

    write_csv(payroll_csv, payroll_rows)
    write_csv(rk_csv, rk_rows)

    results = run_reconciliation(
        payroll_csv=str(payroll_csv),
        rk_csv=str(rk_csv),
        payroll_vendor_hint="ADP",
        rk_vendor_hint="VENDOR_RK_1",
        output_dir=str(out_dir),
        proofs_dir=str(proofs_dir),
        mapping_yaml_path=str(Path(__file__).resolve().parents[1] / "mapping_example.yaml"),
    )

    # Evidence pack should exist
    evidence_pack = Path(results["evidence_pack"])
    assert evidence_pack.exists() and evidence_pack.is_file()

    # Mismatch CSVs should exist and be empty; late file may or may not exist
    def_mismatches = Path(results["deferral_mismatches"])
    loan_mismatches = Path(results["loan_mismatches"])
    late_deferrals = Path(results["late_deferrals"])

    assert def_mismatches.exists()
    assert loan_mismatches.exists()

    assert safe_len_csv(def_mismatches) == 0
    assert safe_len_csv(loan_mismatches) == 0
    # safe_len_csv returns 0 if file doesn't exist, which is fine here
    assert safe_len_csv(late_deferrals) == 0


def test_reconciliation_with_mismatches(tmp_path: Path):
    """
    Intentional mismatches: one deferral delta and one loan delta.
    We just assert counts > 0 and evidence pack exists.
    """
    payroll_csv = tmp_path / "payroll_mismatch.csv"
    rk_csv = tmp_path / "rk_mismatch.csv"
    out_dir = tmp_path / "processed_mismatch"
    proofs_dir = tmp_path / "proofs_mismatch"

    payroll_rows = [
        {
            "employee_id": 1001,
            "pay_date": "2025-01-15",
            "def_amount": 100,
            "roth_amount": 0,
            "loan_amount": 50,
        },
        {
            "employee_id": 1002,
            "pay_date": "2025-01-15",
            "def_amount": 200,
            "roth_amount": 0,
            "loan_amount": 0,
        },
    ]
    rk_rows = [
        # Employee 1001 matches deferral but has loan mismatch
        {
            "employee_id": 1001,
            "record_date": "2025-01-16",
            "def_amount": 100,
            "roth_amount": 0,
            "loan_amount": 40,
        },
        # Employee 1002 has a deferral mismatch
        {
            "employee_id": 1002,
            "record_date": "2025-01-16",
            "def_amount": 195,
            "roth_amount": 0,
            "loan_amount": 0,
        },
    ]

    write_csv(payroll_csv, payroll_rows)
    write_csv(rk_csv, rk_rows)

    results = run_reconciliation(
        payroll_csv=str(payroll_csv),
        rk_csv=str(rk_csv),
        payroll_vendor_hint="ADP",
        rk_vendor_hint="VENDOR_RK_1",
        output_dir=str(out_dir),
        proofs_dir=str(proofs_dir),
        mapping_yaml_path=str(Path(__file__).resolve().parents[1] / "mapping_example.yaml"),
    )

    evidence_pack = Path(results["evidence_pack"])
    assert evidence_pack.exists() and evidence_pack.is_file()

    def_mismatches = Path(results["deferral_mismatches"])
    loan_mismatches = Path(results["loan_mismatches"])

    assert safe_len_csv(def_mismatches) > 0
    assert safe_len_csv(loan_mismatches) > 0
