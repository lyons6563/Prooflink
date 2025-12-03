# main.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from datetime import datetime
import hashlib
import json
import numpy as np
import re
import traceback
import zipfile

import pandas as pd

from vendors import (
    PAYROLL_VENDOR_SIGNATURES,
    RK_VENDOR_SIGNATURES,
    detect_vendor_with_confidence,
    apply_vendor_column_mapping,
)


@dataclass
class RunSummary:
    plan_name: str
    payroll_vendor: str
    rk_vendor: str
    payroll_vendor_confidence: float
    rk_vendor_confidence: float
    total_deferrals_payroll: float
    total_deferrals_rk: float
    total_loans_payroll: float
    total_loans_rk: float
    deferral_mismatch_count: int
    loan_mismatch_count: int
    late_deferral_count: int
    evidence_pack_path: Path
    run_id: str  # e.g. timestamp or UUID


def run_reconciliation_with_summary(
    payroll_csv: Path,
    rk_csv: Path,
    output_dir: Path,
    plan_name: str = "Unknown Plan",
    payroll_vendor_hint: Optional[str] = None,
    rk_vendor_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper that calls the existing run_reconciliation engine and produces
    a RunSummary object with high-level KPIs.
    
    RETURN CONTRACT (CRITICAL - DO NOT BREAK):
    ==========================================
    This function ALWAYS returns a Dict[str, Any] (never a list or tuple).
    The returned dict must have the following structure:
    
    SUCCESS CASE:
    -------------
    {
        "summary": RunSummary,  # REQUIRED: RunSummary dataclass object with:
            # - plan_name: str
            # - payroll_vendor: str
            # - rk_vendor: str
            # - payroll_vendor_confidence: float
            # - rk_vendor_confidence: float
            # - total_deferrals_payroll: float
            # - total_deferrals_rk: float
            # - total_loans_payroll: float
            # - total_loans_rk: float
            # - deferral_mismatch_count: int
            # - loan_mismatch_count: int
            # - late_deferral_count: int
            # - evidence_pack_path: Path
            # - run_id: str
        
        "results_dict": Dict[str, Any],  # REQUIRED: Underlying reconciliation results from run_reconciliation()
            # Must contain (at minimum):
            # - "evidence_pack": str (REQUIRED: path to evidence pack ZIP file)
            # - "vendor_detection": Dict (REQUIRED) with:
            #     - "payroll_vendor": str (or None)
            #     - "rk_vendor": str (or None)
            #     - "payroll_vendor_confidence": float
            #     - "rk_vendor_confidence": float
            # - "totals": Dict (REQUIRED) with:
            #     - "deferrals_payroll": float
            #     - "deferrals_rk": float
            #     - "loans_payroll": float
            #     - "loans_rk": float
            # - "mismatches": Dict (REQUIRED) with:
            #     - "deferral_count": int
            #     - "loan_count": int
            # - "timing": Dict (REQUIRED) with:
            #     - "late_deferral_count": int
            # - "run_id": str (optional, defaults to empty string)
            # - "deferral_mismatches": str (REQUIRED by build_anomaly_narrative(): path to deferral_mismatches.csv)
            # - "loan_mismatches": str (REQUIRED by build_anomaly_narrative(): path to loan_mismatches.csv)
            # - "late_deferrals": str (REQUIRED by build_anomaly_narrative(): path to late_deferrals_contributions.csv)
            # - "reconciliation_report": str (optional: path to reconciliation_report.xlsx)
            # - "only_in_payroll": str (optional: path to only_in_payroll_deferrals.csv)
            # - "only_in_recordkeeper": str (optional: path to only_in_recordkeeper_deferrals.csv)
            # - "late_loans": str (optional: path to late_loans_contributions.csv)
            # - "manifest": str (optional: path to proof manifest JSON)
        
        # "error" key is ABSENT or None on success
    }
    
    FAILURE CASE:
    ------------
    {
        "summary": None,  # REQUIRED: Must be None (not omitted)
        
        "results_dict": Dict,  # REQUIRED: Empty dict {} or partial results if available
        
        "error": str  # REQUIRED: Error message string (full traceback from traceback.format_exc())
    }
    
    STREAMLIT UI DEPENDENCIES:
    --------------------------
    The Streamlit UI (streamlit_app.py) depends on:
    - results.get("summary") -> RunSummary object (for all metrics display)
    - results.get("results_dict") -> Dict (for build_anomaly_narrative() and file paths)
    - results.get("error") -> str (for error display)
    
    If you modify this return structure, you MUST update:
    1. streamlit_app.py: run_reconciliation_with_stdout_capture()
    2. streamlit_app.py: render_reconciliation_tab()
    3. streamlit_app.py: render_batch_reconciliation_tab()
    4. streamlit_app.py: build_anomaly_narrative() (uses results_dict)
    
    This function ALWAYS returns a dict, never a list or tuple.
    """
    try:
        reconciliation_results = run_reconciliation(
            payroll_csv=str(payroll_csv),
            rk_csv=str(rk_csv),
            payroll_vendor_hint=payroll_vendor_hint,
            rk_vendor_hint=rk_vendor_hint,
            output_dir=str(output_dir),
        )
        
        # Normalize: if run_reconciliation returns a tuple or list, convert to dict
        if isinstance(reconciliation_results, (tuple, list)):
            # If it's a tuple/list, try to extract meaningful data
            # Default: treat as error case
            return {
                "summary": None,
                "results_dict": {},
                "error": f"run_reconciliation returned unexpected type: {type(reconciliation_results).__name__}"
            }
        
        # Ensure reconciliation_results is a dict
        if not isinstance(reconciliation_results, dict):
            return {
                "summary": None,
                "results_dict": {},
                "error": f"run_reconciliation returned {type(reconciliation_results)}, expected dict"
            }

        # evidence_pack must exist based on your engine
        evidence_pack_path = Path(reconciliation_results["evidence_pack"])

        vendor_detection = reconciliation_results.get("vendor_detection", {})
        totals = reconciliation_results.get("totals", {})
        mismatches = reconciliation_results.get("mismatches", {})
        timing = reconciliation_results.get("timing", {})

        # ✅ Prefer explicit hints first, then detection, then fallback
        display_payroll_vendor = (
            payroll_vendor_hint
            or vendor_detection.get("payroll_vendor")
            or "Unknown / Generic"
        )
        display_rk_vendor = (
            rk_vendor_hint
            or vendor_detection.get("rk_vendor")
            or "Unknown / Generic"
        )

        run_summary = RunSummary(
            plan_name=plan_name,
            payroll_vendor=display_payroll_vendor,
            rk_vendor=display_rk_vendor,
            payroll_vendor_confidence=float(vendor_detection.get("payroll_vendor_confidence", 0.0)),
            rk_vendor_confidence=float(vendor_detection.get("rk_vendor_confidence", 0.0)),
            total_deferrals_payroll=float(totals.get("deferrals_payroll", 0.0)),
            total_deferrals_rk=float(totals.get("deferrals_rk", 0.0)),
            total_loans_payroll=float(totals.get("loans_payroll", 0.0)),
            total_loans_rk=float(totals.get("loans_rk", 0.0)),
            deferral_mismatch_count=int(mismatches.get("deferral_count", 0)),
            loan_mismatch_count=int(mismatches.get("loan_count", 0)),
            late_deferral_count=int(timing.get("late_deferral_count", 0)),
            evidence_pack_path=evidence_pack_path,
            run_id=reconciliation_results.get("run_id", ""),
        )
        
        # Write JSON summary file to run directory
        run_dir = Path(output_dir)
        summary_dict = {
            "plan_name": run_summary.plan_name,
            "payroll_vendor": run_summary.payroll_vendor,
            "rk_vendor": run_summary.rk_vendor,
            "payroll_vendor_confidence": run_summary.payroll_vendor_confidence,
            "rk_vendor_confidence": run_summary.rk_vendor_confidence,
            "total_deferrals_payroll": run_summary.total_deferrals_payroll,
            "total_deferrals_rk": run_summary.total_deferrals_rk,
            "total_loans_payroll": run_summary.total_loans_payroll,
            "total_loans_rk": run_summary.total_loans_rk,
            "deferral_mismatch_count": run_summary.deferral_mismatch_count,
            "loan_mismatch_count": run_summary.loan_mismatch_count,
            "late_deferral_count": run_summary.late_deferral_count,
            "evidence_pack_path": str(run_summary.evidence_pack_path),
            "run_id": run_summary.run_id,
        }
        
        run_summary_path = run_dir / "run_summary.json"
        with run_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        
        # Return standardized dict structure
        return {
            "summary": run_summary,
            "results_dict": reconciliation_results,
        }
    
    except Exception as e:
        # On any exception, return error dict structure with full traceback
        error_text = traceback.format_exc()
        return {
            "summary": None,
            "results_dict": {},
            "error": error_text
        }


from datetime import datetime
import hashlib
import json
import numpy as np
import zipfile

import pandas as pd

from pathlib import Path




from pathlib import Path

from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np

import pandas as pd

def run_reconciliation(
    payroll_csv: str,
    rk_csv: str,
    payroll_vendor_hint: str | None = None,
    rk_vendor_hint: str | None = None,
    output_dir: str = "data/processed",
    proofs_dir: str = "proofs",
) -> dict:
    """
    Execute a full ProofLink reconciliation run for the given payroll + RK CSVs.

    This is the function Streamlit should call.

    It will:
      - clear/overwrite previous CSV/XLSX outputs in output_dir
      - run deferral + loan reconciliation using the two provided CSV files
      - generate reconciliation_report.xlsx
      - generate a new proof_manifest_*.json in proofs_dir
      - build an evidence_pack.zip bundle
      - return a dict of paths for the UI
    """

    # Resolve and prepare directories
    output_dir_path = Path(output_dir)
    proofs_dir_path = Path(proofs_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    proofs_dir_path.mkdir(parents=True, exist_ok=True)

    # Point the rest of the module at these dirs
    global DATA_OUT, PROOFS_DIR, MAX_BUSINESS_DAYS_LAG
    DATA_OUT = output_dir_path
    PROOFS_DIR = proofs_dir_path

    # Clean previous outputs so metrics are based ONLY on this run
    for pattern in ("*.csv", "*.xlsx"):
        for p in output_dir_path.glob(pattern):
            try:
                p.unlink()
            except Exception as e:
                print(f"[WARN] Could not delete old output {p}: {e}")

    # Load config ONLY for things like business-day threshold (ignore file names)
    try:
        cfg = load_config()
        MAX_BUSINESS_DAYS_LAG = cfg.get("max_business_days_lag", MAX_BUSINESS_DAYS_LAG)
    except FileNotFoundError:
        cfg = {}
        print("[INFO] No config file found; using defaults for lag threshold.")

    # Resolve and validate input paths
    payroll_path = Path(payroll_csv)
    rk_path = Path(rk_csv)

    if not payroll_path.exists():
        raise FileNotFoundError(f"Payroll CSV not found: {payroll_path}")
    if not rk_path.exists():
        raise FileNotFoundError(f"Recordkeeper CSV not found: {rk_path}")

    # Load raw files DIRECTLY from the given paths
    payroll_df = pd.read_csv(payroll_path)
    rk_df = pd.read_csv(rk_path)
    
    # Normalize column names to handle flexible deferral/roth variants
    payroll_df = normalize_column_names(payroll_df)
    rk_df = normalize_column_names(rk_df)

    # =========================
    # Vendor detection with confidence
    # =========================
    payroll_vendor, payroll_confidence = detect_vendor_with_confidence(
        payroll_df, PAYROLL_VENDOR_SIGNATURES, payroll_vendor_hint
    )
    rk_vendor, rk_confidence = detect_vendor_with_confidence(
        rk_df, RK_VENDOR_SIGNATURES, rk_vendor_hint
    )
    
    # If hint provided, use it (confidence still calculated)
    if payroll_vendor_hint:
        payroll_vendor = payroll_vendor_hint
    if rk_vendor_hint:
        rk_vendor = rk_vendor_hint

    print("\n=== Vendor Detection ===")
    print(f"Detected payroll vendor:     {payroll_vendor or 'Unknown / Generic'} (confidence: {payroll_confidence:.2f})")
    print(f"Detected recordkeeper:       {rk_vendor or 'Unknown / Generic'} (confidence: {rk_confidence:.2f})")
    
    if payroll_confidence < 0.65 and not payroll_vendor_hint:
        print(f"[WARN] Low confidence ({payroll_confidence:.2f}) for payroll vendor detection. Manual verification recommended.")
    if rk_confidence < 0.65 and not rk_vendor_hint:
        print(f"[WARN] Low confidence ({rk_confidence:.2f}) for recordkeeper vendor detection. Manual verification recommended.")
    
    # Apply vendor-specific column mapping
    payroll_df = apply_vendor_column_mapping(payroll_df, payroll_vendor, PAYROLL_VENDOR_SIGNATURES)
    rk_df = apply_vendor_column_mapping(rk_df, rk_vendor, RK_VENDOR_SIGNATURES)

    # =========================
    # Column mapping
    # =========================
    payroll_cols = infer_vendor_mapping(
        payroll_df,
        payroll_vendor,
        PAYROLL_VENDOR_COLUMN_MAPS,
        COLUMN_MAP,
    )
    rk_cols = infer_vendor_mapping(
        rk_df,
        rk_vendor,
        RK_VENDOR_COLUMN_MAPS,
        COLUMN_MAP,
    )

    print("\n=== Column Mapping (Payroll) ===")
    for k, v in payroll_cols.items():
        print(f"  {k} -> {v}")
    print("\n=== Column Mapping (Recordkeeper) ===")
    for k, v in rk_cols.items():
        print(f"  {k} -> {v}")

    # Guardrails
    if "employee_id" not in payroll_cols or "employee_id" not in rk_cols:
        raise ValueError(
            f"Cannot proceed: employee_id not found on both sides. "
            f"Payroll columns: {list(payroll_df.columns)}, RK columns: {list(rk_df.columns)}"
        )

    if "pay_date" not in payroll_cols:
        print("[WARN] No pay_date mapped on payroll file; late logic will be limited.")
    if "deposit_date" not in rk_cols:
        print("[WARN] No deposit_date mapped on recordkeeper file; late logic will be limited.")

    # =========================
    # Build derived columns for deferrals + loans
    # =========================
    p = payroll_df.copy()
    r = rk_df.copy()

    # DEFERRALS – payroll side
    # Calculate total deferral (pretax + roth) when available
    payroll_pretax_col = payroll_cols.get("payroll_pretax")
    payroll_roth_col = payroll_cols.get("payroll_roth")
    payroll_amount_col = payroll_cols.get("amount")
    
    # Check for normalized column names directly if not mapped
    if not payroll_pretax_col and "EE Deferral $" in p.columns:
        payroll_pretax_col = "EE Deferral $"
    if not payroll_roth_col and "EE Roth $" in p.columns:
        payroll_roth_col = "EE Roth $"
    
    if payroll_pretax_col or payroll_roth_col or payroll_amount_col:
        pretax = (
            parse_amount(p[payroll_pretax_col])
            if payroll_pretax_col
            else 0.0
        )
        roth = (
            parse_amount(p[payroll_roth_col])
            if payroll_roth_col
            else 0.0
        )

        # fallback: single-amount column
        if (
            not payroll_pretax_col
            and not payroll_roth_col
            and payroll_amount_col
        ):
            pretax = parse_amount(p[payroll_amount_col])
            roth = 0.0

        # Total deferral = pretax + roth (use both if available)
        p["deferral_amount"] = pretax + roth

    # DEFERRALS – RK side
    # Calculate total deferral (pretax + roth) when available
    rk_pretax_col = rk_cols.get("rk_pretax")
    rk_roth_col = rk_cols.get("rk_roth")
    rk_amount_col = rk_cols.get("amount")
    
    # Check for normalized column names directly if not mapped
    if not rk_pretax_col and "EE Deferral $" in r.columns:
        rk_pretax_col = "EE Deferral $"
    if not rk_roth_col and "EE Roth $" in r.columns:
        rk_roth_col = "EE Roth $"
    
    if rk_pretax_col or rk_roth_col or rk_amount_col:
        pretax_rk = (
            parse_amount(r[rk_pretax_col])
            if rk_pretax_col
            else 0.0
        )
        roth_rk = (
            parse_amount(r[rk_roth_col])
            if rk_roth_col
            else 0.0
        )

        if (
            not rk_pretax_col
            and not rk_roth_col
            and rk_amount_col
        ):
            pretax_rk = parse_amount(r[rk_amount_col])
            roth_rk = 0.0

        # Total deferral = pretax_rk + roth_rk (use both if available)
        r["deferral_amount"] = pretax_rk + roth_rk

    # LOANS
    if "payroll_loan" in payroll_cols:
        p["loan_amount"] = parse_amount(p[payroll_cols["payroll_loan"]])

    if "rk_loan" in rk_cols:
        r["loan_amount"] = parse_amount(r[rk_cols["rk_loan"]])
    elif "payroll_loan" in payroll_cols:
        # Payroll has loans, RK has no explicit loan column -> treat RK as zero loans
        r["loan_amount"] = 0.0

    # Extended logical mappings
    payroll_cols_ext = payroll_cols.copy()
    rk_cols_ext = rk_cols.copy()

    if "deferral_amount" in p.columns:
        payroll_cols_ext["def_amount"] = "deferral_amount"
    if "deferral_amount" in r.columns:
        rk_cols_ext["def_amount"] = "deferral_amount"

    if "loan_amount" in p.columns:
        payroll_cols_ext["loan_amount"] = "loan_amount"
    if "loan_amount" in r.columns:
        rk_cols_ext["loan_amount"] = "loan_amount"

    # =========================
    # Run reconciliations (aggregated per employee)
    # =========================
    reconcile_stream(
        stream_name="deferrals",
        payroll_df=p,
        rk_df=r,
        payroll_cols=payroll_cols_ext,
        rk_cols=rk_cols_ext,
        required_keys=["employee_id", "def_amount"],
        aggregate_by_employee=True,
    )

    reconcile_stream(
        stream_name="loans",
        payroll_df=p,
        rk_df=r,
        payroll_cols=payroll_cols_ext,
        rk_cols=rk_cols_ext,
        required_keys=["employee_id", "loan_amount"],
        aggregate_by_employee=True,
    )

    # =========================
    # Excel report + proof manifest
    # =========================
    outputs = generate_excel_report()
    manifest_path = write_proof_manifest(
        payroll_file=payroll_path,
        rk_file=rk_path,
        cfg=cfg,
        outputs=outputs,
    )

    # =========================
    # Metrics for summary
    # =========================
    def safe_sum(df: pd.DataFrame, col: str) -> float:
        if col in df.columns:
            try:
                return float(pd.to_numeric(df[col], errors="coerce").sum())
            except Exception:
                return 0.0
        return 0.0

    def safe_len_csv(path: Path) -> int:
        if path.exists() and path.is_file():
            try:
                df = pd.read_csv(path)
                return len(df)
            except Exception:
                return 0
        return 0

    # Totals
    totals = {
        "deferrals_payroll": safe_sum(p, "deferral_amount"),
        "deferrals_rk": safe_sum(r, "deferral_amount"),
        "loans_payroll": safe_sum(p, "loan_amount"),
        "loans_rk": safe_sum(r, "loan_amount"),
    }

    # Output paths
    deferral_mismatches_path = output_dir_path / "deferral_mismatches.csv"
    loan_mismatches_path = output_dir_path / "loan_mismatches.csv"
    only_in_payroll_path = output_dir_path / "only_in_payroll_deferrals.csv"
    only_in_recordkeeper_path = output_dir_path / "only_in_recordkeeper_deferrals.csv"
    late_deferrals_path = output_dir_path / "late_deferrals_contributions.csv"
    late_loans_path = output_dir_path / "late_loans_contributions.csv"

    # Mismatch + timing counts based on files
    mismatches = {
        "deferral_count": safe_len_csv(deferral_mismatches_path),
        "loan_count": safe_len_csv(loan_mismatches_path),
        "only_in_payroll_count": safe_len_csv(only_in_payroll_path),
        "only_in_recordkeeper_count": safe_len_csv(only_in_recordkeeper_path),
    }

    timing = {
        "late_deferral_count": safe_len_csv(late_deferrals_path),
        "late_loan_count": safe_len_csv(late_loans_path),
    }

    vendor_detection = {
        "payroll_vendor": payroll_vendor or "Unknown / Generic",
        "rk_vendor": rk_vendor or "Unknown / Generic",
        "payroll_vendor_confidence": payroll_confidence,
        "rk_vendor_confidence": rk_confidence,
    }

    # =========================
    # Return paths + metrics for Streamlit
    # =========================
    results = {
        "reconciliation_report": str(output_dir_path / "reconciliation_report.xlsx"),
        "deferral_mismatches": str(deferral_mismatches_path),
        "loan_mismatches": str(loan_mismatches_path),
        "only_in_payroll": str(only_in_payroll_path),
        "only_in_recordkeeper": str(only_in_recordkeeper_path),
        "late_deferrals": str(late_deferrals_path),
        "late_loans": str(late_loans_path),
        "manifest": str(manifest_path),
        "vendor_detection": vendor_detection,
        "totals": totals,
        "mismatches": mismatches,
        "timing": timing,
    }

    # Build consolidated evidence pack ZIP
    evidence_zip = build_evidence_pack(results)
    results["evidence_pack"] = str(evidence_zip)

    print("\nRun complete. Key outputs:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    return results
   
    print("\nRun complete. Key outputs:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results

def build_evidence_pack(results: dict) -> Path:
    """
    Build a single ZIP that bundles the key outputs for this run:
      - Excel report
      - mismatch CSVs
      - late contribution CSVs
      - only-in-* CSVs
      - manifest JSON (if present)
    """
    zip_path = DATA_OUT / "prooflink_evidence_pack.zip"

    # Remove old pack if it exists
    if zip_path.exists():
        try:
            zip_path.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete old evidence pack: {e}")

    keys_to_include = [
        "reconciliation_report",
        "deferral_mismatches",
        "loan_mismatches",
        "only_in_payroll",
        "only_in_recordkeeper",
        "late_deferrals",
        "late_loans",
        "manifest",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in keys_to_include:
            path_str = results.get(key)
            if not path_str:
                continue
            p = Path(path_str)
            if p.exists() and p.is_file():
                zf.write(p, arcname=p.name)

    print(f"Evidence pack written to: {zip_path}")
    return zip_path

    # =========================
    # Return paths for Streamlit
    # =========================
    results = {
        "reconciliation_report": str(output_dir_path / "reconciliation_report.xlsx"),
        "deferral_mismatches": str(output_dir_path / "deferral_mismatches.csv"),
        "loan_mismatches": str(output_dir_path / "loan_mismatches.csv"),
        "only_in_payroll": str(output_dir_path / "only_in_payroll_deferrals.csv"),
        "only_in_recordkeeper": str(output_dir_path / "only_in_recordkeeper_deferrals.csv"),
        "late_deferrals": str(output_dir_path / "late_deferrals_contributions.csv"),
        "late_loans": str(output_dir_path / "late_loans_contributions.csv"),
        "manifest": str(manifest_path),
    }

    # Build consolidated evidence pack ZIP
    evidence_zip = build_evidence_pack(results)
    results["evidence_pack"] = str(evidence_zip)

    print("\nRun complete. Key outputs:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results



# =========================
# CONFIGURATION SECTION
# =========================

# File names in data/raw
PAYROLL_FILE = "payroll_soc1_challenge.csv"
RECORDKEEPER_FILE = "rk_soc1_challenge.csv"

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

# Compliance threshold: max allowed business days between pay_date and deposit_date
MAX_BUSINESS_DAYS_LAG = 5  # adjust per policy if needed

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CONFIG_NAME = "synthetic_400_adp_empower.json"

PROOFS_DIR = Path(__file__).resolve().parents[1] / "proofs"
PROOFS_DIR.mkdir(exist_ok=True)


def load_config(config_name: str = CONFIG_NAME) -> dict:
    path = CONFIG_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)
# ============================================================
# Hashing + Merkle helper functions
# ============================================================

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of an entire file (binary)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    """SHA-256 of a string, for Merkle layers."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def merkle_root(hashes: list[str]) -> str:
    """
    Build a simple Merkle root from a list of hex hashes.
    If list is empty, return empty string.
    If odd count, last hash is duplicated at that layer.
    """
    if not hashes:
        return ""
    layer = hashes[:]
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else layer[i]
            combined = sha256_string(left + right)
            next_layer.append(combined)
        layer = next_layer
    return layer[0]


def hash_csv_rows(path: Path, max_samples: int = 5) -> dict:
    """
    Compute row-level hashes and a Merkle root for a CSV.
    Normalizes rows by sorted column order.
    Returns summary metadata, not all row hashes.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {
            "row_count": 0,
            "error": f"Failed to read CSV: {e}",
            "merkle_root": "",
            "row_hash_sample": [],
        }

    col_names = sorted(df.columns.tolist())
    row_hashes: list[str] = []

    for _, row in df.iterrows():
        parts = []
        for col in col_names:
            val = row[col]
            parts.append(f"{col}={val}")
        row_str = "|".join(parts)
        row_hashes.append(sha256_string(row_str))

    root = merkle_root(row_hashes)
    sample = row_hashes[:max_samples]

    return {
        "row_count": int(len(df)),
        "columns": col_names,
        "merkle_root": root,
        "row_hash_sample": sample,
    }
def write_proof_manifest(
    payroll_file: Path,
    rk_file: Path,
    cfg: dict,
    outputs: dict,
) -> Path:
    """
    Write a JSON manifest that ties this run to:
      - the payroll and recordkeeper input files
      - the config used
      - the hashed outputs (CSVs/XLSX)
    """
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")


    manifest = {
        "run_timestamp_utc": timestamp,
        "payroll_file": str(payroll_file),
        "recordkeeper_file": str(rk_file),
        "config_name": CONFIG_NAME,
        "outputs": outputs,
    }

    out_name = f"proof_manifest_{timestamp.replace(':', '').replace('-', '')}.json"
    out_path = PROOFS_DIR / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Proof manifest written to: {out_path}")
    return out_path

# Logical column names → candidate physical column names
# We separate payroll vs RK and deferrals vs loans
COLUMN_MAP = {
    # IDs
    "employee_id": [
        "employee_id",
        "employee id",
        "ee id",
        "emp id",
        "empid",
        "empnumber",
        "employee_number",
        "participant id",
        "participant_id",
        "participant",
        "part_id",
    ],

    # Dates
    "pay_date": [
        "pay_date",
        "pay date",
        "payroll date",          # <-- your Payroll_Date
        "check date",
        "checkdt",
        "payroll_run_date",
        "pay period end date",
        "pay period date",
        "date",
    ],
    "deposit_date": [
        "deposit_date",
        "deposit date",
        "recordkeeper date",
        "trade date",
        "post_date",
        "posting date",
        "funding date",
        "contribution date",
        "transaction_effective_date",   # <-- add this line
    ],

    # PAYROLL side amounts
    "payroll_pretax": [
        "pretax_defl",
        "ee deferral $",
        "ee deferral",  # Without $ sign
        "EE Deferral $",  # Exact normalized name
        "employee contribution",
        "pre-tax",
        "pre tax",
        "amount",
        "457b_ee_pretax_amt",
    ],
    "payroll_roth": [
        "roth_defl",
        "ee roth $",
        "ee roth",  # Without $ sign
        "EE Roth $",  # Exact normalized name
        "roth contribution",
        "roth",
        "457b_ee_roth_amt",
    ],
    "payroll_loan": [
        "loan_pmt",
        "loan repay $",
        "loan repayment",
        "457b_loan_repay_amt",         # <-- add this line
    ],

    # RECORDKEEPER side amounts
    "rk_pretax": [
        "ee_pretax",
        "ee deferral $",
        "ee deferral",  # Without $ sign
        "EE Deferral $",  # Exact normalized name
        "employee contribution",
        "pre-tax cont",
        "pre tax cont",
        "pre-tax",
        "amount",
    ],
    "rk_roth": [
        "ee_roth",
        "ee roth $",
        "ee roth",  # Without $ sign
        "EE Roth $",  # Exact normalized name
        "roth contribution",
        "roth cont",
        "roth",
    ],
    "rk_loan": [
        "loan_contr",
        "loan repayment",
    ],

    # Fallback single-amount column (legacy/simple files)
    "amount": [
        "amount",
        "deferral",
        "ee deferral $",
        "deposit amount",
        "employee contribution",
        "pre-tax cont",
        "roth cont",
        "contribution_amount",        # <-- add this line (RK 457b file)
    ],
}



# =========================
# VENDOR SIGNATURES
# =========================
# Note: PAYROLL_VENDOR_SIGNATURES and RK_VENDOR_SIGNATURES are imported from vendors.py
# The duplicate definitions below have been removed to prevent conflicts.
# All vendor signatures should be defined in vendors.py with the proper structure:
# {
#     "VendorName": {
#         "signature_keywords": [...],
#         "column_map": {...}
#     }
# }

# =========================
# VENDOR-SPECIFIC COLUMN MAPS (HYBRID: STRICT + FLEXIBLE)
# =========================

# For now we fully define ADP + Empower; others can fallback to the generic COLUMN_MAP.
# Structure:
#   { vendor_name: { "required": {logical: [candidates...]}, "optional": {...} } }

PAYROLL_VENDOR_COLUMN_MAPS = {
    "ADP": {
        "required": {
            # Core identity + timing
            "employee_id": ["EE ID"],
            "pay_date": ["Check Date"],
            # Core amounts (deferrals + loans)
            "payroll_pretax": ["EE Deferral $"],
            "payroll_roth": ["EE Roth $"],
            "payroll_loan": ["Loan Repay $"],
        },
        "optional": {
            # Fallback single-amount column for legacy flows
            "amount": ["EE Deferral $"],
            # You can add division/location/paygroup later if needed
        },
    },
    # Stubs for future vendors – will fall back to generic COLUMN_MAP for now
    "Paychex": {"required": {}, "optional": {}},
    "Paylocity": {"required": {}, "optional": {}},
    "Paycor": {"required": {}, "optional": {}},
    "Workday": {"required": {}, "optional": {}},
}

RK_VENDOR_COLUMN_MAPS = {
    "Empower": {
        "required": {
            "employee_id": ["Participant ID"],
            "deposit_date": ["Trade Date"],
            "rk_pretax": ["Employee Contribution"],
            "rk_roth": ["Roth Contribution"],
            "rk_loan": ["Loan Repayment"],
        },
        "optional": {
            "amount": ["Employee Contribution"],
        },
    },
    # Stubs for future recordkeepers – generic COLUMN_MAP will handle them for now
    "Fidelity": {"required": {}, "optional": {}},
    "Vanguard": {"required": {}, "optional": {}},
    "Principal": {"required": {}, "optional": {}},
    "Voya": {"required": {}, "optional": {}},
}

# =========================
# CORE UTILITIES
# =========================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standardize deferral/roth column variants.
    
    Maps various column name variants to standard names:
    - Deferral variants → "EE Deferral $"
    - Roth variants → "EE Roth $"
    
    Case-insensitive and whitespace-normalized.
    Does not modify loan_amount or other columns.
    """
    df = df.copy()
    
    # Normalize column names: lowercase, strip whitespace, normalize whitespace to underscores
    normalized_cols = {}
    for col in df.columns:
        # Lowercase, strip, and replace any whitespace (spaces, tabs, etc.) with underscores
        normalized = re.sub(r'\s+', '_', col.strip().lower())
        normalized_cols[col] = normalized
    
    # Deferral column variants
    deferral_variants = {
        "def_amount",
        "employee_deferral",
        "deferral",
        "ee_deferral",
        "contribution_amount",
        "pretax",
        "employee_pre_tax",
    }
    
    # Roth column variants
    roth_variants = {
        "roth_amount",
        "roth_deferral",
        "roth",
        "roth_contribution",
    }
    
    # Build rename mapping
    rename_map = {}
    for original_col in df.columns:
        normalized = normalized_cols[original_col]
        
        if normalized in deferral_variants:
            rename_map[original_col] = "EE Deferral $"
        elif normalized in roth_variants:
            rename_map[original_col] = "EE Roth $"
        # loan_amount and other columns are left unchanged
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    df = pd.read_csv(path)
    return normalize_column_names(df)


def infer_column_mapping(df: pd.DataFrame, logical_map: dict[str, list[str]]) -> dict[str, str]:
    """
    Given a DataFrame and a dict of logical_name -> list of possible column names,
    return a dict of logical_name -> actual column name in df where possible.
    """
    actual = {}
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

def infer_vendor_mapping(
    df: pd.DataFrame,
    vendor_name: str | None,
    vendor_maps: dict[str, dict],
    generic_map: dict[str, list[str]],
) -> dict[str, str]:
    """
    Hybrid mapping:
      - If we have a vendor-specific map with required columns, try strict matching for those.
      - If any required logical column is missing, fall back to generic mapping.
      - Optional columns are mapped if present.
      - Any remaining logicals can be filled by the generic infer_column_mapping as a fallback.
    """
    # No vendor or no mapping configured -> generic
    if not vendor_name or vendor_name not in vendor_maps:
        return infer_column_mapping(df, generic_map)

    vendor_cfg = vendor_maps[vendor_name]
    required_cfg = vendor_cfg.get("required") or {}
    optional_cfg = vendor_cfg.get("optional") or {}

    if not required_cfg:
        # Nothing vendor-specific defined yet -> generic
        return infer_column_mapping(df, generic_map)

    normalized_to_actual = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str] = {}
    missing_required: list[str] = []

    # Strict pass for required columns
    for logical, candidates in required_cfg.items():
        found = None
        for cand in candidates:
            key = cand.lower().strip()
            if key in normalized_to_actual:
                found = normalized_to_actual[key]
                break
        if found:
            mapping[logical] = found
        else:
            missing_required.append(logical)

    if missing_required:
        print(
            f"[WARN] Vendor {vendor_name}: missing required logical columns {missing_required}. "
            f"Falling back to generic mapping."
        )
        return infer_column_mapping(df, generic_map)

    # Optional columns: best-effort
    for logical, candidates in optional_cfg.items():
        for cand in candidates:
            key = cand.lower().strip()
            if key in normalized_to_actual:
                mapping[logical] = normalized_to_actual[key]
                break

    # Let the generic mapping fill any remaining logical keys we didn't set
    generic_mapping = infer_column_mapping(df, generic_map)
    for logical, actual in generic_mapping.items():
        mapping.setdefault(logical, actual)

    return mapping

def detect_vendor(df: pd.DataFrame, signatures: dict[str, list[str]]) -> str | None:
    """
    Try to detect a vendor by checking if all signature patterns appear
    in the dataframe's columns (case-insensitive, substring match).
    Returns the vendor name or None if no match.
    """
    cols = [c.lower().strip() for c in df.columns]

    for vendor, patterns in signatures.items():
        match_all = True
        for pattern in patterns:
            pattern = pattern.lower().strip()
            if not any(pattern in col for col in cols):
                match_all = False
                break
        if match_all:
            return vendor

    return None

def parse_amount(series: pd.Series) -> pd.Series:
    """
    Clean a numeric/amount series: strip $, commas, spaces, and coerce to float.
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
        .replace({"": "0", "nan": "0", "None": "0"})
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """
    Read a CSV if it exists, otherwise return None.
    """
    if path.exists():
        return pd.read_csv(path)
    return None
def _find_id_column(df: pd.DataFrame) -> str | None:
    """
    Try to find an employee/participant id column by common names.
    """
    if df is None or df.empty:
        return None

    candidates = [
        "employee_id",
        "empnumber",
        "ee id",
        "participant id",
        "part_id",
        "ee_id",
    ]
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    # Fallback: any column with 'id' in it
    for col in df.columns:
        if "id" in col.lower():
            return col
    return None


def _find_lag_column(df: pd.DataFrame) -> str | None:
    """
    Try to find a lag / business-day difference column.
    """
    if df is None or df.empty:
        return None

    for col in df.columns:
        name = col.lower()
        if "lag" in name and "day" in name:
            return col
        if "business" in name and "day" in name:
            return col
    return None
def compute_compliance_metrics() -> tuple[list[dict], list[str]]:
    """
    Compute high-level compliance metrics from the processed CSVs.
    Returns:
      - a list of dicts for writing to Excel
      - a list of strings for printing to the console
    """
    metrics_rows: list[dict] = []
    console_lines: list[str] = []

    # Load processed CSVs if they exist
    late_def = safe_read_csv(DATA_OUT / "late_deferrals_contributions.csv")
    late_loan = safe_read_csv(DATA_OUT / "late_loans_contributions.csv")
    mis_def = safe_read_csv(DATA_OUT / "deferral_mismatches.csv")
    mis_loan = safe_read_csv(DATA_OUT / "loan_mismatches.csv")
    only_p_def = safe_read_csv(DATA_OUT / "only_in_payroll_deferrals.csv")
    only_r_def = safe_read_csv(DATA_OUT / "only_in_recordkeeper_deferrals.csv")
    only_p_loan = safe_read_csv(DATA_OUT / "only_in_payroll_loans.csv")
    only_r_loan = safe_read_csv(DATA_OUT / "only_in_recordkeeper_loans.csv")

    # Convenience list
    def _safe_len(df: pd.DataFrame | None) -> int:
        return 0 if df is None else len(df)

    # Late contributions
    late_def_rows = _safe_len(late_def)
    late_loan_rows = _safe_len(late_loan)

    # Unique employees with any late contribution
    late_emp_ids: set = set()
    for df in (late_def, late_loan):
        if df is not None and not df.empty:
            id_col = _find_id_column(df)
            if id_col:
                late_emp_ids.update(df[id_col].dropna().astype(str).unique().tolist())

    # Lag stats
    max_lag = None
    avg_lag = None
    lag_values = []
    for df in (late_def, late_loan):
        if df is not None and not df.empty:
            lag_col = _find_lag_column(df)
            if lag_col and pd.api.types.is_numeric_dtype(df[lag_col]):
                lag_values.extend(df[lag_col].dropna().tolist())
    if lag_values:
        max_lag = max(lag_values)
        avg_lag = sum(lag_values) / len(lag_values)

    metrics_rows.append(
        {
            "Category": "Late Contributions",
            "Metric": "Late deferral rows",
            "Value": late_def_rows,
            "Notes": "Count of rows flagged as late for deferrals",
        }
    )
    metrics_rows.append(
        {
            "Category": "Late Contributions",
            "Metric": "Late loan rows",
            "Value": late_loan_rows,
            "Notes": "Count of rows flagged as late for loans",
        }
    )
    metrics_rows.append(
        {
            "Category": "Late Contributions",
            "Metric": "Unique employees with any late event",
            "Value": len(late_emp_ids),
            "Notes": "Across deferrals and loans",
        }
    )
    metrics_rows.append(
        {
            "Category": "Late Contributions",
            "Metric": "Max business-day lag (late rows only)",
            "Value": max_lag if max_lag is not None else "",
            "Notes": "",
        }
    )
    metrics_rows.append(
        {
            "Category": "Late Contributions",
            "Metric": "Average business-day lag (late rows only)",
            "Value": round(avg_lag, 2) if avg_lag is not None else "",
            "Notes": "",
        }
    )

    # Mismatches
    mis_def_rows = _safe_len(mis_def)
    mis_loan_rows = _safe_len(mis_loan)

    # Unique employees with mismatches
    mis_emp_ids: set = set()
    for df in (mis_def, mis_loan):
        if df is not None and not df.empty:
            id_col = _find_id_column(df)
            if id_col:
                mis_emp_ids.update(df[id_col].dropna().astype(str).unique().tolist())

    metrics_rows.append(
        {
            "Category": "Mismatches",
            "Metric": "Deferral mismatch rows",
            "Value": mis_def_rows,
            "Notes": "Rows where payroll vs RK deferral amounts differ",
        }
    )
    metrics_rows.append(
        {
            "Category": "Mismatches",
            "Metric": "Loan mismatch rows",
            "Value": mis_loan_rows,
            "Notes": "Rows where payroll vs RK loan amounts differ",
        }
    )
    metrics_rows.append(
        {
            "Category": "Mismatches",
            "Metric": "Unique employees with any mismatch",
            "Value": len(mis_emp_ids),
            "Notes": "",
        }
    )

    # Coverage / completeness
    metrics_rows.append(
        {
            "Category": "Coverage",
            "Metric": "Only-in-payroll deferral rows",
            "Value": _safe_len(only_p_def),
            "Notes": "Payroll has row; RK is missing it (deferrals)",
        }
    )
    metrics_rows.append(
        {
            "Category": "Coverage",
            "Metric": "Only-in-recordkeeper deferral rows",
            "Value": _safe_len(only_r_def),
            "Notes": "RK has row; payroll is missing it (deferrals)",
        }
    )
    metrics_rows.append(
        {
            "Category": "Coverage",
            "Metric": "Only-in-payroll loan rows",
            "Value": _safe_len(only_p_loan),
            "Notes": "Payroll has row; RK is missing it (loans)",
        }
    )
    metrics_rows.append(
        {
            "Category": "Coverage",
            "Metric": "Only-in-recordkeeper loan rows",
            "Value": _safe_len(only_r_loan),
            "Notes": "RK has row; payroll is missing it (loans)",
        }
    )

    # Console summary
    console_lines.append("=== Compliance Dashboard (High-Level) ===")
    console_lines.append(
        f"Late deferral rows: {late_def_rows:,} | Late loan rows: {late_loan_rows:,}"
    )
    console_lines.append(
        f"Employees with any late event: {len(late_emp_ids):,}"
    )
    if max_lag is not None and avg_lag is not None:
        console_lines.append(
            f"Late funding lag (BD) — max: {max_lag}, average (late only): {avg_lag:.2f}"
        )
    console_lines.append(
        f"Deferral mismatch rows: {mis_def_rows:,} | Loan mismatch rows: {mis_loan_rows:,}"
    )
    console_lines.append(
        f"Employees with any mismatch: {len(mis_emp_ids):,}"
    )
    console_lines.append(
        "Coverage gaps — "
        f"only-in-payroll (def/loan): "
        f"{_safe_len(only_p_def):,}/{_safe_len(only_p_loan):,}, "
        f"only-in-RK (def/loan): "
        f"{_safe_len(only_r_def):,}/{_safe_len(only_r_loan):,}"
    )

    return metrics_rows, console_lines



def compute_business_days_lag(df: pd.DataFrame, pay_col: str, dep_col: str) -> pd.Series:
    """
    Compute business day lag between pay_col and dep_col.
    Returns a Series aligned with df index with NaN where invalid.
    """
    pay_parsed = pd.to_datetime(df[pay_col], errors="coerce")
    dep_parsed = pd.to_datetime(df[dep_col], errors="coerce")

    mask_valid = pay_parsed.notna() & dep_parsed.notna()
    result = pd.Series(np.nan, index=df.index, dtype="float")

    if mask_valid.any():
        pay = pay_parsed[mask_valid].dt.date.values.astype("datetime64[D]")
        dep = dep_parsed[mask_valid].dt.date.values.astype("datetime64[D]")
        result.loc[mask_valid] = np.busday_count(pay, dep)

    return result

def reconcile_stream(
    stream_name: str,
    payroll_df: pd.DataFrame,
    rk_df: pd.DataFrame,
    payroll_cols: dict[str, str],
    rk_cols: dict[str, str],
    required_keys: list[str],
    write_outputs: bool = True,
    aggregate_by_employee: bool = True,
):
    """
    Generic reconciliation function for a given stream, e.g. "deferrals" or "loans".

    required_keys is the list of logical keys needed on both sides
    (e.g. ["employee_id", "def_amount"] or ["employee_id", "loan_amount"]).

    aggregate_by_employee=True:
      - Aggregates to one row per employee in each file before reconciling.
      - Collapses money-type splits and duplicate rows.
    """

    # Validate required logical keys exist
    missing_payroll = [k for k in required_keys if k not in payroll_cols]
    missing_rk = [k for k in required_keys if k not in rk_cols]

    if missing_payroll or missing_rk:
        print(
            f"\n[WARN] Stream '{stream_name}': missing columns. "
            f"Payroll missing: {missing_payroll}, RK missing: {missing_rk}. Skipping this stream."
        )
        return None

    # Base columns
    p_id = payroll_df[payroll_cols["employee_id"]]
    r_id = rk_df[rk_cols["employee_id"]]

    # For deferrals, calculate total deferral (pretax + roth) directly from normalized columns
    if stream_name.lower() == "deferrals":
        # Payroll side: total deferral = EE Deferral $ + EE Roth $ (if exists)
        payroll_total_deferral = pd.Series(0.0, index=payroll_df.index)
        if "EE Deferral $" in payroll_df.columns:
            payroll_total_deferral += parse_amount(payroll_df["EE Deferral $"])
        if "EE Roth $" in payroll_df.columns:
            payroll_total_deferral += parse_amount(payroll_df["EE Roth $"])
        
        # RK side: total deferral = EE Deferral $ + EE Roth $ (if exists)
        rk_total_deferral = pd.Series(0.0, index=rk_df.index)
        if "EE Deferral $" in rk_df.columns:
            rk_total_deferral += parse_amount(rk_df["EE Deferral $"])
        if "EE Roth $" in rk_df.columns:
            rk_total_deferral += parse_amount(rk_df["EE Roth $"])
        
        p_amt = payroll_total_deferral
        r_amt = rk_total_deferral
    else:
        # For loans or other streams, use the column mapping as before
        p_amt = parse_amount(payroll_df[payroll_cols[required_keys[1]]])
        r_amt = parse_amount(rk_df[rk_cols[required_keys[1]]])

    # Attach dates if available
    p_date = payroll_df[payroll_cols["pay_date"]] if "pay_date" in payroll_cols else None
    r_date = rk_df[rk_cols["deposit_date"]] if "deposit_date" in rk_cols else None

    payroll_norm = pd.DataFrame({"employee_id": p_id, "amount": p_amt})
    rk_norm = pd.DataFrame({"employee_id": r_id, "amount": r_amt})

    if p_date is not None:
        payroll_norm["pay_date"] = p_date
    if r_date is not None:
        rk_norm["deposit_date"] = r_date

    # Heuristic: pull first/last name off payroll if available
    lower_map_p = {c.lower(): c for c in payroll_df.columns}
    first_col = next((lower_map_p[c] for c in lower_map_p if "first" in c), None)
    last_col = next((lower_map_p[c] for c in lower_map_p if "last" in c), None)

    if first_col:
        payroll_norm["first_name"] = payroll_df[first_col]
    if last_col:
        payroll_norm["last_name"] = payroll_df[last_col]

    # === Aggregation layer ===
    if aggregate_by_employee:
        agg_p = {"amount": "sum"}
        if "pay_date" in payroll_norm.columns:
            agg_p["pay_date"] = "min"
        if "first_name" in payroll_norm.columns:
            agg_p["first_name"] = "first"
        if "last_name" in payroll_norm.columns:
            agg_p["last_name"] = "first"

        agg_r = {"amount": "sum"}
        if "deposit_date" in rk_norm.columns:
            agg_r["deposit_date"] = "min"

        payroll_norm = payroll_norm.groupby("employee_id", as_index=False).agg(agg_p)
        rk_norm = rk_norm.groupby("employee_id", as_index=False).agg(agg_r)

    # === Merge + mismatch logic ===
    merged = payroll_norm.merge(
        rk_norm,
        on="employee_id",
        how="outer",
        suffixes=("_payroll", "_rk"),
        indicator=True,
    )

    only_in_payroll = merged[merged["_merge"] == "left_only"].copy()
    only_in_rk = merged[merged["_merge"] == "right_only"].copy()
    amount_mismatch = merged[
        (merged["_merge"] == "both")
        & (merged["amount_payroll"].fillna(0) != merged["amount_rk"].fillna(0))
    ].copy()

    # Delta column for mismatches
    if not amount_mismatch.empty:
        amount_mismatch["delta"] = (
            amount_mismatch["amount_payroll"].fillna(0)
            - amount_mismatch["amount_rk"].fillna(0)
        )

    print(f"\n=== {stream_name.upper()} Reconciliation Summary ===")
    print(f"Total in payroll ({stream_name}):      {len(payroll_norm):>4}")
    print(f"Total in recordkeeper ({stream_name}): {len(rk_norm):>4}")
    print(f"Only in payroll ({stream_name}):       {len(only_in_payroll):>4}")
    print(f"Only in recordkeeper ({stream_name}):  {len(only_in_rk):>4}")
    print(f"Amount mismatches ({stream_name}):     {len(amount_mismatch):>4}")

    # Ensure output folder exists
    DATA_OUT.mkdir(exist_ok=True, parents=True)

    if write_outputs:
        base = stream_name.lower()
        only_in_payroll.to_csv(DATA_OUT / f"only_in_payroll_{base}.csv", index=False)
        only_in_rk.to_csv(DATA_OUT / f"only_in_recordkeeper_{base}.csv", index=False)

        if base == "deferrals":
            mismatch_filename = "deferral_mismatches.csv"
        elif base == "loans":
            mismatch_filename = "loan_mismatches.csv"
        else:
            mismatch_filename = f"{base}_mismatch.csv"

        amount_mismatch.to_csv(DATA_OUT / mismatch_filename, index=False)

    # Late funding detection if we have dates
    if "pay_date" in merged.columns and "deposit_date" in merged.columns:
        lag = compute_business_days_lag(merged, "pay_date", "deposit_date")
        merged["business_days_lag"] = lag

        late_mask = lag > MAX_BUSINESS_DAYS_LAG
        late_df = merged[late_mask].copy()

        if not late_df.empty:
            late_path = DATA_OUT / f"late_{stream_name.lower()}_contributions.csv"
            late_df.to_csv(late_path, index=False)
            print(
                f"Late {stream_name} contributions: {len(late_df)} rows "
                f"(> {MAX_BUSINESS_DAYS_LAG} business days). Written to: {late_path}"
            )
        else:
            print(
                f"No late {stream_name} contributions detected "
                f"(threshold = {MAX_BUSINESS_DAYS_LAG} business days)."
            )
    else:
        print(f"No usable dates for late {stream_name} detection.")




# =========================
# MAIN ORCHESTRATION
# =========================

def reconcile_payroll_vs_recordkeeper():
    # Load config and override defaults if present
    cfg = load_config()
    global MAX_BUSINESS_DAYS_LAG

    payroll_file = cfg.get("payroll_file", PAYROLL_FILE)
    rk_file = cfg.get("recordkeeper_file", RECORDKEEPER_FILE)
    MAX_BUSINESS_DAYS_LAG = cfg.get("max_business_days_lag", MAX_BUSINESS_DAYS_LAG)

    # Load raw files (normalization happens inside load_csv)
    payroll_df = load_csv(payroll_file)
    rk_df = load_csv(rk_file)

    # Detect vendors with confidence
    payroll_vendor, payroll_confidence = detect_vendor_with_confidence(
        payroll_df, PAYROLL_VENDOR_SIGNATURES, None
    )
    rk_vendor, rk_confidence = detect_vendor_with_confidence(
        rk_df, RK_VENDOR_SIGNATURES, None
    )
    
    # Apply vendor-specific column mapping
    payroll_df = apply_vendor_column_mapping(payroll_df, payroll_vendor, PAYROLL_VENDOR_SIGNATURES)
    rk_df = apply_vendor_column_mapping(rk_df, rk_vendor, RK_VENDOR_SIGNATURES)

    print("\n=== Vendor Detection ===")
    print(f"Detected payroll vendor:     {payroll_vendor or 'Unknown / Generic'}")
    print(f"Detected recordkeeper:       {rk_vendor or 'Unknown / Generic'}")

    # Vendor-aware mapping: strict for vendor-required fields, flexible elsewhere,
    # with a clean fallback to the generic COLUMN_MAP.
    payroll_cols = infer_vendor_mapping(
        payroll_df,
        payroll_vendor,
        PAYROLL_VENDOR_COLUMN_MAPS,
        COLUMN_MAP,
    )
    rk_cols = infer_vendor_mapping(
        rk_df,
        rk_vendor,
        RK_VENDOR_COLUMN_MAPS,
        COLUMN_MAP,
    )

    print("\n=== Column Mapping (Payroll) ===")
    for k, v in payroll_cols.items():
        print(f"  {k} -> {v}")
    print("\n=== Column Mapping (Recordkeeper) ===")
    for k, v in rk_cols.items():
        print(f"  {k} -> {v}")

    # Ensure we at least know employee_id
    if "employee_id" not in payroll_cols or "employee_id" not in rk_cols:
        raise ValueError(
            f"Cannot proceed: employee_id not found on both sides. "
            f"Payroll columns: {list(payroll_df.columns)}, RK columns: {list(rk_df.columns)}"
        )

    # Ensure pay/deposit dates map if present
    # Not required, but used for late logic
    if "pay_date" not in payroll_cols:
        print("[WARN] No pay_date mapped on payroll file; late logic will be limited.")
    if "deposit_date" not in rk_cols:
        print("[WARN] No deposit_date mapped on recordkeeper file; late logic will be limited.")

    # Build synthetic logical keys for streams:
    #  - deferrals: sum of pretax + roth
    #  - loans: loan column
    # To leverage reconcile_stream, we materialize these as virtual columns in temporary frames.

    # Copy dataframes so we don't mutate originals
    p = payroll_df.copy()
    r = rk_df.copy()

    # DEFERRALS
    # Payroll side - Calculate total deferral (pretax + roth) when available
    p_def = None
    payroll_pretax_col = payroll_cols.get("payroll_pretax")
    payroll_roth_col = payroll_cols.get("payroll_roth")
    payroll_amount_col = payroll_cols.get("amount")
    
    # Check for normalized column names directly if not mapped
    if not payroll_pretax_col and "EE Deferral $" in p.columns:
        payroll_pretax_col = "EE Deferral $"
    if not payroll_roth_col and "EE Roth $" in p.columns:
        payroll_roth_col = "EE Roth $"
    
    if payroll_pretax_col or payroll_roth_col or payroll_amount_col:
        pretax = parse_amount(p[payroll_pretax_col]) if payroll_pretax_col else 0.0
        roth = parse_amount(p[payroll_roth_col]) if payroll_roth_col else 0.0
        # fallback: single amount column if no pretax/roth split
        if not payroll_pretax_col and not payroll_roth_col and payroll_amount_col:
            pretax = parse_amount(p[payroll_amount_col])
            roth = 0.0
        # Total deferral = pretax + roth (use both if available)
        p_def = pretax + roth
        p["deferral_amount"] = p_def

    # RK side - Calculate total deferral (pretax + roth) when available
    r_def = None
    rk_pretax_col = rk_cols.get("rk_pretax")
    rk_roth_col = rk_cols.get("rk_roth")
    rk_amount_col = rk_cols.get("amount")
    
    # Check for normalized column names directly if not mapped
    if not rk_pretax_col and "EE Deferral $" in r.columns:
        rk_pretax_col = "EE Deferral $"
    if not rk_roth_col and "EE Roth $" in r.columns:
        rk_roth_col = "EE Roth $"
    
    if rk_pretax_col or rk_roth_col or rk_amount_col:
        pretax_rk = parse_amount(r[rk_pretax_col]) if rk_pretax_col else 0.0
        roth_rk = parse_amount(r[rk_roth_col]) if rk_roth_col else 0.0
        if not rk_pretax_col and not rk_roth_col and rk_amount_col:
            pretax_rk = parse_amount(r[rk_amount_col])
            roth_rk = 0.0
        # Total deferral = pretax_rk + roth_rk (use both if available)
        r_def = pretax_rk + roth_rk
        r["deferral_amount"] = r_def

        # LOANS
    if "payroll_loan" in payroll_cols:
        p["loan_amount"] = parse_amount(p[payroll_cols["payroll_loan"]])

    if "rk_loan" in rk_cols:
        r["loan_amount"] = parse_amount(r[rk_cols["rk_loan"]])
    elif "payroll_loan" in payroll_cols:
        # Payroll has loans, RK has no explicit loan column -> treat RK as zero loans
        r["loan_amount"] = 0.0


    # Build extended mapping dicts with virtual keys
    payroll_cols_ext = payroll_cols.copy()
    rk_cols_ext = rk_cols.copy()

    if "deferral_amount" in p.columns:
        payroll_cols_ext["def_amount"] = "deferral_amount"
    if "deferral_amount" in r.columns:
        rk_cols_ext["def_amount"] = "deferral_amount"

    if "loan_amount" in p.columns:
        payroll_cols_ext["loan_amount"] = "loan_amount"
    if "loan_amount" in r.columns:
        rk_cols_ext["loan_amount"] = "loan_amount"

    # Map pay/deposit dates into logical keys if present
    # (For late logic inside reconcile_stream)
    if "pay_date" in payroll_cols_ext:
        pass  # already mapped
    else:
        # try to map if payroll has something date-like but not in COLUMN_MAP (unlikely but defensive)
        pass

    if "deposit_date" in rk_cols_ext:
        pass
    else:
        pass

    reconcile_stream(
        stream_name="deferrals",
        payroll_df=p,
        rk_df=r,
        payroll_cols=payroll_cols_ext,
        rk_cols=rk_cols_ext,
        required_keys=["employee_id", "def_amount"],
        aggregate_by_employee=True,
    )

    reconcile_stream(
        stream_name="loans",
        payroll_df=p,
        rk_df=r,
        payroll_cols=payroll_cols_ext,
        rk_cols=rk_cols_ext,
        required_keys=["employee_id", "loan_amount"],
        aggregate_by_employee=True,
    )



    
def generate_excel_report():
    """
    Build a consolidated Excel report from whatever CSVs exist in data/processed.
    Sheets:
      - Summary (raw category counts)
      - Summary_Compliance_Metrics (advisor-style metrics)
      - Deferrals/Loans detail sheets (mismatches, only-in-*, late, etc.)
    """
    report_path = DATA_OUT / "reconciliation_report.xlsx"

    # Ensure output dir exists
    DATA_OUT.mkdir(exist_ok=True, parents=True)

    streams = ["deferrals", "loans"]
    summary_rows: list[dict] = []

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        # Detail + raw summary rows
        for stream in streams:
            base = stream.lower()

            # Align mismatch filenames with what reconcile_stream writes
            if base == "deferrals":
                mismatch_file = DATA_OUT / "deferral_mismatches.csv"
            elif base == "loans":
                mismatch_file = DATA_OUT / "loan_mismatches.csv"
            else:
                mismatch_file = DATA_OUT / f"{base}_mismatch.csv"

            files = {
                "only_in_payroll":      DATA_OUT / f"only_in_payroll_{base}.csv",
                "only_in_recordkeeper": DATA_OUT / f"only_in_recordkeeper_{base}.csv",
                "mismatch":             mismatch_file,
                "late":                 DATA_OUT / f"late_{base}_contributions.csv",
            }

            for label, path in files.items():
                df = safe_read_csv(path)
                sheet_name = f"{stream[:3].title()} - {label.replace('_', ' ').title()}"

                if df is not None and not df.empty:
                    # Write detailed sheet
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

                    # Add summary row
                    summary_rows.append(
                        {
                            "stream": stream,
                            "category": label,
                            "rows": len(df),
                        }
                    )
                else:
                    # Write placeholder sheet so structure is predictable
                    placeholder = pd.DataFrame(
                        [{"info": f"No rows for {stream}/{label} or file missing"}]
                    )
                    placeholder.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    summary_rows.append(
                        {
                            "stream": stream,
                            "category": label,
                            "rows": 0,
                        }
                    )

        # Sheet 1: raw summary counts
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: compliance metrics
        metrics_rows, _ = compute_compliance_metrics()
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_excel(
            writer, sheet_name="Summary_Compliance_Metrics", index=False
        )

    # Console dashboard
    _, console_lines = compute_compliance_metrics()
    print("\n" + "\n".join(console_lines))
    print(f"\nConsolidated Excel report written to: {report_path}")
    print("You can drag this into Google Sheets or email it as-is.")

    # ====================================================
    # Build outputs map for proof manifest
    # ====================================================
    outputs: dict = {}

    def add_output(logical_name: str, filename: str):
        path = DATA_OUT / filename
        if path.exists():
            entry = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            if path.suffix == ".csv":
                entry.update(hash_csv_rows(path))
            outputs[logical_name] = entry
        else:
            outputs[logical_name] = {"missing": True}

    add_output("deferral_mismatches", "deferral_mismatches.csv")
    add_output("loan_mismatches", "loan_mismatches.csv")
    add_output("late_deferrals", "late_deferrals_contributions.csv")
    add_output("late_loans", "late_loans_contributions.csv")
    add_output("excel_report", "reconciliation_report.xlsx")

    return outputs

def main():
    print("Running Prooflink reconciliation...")
    reconcile_payroll_vs_recordkeeper()
    outputs = generate_excel_report()
    cfg = load_config()

    payroll_path = DATA_RAW / cfg.get("payroll_file", PAYROLL_FILE)
    rk_path = DATA_RAW / cfg.get("recordkeeper_file", RECORDKEEPER_FILE)

    write_proof_manifest(
        payroll_file=payroll_path,
        rk_file=rk_path,
        cfg=cfg,
        outputs=outputs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--payroll_csv", required=True)
    parser.add_argument("--rk_csv", required=True)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--proofs_dir", default="proofs")
    args = parser.parse_args()

    run_reconciliation(
        payroll_csv=args.payroll_csv,
        rk_csv=args.rk_csv,
        output_dir=args.output_dir,
        proofs_dir=args.proofs_dir,
    )


    main()


    

