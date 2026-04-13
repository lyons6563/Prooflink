import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import re
from typing import Optional, Dict, Any, List
import sys
import io
import traceback
from datetime import datetime
import json
import requests
import os
import uuid

# Import engine for direct mode
from main import run_prooflink_engine, EngineConfig, EngineResult

# Import preflight for validation
from preflight import run_preflight

# API client configuration
API_BASE_URL = os.getenv("PROOFLINK_API_URL")
USE_API_BACKEND = bool(API_BASE_URL)

# Simple local dev password (use environment variable for production)
APP_DEV_PASSWORD = os.getenv("APP_DEV_PASSWORD", "prooflink")


def api_create_run(
    payroll_bytes: bytes,
    payroll_filename: str,
    rk_bytes: bytes,
    rk_filename: str,
    plan_name: str,
    plan_rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call the ProofLink backend API to create a run.

    Returns a dict with at least:
    - run_id (str)
    - summary (dict)
    - status (str)
    - evidence_pack_available (bool)
    """
    files = {
        "payroll_file": (payroll_filename, payroll_bytes, "text/csv"),
        "rk_file": (rk_filename, rk_bytes, "text/csv"),
    }
    data = {"plan_name": plan_name or "Untitled Plan"}
    
    # Add plan_rules to data if provided
    if plan_rules:
        # Convert plan_rules dict to JSON string for form data
        data["plan_rules"] = json.dumps(plan_rules)

    resp = requests.post(f"{API_BASE_URL}/api/v1/runs", files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


def create_run(
    payroll_bytes: bytes,
    payroll_filename: str,
    rk_bytes: bytes,
    rk_filename: str,
    plan_name: str,
    plan_rules: Optional[Dict[str, Any]] = None,
    payroll_vendor_hint: Optional[str] = None,
    rk_vendor_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a reconciliation run either via API or directly.
    
    Returns a dict with:
    - run_id (str)
    - summary (dict)
    - status (str)
    - evidence_pack_available (bool)
    """
    if USE_API_BACKEND:
        # Use API backend
        return api_create_run(
            payroll_bytes=payroll_bytes,
            payroll_filename=payroll_filename,
            rk_bytes=rk_bytes,
            rk_filename=rk_filename,
            plan_name=plan_name,
            plan_rules=plan_rules,
        )
    else:
        # Run engine directly
        run_id = str(uuid.uuid4())
        run_dir = Path("streamlit_runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        payroll_path = run_dir / (payroll_filename or "payroll.csv")
        rk_path = run_dir / (rk_filename or "rk.csv")
        
        with open(payroll_path, "wb") as f:
            f.write(payroll_bytes)
        with open(rk_path, "wb") as f:
            f.write(rk_bytes)
        
        # Create engine config
        config = EngineConfig(
            plan_name=plan_name or "Untitled Plan",
            payroll_vendor_hint=payroll_vendor_hint,
            rk_vendor_hint=rk_vendor_hint,
            output_dir=str(run_dir / "output"),
            proofs_dir=str(run_dir / "proofs"),
        )
        
        # Run the engine
        engine_result: EngineResult = run_prooflink_engine(
            payroll_path=str(payroll_path),
            rk_path=str(rk_path),
            config=config,
            run_id=run_id,
            plan_rules=plan_rules,
        )
        
        # Store in session state for later retrieval
        st.session_state["current_manifest"] = engine_result.manifest
        
        # Convert EngineResult to API-compatible format
        return {
            "run_id": engine_result.run_id,
            "summary": engine_result.summary,
            "status": "completed",
            "evidence_pack_available": bool(engine_result.evidence_pack_path and Path(engine_result.evidence_pack_path).exists()),
        }


def api_get_run(run_id: str) -> Dict[str, Any]:
    """Get run details from the API."""
    if not USE_API_BACKEND:
        # In direct mode, return from session state if available
        if "current_run_id" in st.session_state and st.session_state["current_run_id"] == run_id:
            summary = st.session_state.get("current_summary", {})
            return {
                "run_id": run_id,
                "summary": summary,
                "manifest": st.session_state.get("current_manifest", {}),
            }
        # Otherwise return empty (run history not supported in direct mode)
        return {"run_id": run_id, "summary": {}, "manifest": {}}
    
    resp = requests.get(f"{API_BASE_URL}/api/v1/runs/{run_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_download_evidence_pack(run_id: str) -> Optional[bytes]:
    """Download evidence pack ZIP from the API or filesystem."""
    if not USE_API_BACKEND:
        # In direct mode, read from filesystem
        run_dir = Path("streamlit_runs") / run_id
        evidence_pack_path = run_dir / "output" / "prooflink_evidence_pack.zip"
        if evidence_pack_path.exists():
            with open(evidence_pack_path, "rb") as f:
                return f.read()
        return None
    
    resp = requests.get(
        f"{API_BASE_URL}/api/v1/runs/{run_id}/evidence-pack",
        timeout=60,
        stream=True,
    )
    if resp.status_code != 200:
        return None
    return resp.content


def api_list_runs(limit: int = 50) -> Dict[str, Any]:
    """
    Call the ProofLink backend API to list recent runs, or return current run in direct mode.

    Returns a dict like:
    {
        "count": int,
        "items": [
            {
                "run_id": str,
                "status": str,
                "plan_name": str,
                "created_at": str,
                "updated_at": str,
                "payroll_filename": str,
                "rk_filename": str,
                "has_evidence_pack": bool,
                "summary": dict,
            },
            ...
        ],
    }
    """
    if not USE_API_BACKEND:
        # In direct mode, return current run from session state if available
        if "current_run_id" in st.session_state:
            run_id = st.session_state["current_run_id"]
            summary = st.session_state.get("current_summary", {})
            return {
                "count": 1,
                "items": [
                    {
                        "run_id": run_id,
                        "status": st.session_state.get("current_status", "completed"),
                        "plan_name": summary.get("plan_name", "Unknown Plan"),
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "payroll_filename": "uploaded.csv",
                        "rk_filename": "uploaded.csv",
                        "has_evidence_pack": bool(st.session_state.get("current_summary", {}).get("evidence_pack_path")),
                        "summary": summary,
                    }
                ],
            }
        return {"count": 0, "items": []}
    
    params = {"limit": limit}
    resp = requests.get(f"{API_BASE_URL}/api/v1/runs", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def format_timing_risk_badge(timing_risk: str) -> str:
    """
    Map raw timing risk level to a badge string with emoji.

    Expected values: "High", "Medium", "Low", or None/other.
    """
    if not timing_risk:
        return "⚪ N/A"

    level = timing_risk.strip().lower()
    if level == "high":
        return "🔴 High"
    if level == "medium":
        return "🟡 Medium"
    if level == "low":
        return "🟢 Low"
    return "⚪ N/A"


def check_password() -> bool:
    """Simple local dev password gate."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.sidebar.header("Login")
        password = st.sidebar.text_input("Password", type="password")

        if password:
            if password == APP_DEV_PASSWORD:
                st.session_state["password_correct"] = True
            else:
                st.sidebar.error("Incorrect password")
                st.session_state["password_correct"] = False

    return st.session_state["password_correct"]


def validate_recordkeeper_headers(rk_file) -> tuple:
    """
    Validate that the recordkeeper CSV contains expected amount fields.
    
    Returns:
        (is_valid, error_message)
    """
    if rk_file is None:
        return True, ""  # Will be caught by other validation
    
    try:
        # Read only the header row
        rk_file.seek(0)
        header_line = rk_file.readline().decode('utf-8').strip()
        rk_file.seek(0)  # Reset for later use
        
        # Parse CSV header (handle quoted fields)
        import csv
        reader = csv.reader([header_line])
        headers = next(reader)
        
        # Normalize headers (lowercase, strip whitespace, remove special chars for comparison)
        headers_normalized = [h.lower().strip().replace(" ", "_").replace("$", "").replace("-", "_") for h in headers]
        
        # Check for at least one of the expected amount fields
        # Looking for: ee_deferral, ee_roth, loan_amount (or common variations)
        # Common RK variations: ee_pretax, ee_roth, loan_contr, def_amount, roth_amount
        expected_fields = [
            "ee_deferral", "ee_pretax", "ee_roth", 
            "loan_amount", "loan_contr", "loan_payment",
            "def_amount", "roth_amount"  # Also accept canonical names
        ]
        
        found = any(
            any(field in h for h in headers_normalized)
            for field in expected_fields
        )
        
        if not found:
            return False, (
                "Recordkeeper file is missing expected amount fields "
                "(ee_deferral, ee_roth, loan_amount). "
                "You likely uploaded a payroll file in the recordkeeper slot. "
                "Please upload a recordkeeper export."
            )
        
        return True, ""
    except Exception as e:
        # If validation fails due to parsing error, allow it through
        # (preflight will catch actual format issues)
        return True, ""


# ---------- Paths / Directories ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROOFS_DIR = BASE_DIR.parent / "proofs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROOFS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Page Config ----------
st.set_page_config(page_title="ProofLink", layout="wide")

if not check_password():
    st.stop()


# ==============================
#  Contribution Timing TAB
# ==============================

# DEPRECATED: This function is no longer used - replaced with API calls
# Keeping for backward compatibility if needed elsewhere
# def run_reconciliation_with_stdout_capture(
#     payroll_file, rk_file, plan_name: str = "Demo Plan",
#     payroll_vendor_hint: Optional[str] = None,
#     rk_vendor_hint: Optional[str] = None,
#     output_dir: Path = None
# ) -> tuple[bool, str, dict]:
#     """
#     Run reconciliation and capture stdout.
#     
#     DEPRECATED: Use api_create_run() instead.
#     """
#     pass


def load_run_history() -> list[dict]:
    """
    DEPRECATED: Load run history from run_summary.json files in run folders.
    
    This function is no longer used. Run history is now loaded from the API
    via api_list_runs() instead of scanning the filesystem.
    
    Returns a list of dicts with run information, sorted by timestamp descending.
    """
    processed_dir = BASE_DIR.parent / "data" / "processed"
    
    if not processed_dir.exists():
        return []
    
    runs = []
    
    # Iterate over subfolders starting with "run_"
    for run_folder in processed_dir.iterdir():
        if not run_folder.is_dir() or not run_folder.name.startswith("run_"):
            continue
        
        run_summary_path = run_folder / "run_summary.json"
        if not run_summary_path.exists():
            continue
        
        try:
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            
            # Extract timestamp from folder name (e.g., "run_20251127_163259" -> "20251127_163259")
            timestamp_str = run_folder.name.replace("run_", "")
            
            # Calculate total issues
            deferral_mismatches = summary_data.get("deferral_mismatch_count", 0)
            loan_mismatches = summary_data.get("loan_mismatch_count", 0)
            late_deferrals = summary_data.get("late_deferral_count", 0)
            total_issues = deferral_mismatches + loan_mismatches + late_deferrals
            
            # Compute risk level using same thresholds as compute_run_risk_level
            if total_issues == 0:
                risk_label, risk_color = ("Low", "green")
            elif 1 <= total_issues <= 20:
                risk_label, risk_color = ("Medium", "orange")
            else:
                risk_label, risk_color = ("High", "red")
            
            run_info = {
                "run_folder": run_folder.name,
                "timestamp": timestamp_str,
                "plan_name": summary_data.get("plan_name", "Unknown"),
                "payroll_vendor": summary_data.get("payroll_vendor", "Unknown"),
                "rk_vendor": summary_data.get("rk_vendor", "Unknown"),
                "deferral_mismatch_count": deferral_mismatches,
                "loan_mismatch_count": loan_mismatches,
                "late_deferral_count": late_deferrals,
                "total_issues": total_issues,
                "risk_label": risk_label,
                "risk_color": risk_color,
                "evidence_pack_path": summary_data.get("evidence_pack_path", ""),
            }
            
            runs.append(run_info)
        except Exception as e:
            # Skip runs with invalid JSON or other errors
            continue
    
    # Sort by timestamp descending (most recent first)
    runs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return runs


def compute_run_risk_level(summary: Dict[str, Any], results_dict: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    """
    Compute run risk level and label from the summary dict.

    Uses:
    - summary["timing_metrics"]:
        - "late_rows"
        - "missing_deposits"
        - "timing_risk"
    - summary["deferral_mismatch_count"]
    - summary["loan_mismatch_count"]
    - summary["late_deferral_count"]
    """
    timing_metrics = summary.get("timing_metrics") or {}
    timing_risk = (timing_metrics.get("timing_risk") or "").lower()
    missing_deposits = timing_metrics.get("missing_deposits", 0) or 0
    late_rows = timing_metrics.get("late_rows", 0) or 0

    deferral_mismatches = summary.get("deferral_mismatch_count", 0) or 0
    loan_mismatches = summary.get("loan_mismatch_count", 0) or 0
    late_deferral_count = summary.get("late_deferral_count", 0) or 0

    # Default
    risk_icon = ":green_circle:"
    risk_label = "Low"

    # HIGH RISK:
    # - any missing deposits
    # - or explicit timing_risk "high"
    if missing_deposits > 0 or timing_risk == "high":
        risk_icon = ":red_circle:"
        risk_label = "High"
    # MEDIUM RISK:
    # - late rows or mismatches or late deferrals, but no missing deposits/high flag
    elif late_rows > 0 or deferral_mismatches > 0 or loan_mismatches > 0 or late_deferral_count > 0:
        risk_icon = ":orange_circle:"
        risk_label = "Medium"

    return risk_icon, risk_label


def parse_reconciliation_output(stdout: str) -> dict:
    """
    Parse stdout from main.py reconciliation to extract key metrics.
    
    Returns a dict with keys:
    - total_payroll_deferrals: int or None
    - total_rk_deferrals: int or None
    - deferral_mismatches: int or None
    - loan_mismatches: int or None
    - late_deferral_rows: int or None
    - total_payroll_loans: int or None
    - total_rk_loans: int or None
    - late_loan_rows: int or None
    """
    result = {
        "total_payroll_deferrals": None,
        "total_rk_deferrals": None,
        "deferral_mismatches": None,
        "loan_mismatches": None,
        "late_deferral_rows": None,
        "total_payroll_loans": None,
        "total_rk_loans": None,
        "late_loan_rows": None,
    }
    
    if not stdout:
        return result
    
    # Parse DEFERRALS reconciliation summary
    # Format: "Total in payroll (deferrals):      {count}"
    def_match = re.search(r"Total in payroll \(deferrals\):\s+(\d+)", stdout)
    if def_match:
        result["total_payroll_deferrals"] = int(def_match.group(1))
    
    def_rk_match = re.search(r"Total in recordkeeper \(deferrals\):\s+(\d+)", stdout)
    if def_rk_match:
        result["total_rk_deferrals"] = int(def_rk_match.group(1))
    
    def_mismatch_match = re.search(r"Amount mismatches \(deferrals\):\s+(\d+)", stdout)
    if def_mismatch_match:
        result["deferral_mismatches"] = int(def_mismatch_match.group(1))
    
    # Parse LOANS reconciliation summary
    loan_match = re.search(r"Total in payroll \(loans\):\s+(\d+)", stdout)
    if loan_match:
        result["total_payroll_loans"] = int(loan_match.group(1))
    
    loan_rk_match = re.search(r"Total in recordkeeper \(loans\):\s+(\d+)", stdout)
    if loan_rk_match:
        result["total_rk_loans"] = int(loan_rk_match.group(1))
    
    loan_mismatch_match = re.search(r"Amount mismatches \(loans\):\s+(\d+)", stdout)
    if loan_mismatch_match:
        result["loan_mismatches"] = int(loan_mismatch_match.group(1))
    
    # Parse late contributions
    late_def_match = re.search(r"Late deferrals contributions:\s+(\d+)\s+rows", stdout)
    if late_def_match:
        result["late_deferral_rows"] = int(late_def_match.group(1))
    
    late_loan_match = re.search(r"Late loans contributions:\s+(\d+)\s+rows", stdout)
    if late_loan_match:
        result["late_loan_rows"] = int(late_loan_match.group(1))
    
    return result


def parse_analyzer_output(stdout: str) -> dict:
    """
    Parse stdout from contribution_timing_analyzer_v2.py to extract key metrics.
    
    Returns a dict with keys:
    - payroll_vendor: str or None
    - rk_vendor: str or None
    - total_rows: int or None
    - late_contributions: int or None
    - missing_deposits: int or None
    - late_threshold: int or None
    - timing_risk: str or None
    """
    result = {
        "payroll_vendor": None,
        "rk_vendor": None,
        "total_rows": None,
        "late_contributions": None,
        "missing_deposits": None,
        "late_threshold": None,
        "timing_risk": None,
    }
    
    if not stdout:
        return result
    
    # Parse vendor detection section
    vendor_match = re.search(r"Detected payroll vendor:\s+(\S+)", stdout)
    if vendor_match:
        result["payroll_vendor"] = vendor_match.group(1)
    
    rk_vendor_match = re.search(r"Detected recordkeeper:\s+(\S+)", stdout)
    if rk_vendor_match:
        result["rk_vendor"] = rk_vendor_match.group(1)
    
    # Parse contribution timing summary
    total_rows_match = re.search(r"Total payroll rows analyzed:\s+(\d+)", stdout)
    if total_rows_match:
        result["total_rows"] = int(total_rows_match.group(1))
    
    # Parse late contributions with threshold
    late_match = re.search(r"Late contributions \(>\s+(\d+)\s+days\):\s+(\d+)", stdout)
    if late_match:
        result["late_threshold"] = int(late_match.group(1))
        result["late_contributions"] = int(late_match.group(2))
    
    # Parse missing deposits
    missing_match = re.search(r"Missing deposit rows:\s+(\d+)", stdout)
    if missing_match:
        result["missing_deposits"] = int(missing_match.group(1))
    
    # Parse timing risk
    timing_risk_match = re.search(r"Timing Risk:\s+(\S+)", stdout)
    if timing_risk_match:
        result["timing_risk"] = timing_risk_match.group(1)
    
    return result


def run_prooflink_analysis(payroll_file, rk_file, output_dir: Path = None, late_threshold: int = 5) -> tuple[bool, str, str]:
    """
    Run the contribution timing analyzer via CLI subprocess.

    Args:
        payroll_file: Streamlit uploaded file for payroll CSV
        rk_file: Streamlit uploaded file for recordkeeper CSV
        output_dir: Directory for output files (defaults to data/processed)
        late_threshold: Number of days threshold for late contributions (default: 5)

    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    if output_dir is None:
        output_dir = BASE_DIR.parent / "data" / "processed"

    # Create temp directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save uploaded files to temp location
        payroll_temp_path = temp_path / "payroll_uploaded.csv"
        rk_temp_path = temp_path / "rk_uploaded.csv"

        with open(payroll_temp_path, "wb") as f:
            f.write(payroll_file.getbuffer())

        with open(rk_temp_path, "wb") as f:
            f.write(rk_file.getbuffer())

        # Build CLI command
        cmd = [
            sys.executable, str(BASE_DIR / "contribution_timing_analyzer_v2.py"),
            str(payroll_temp_path),
            str(rk_temp_path),
            "--late-threshold", str(late_threshold),
            "--output-dir", str(output_dir)
        ]

        try:
            # Run the CLI command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=BASE_DIR
            )

            return result.returncode == 0, result.stdout, result.stderr

        except Exception as e:
            return False, "", f"Exception running analysis: {str(e)}"


def render_contribution_timing_tab():
    st.subheader("Contribution Timing Analysis")
    
    summary = st.session_state.get("current_summary") or {}
    timing_metrics = summary.get("timing_metrics") or {}
    
    if not timing_metrics:
        st.info("No timing metrics available for this run. Run a new analysis in the Reconciliation tab.")
    else:
        timing_risk = timing_metrics.get("timing_risk", "N/A")
        total_rows = timing_metrics.get("total_rows", 0)
        late_rows = timing_metrics.get("late_rows", 0)
        missing_deposits = timing_metrics.get("missing_deposits", 0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Timing Risk", timing_risk)
        col2.metric("Total Rows Analyzed", total_rows)
        col3.metric("Late Rows", late_rows)
        col4.metric("Missing Deposits", missing_deposits)
        
        with st.expander("Raw timing metrics"):
            st.json(timing_metrics)


def classify_run_risk(summary: Dict[str, Any]) -> str:
    """
    Simple risk classifier based on mismatch + late counts.
    We'll upgrade later with dollar thresholds / percentages.
    """
    if summary.get("late_deferral_count", 0) > 0:
        return "High"
    if summary.get("deferral_mismatch_count", 0) > 0 or summary.get("loan_mismatch_count", 0) > 0:
        return "Medium"
    return "Low"


def render_run_summary(summary: Dict[str, Any]) -> None:
    st.subheader("Run Summary")

    # Use compute_run_risk_level for consistency
    results_dict = {}  # Empty results_dict for this function
    risk_icon, risk_label = compute_run_risk_level(summary, results_dict)
    st.markdown(f"**Run Risk Level:** {risk_icon} {risk_label}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Plan", summary.get("plan_name", "Unknown"))
        st.text(f"Payroll vendor: {summary.get('payroll_vendor', 'Unknown')}")
        st.text(f"Recordkeeper: {summary.get('rk_vendor', 'Unknown')}")

    with col2:
        st.metric(
            "Deferrals (Payroll)",
            f"{summary.get('total_deferrals_payroll', 0.0):,.2f}",
        )
        st.metric(
            "Deferrals (RK)",
            f"{summary.get('total_deferrals_rk', 0.0):,.2f}",
        )

    with col3:
        st.metric("Deferral mismatches", summary.get("deferral_mismatch_count", 0))
        st.metric("Loan mismatches", summary.get("loan_mismatch_count", 0))

    st.divider()

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Late deferrals (rows)", summary.get("late_deferral_count", 0))

    with col5:
        st.write("Evidence Pack")
        evidence_pack_path = summary.get("evidence_pack_path")
        if evidence_pack_path:
            evidence_path = Path(evidence_pack_path)
            if evidence_path.exists():
                st.download_button(
                    label="Download Evidence Pack",
                    data=open(evidence_path, "rb").read(),
                    file_name=evidence_path.name,
                    mime="application/zip",
                )
            else:
                st.info("Evidence pack not available locally")
        else:
            st.info("Evidence pack path not available")


def build_anomaly_narrative(summary: Dict[str, Any], results_dict: Dict[str, Any]) -> str:
    """
    Turn the run metrics into a one-line narrative an auditor / committee can consume.
    
    Args:
        summary: Summary dict with high-level KPIs from the API
        results_dict: Dictionary with paths and metrics from reconciliation
    """
    def safe_len(path_key: str) -> int:
        p = Path(results_dict.get(path_key, ""))
        if p.exists() and p.is_file():
            try:
                df = pd.read_csv(p)
                return len(df)
            except Exception:
                return 0
        return 0

    def sample_ids(path_key: str, label: str) -> str:
        """
        Pull up to 3 distinct employee IDs from a CSV (if column present).
        Returns formatted string or '' if none.
        """
        p = Path(results_dict.get(path_key, ""))
        if not (p.exists() and p.is_file()):
            return ""

        try:
            df = pd.read_csv(p)
        except Exception:
            return ""

        # Try common column names
        for col in ["employee_id", "EmpNumber", "Part_ID"]:
            if col in df.columns:
                ids = (
                    df[col]
                    .astype(str)
                    .dropna()
                    .unique()
                    .tolist()
                )
                if not ids:
                    return ""
                ids_sample = ids[:3]
                return f"{label} (e.g., {', '.join(ids_sample)})"
        return ""

    late_deferrals = safe_len("late_deferrals")
    def_mismatch = safe_len("deferral_mismatches")
    loan_mismatch = safe_len("loan_mismatches")

    parts = []

    # Core counts
    parts.append(
        f"Flagged {def_mismatch} deferral mismatches and {loan_mismatch} loan mismatches"
    )

    if late_deferrals > 0:
        parts.append(f"{late_deferrals} payroll rows with late or missing deferrals")
    else:
        parts.append("no late or missing deferral rows based on the current threshold")

    # Sample IDs
    id_snippets = []
    def_snip = sample_ids("deferral_mismatches", "Deferral variances tied to IDs")
    if def_snip:
        id_snippets.append(def_snip)

    loan_snip = sample_ids("loan_mismatches", "Loan variances tied to IDs")
    if loan_snip:
        id_snippets.append(loan_snip)

    if id_snippets:
        parts.append("; ".join(id_snippets))

    # Risk label
    risk = classify_run_risk(summary)
    if risk == "High":
        parts.append("overall run classified as HIGH risk")
    elif risk == "Medium":
        parts.append("overall run classified as MEDIUM risk")
    else:
        parts.append("overall run classified as LOW risk")

    return " – ".join(parts) + "."
def render_batch_reconciliation_tab():
    st.title("📦 Batch Reconciliation (Beta)")
    st.write(
        "Upload multiple payroll and recordkeeper files and run reconciliation in batch."
    )

    st.header("1. Upload Batch Files")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Upload Payroll CSVs (batch)")
        batch_payroll_files = st.file_uploader(
            " ", type=["csv"], key="batch_payroll_csv", accept_multiple_files=True
        )

    with col2:
        st.caption("Upload Recordkeeper CSVs (batch)")
        batch_rk_files = st.file_uploader(
            "  ", type=["csv"], key="batch_rk_csv", accept_multiple_files=True
        )

    plan_prefix = st.text_input("Plan / batch label prefix", value="Batch Plan")

    st.markdown("---")

    run_batch_button = st.button(
        "▶️ Run Batch Reconciliation", type="primary", use_container_width=True
    )

    if not run_batch_button:
        return

    # Basic validation
    if not batch_payroll_files or not batch_rk_files:
        st.error("Upload at least one payroll CSV and one recordkeeper CSV.")
        return

    if len(batch_payroll_files) != len(batch_rk_files):
        st.error(
            f"Batch mismatch: {len(batch_payroll_files)} payroll file(s) vs "
            f"{len(batch_rk_files)} recordkeeper file(s). Counts must match."
        )
        return

    batch_rows = []

    with st.spinner("Running batch reconciliation via API..."):
        for idx, (p_file, rk_file) in enumerate(
            zip(batch_payroll_files, batch_rk_files), start=1
        ):
            # Plan name for this run
            plan_name = f"{plan_prefix} #{idx}"

            try:
                # Call create_run (works in both API and direct modes)
                api_result = create_run(
                    payroll_bytes=p_file.getvalue(),
                    payroll_filename=p_file.name,
                    rk_bytes=rk_file.getvalue(),
                    rk_filename=rk_file.name,
                    plan_name=plan_name,
                    plan_rules=None,  # Batch runs don't have eligibility rules UI yet
                )
                
                summary_dict = api_result.get("summary", {})
                if not summary_dict:
                    # Skip this run if it failed
                    continue
                
                # Simple risk calculation from summary
                deferral_mismatches = summary_dict.get("deferral_mismatch_count", 0)
                loan_mismatches = summary_dict.get("loan_mismatch_count", 0)
                late_deferrals = summary_dict.get("late_deferral_count", 0)
                
                if deferral_mismatches > 100 or loan_mismatches > 50 or late_deferrals > 50:
                    risk = "High"
                elif deferral_mismatches > 10 or loan_mismatches > 5 or late_deferrals > 10:
                    risk = "Medium"
                else:
                    risk = "Low"
                
                # Create a simple summary object for compatibility
                batch_rows.append(
                    {
                        "Run #": idx,
                        "Plan": summary_dict.get("plan_name", plan_name),
                        "Payroll file": p_file.name,
                        "Recordkeeper file": rk_file.name,
                        "Payroll vendor": summary_dict.get("payroll_vendor", "Unknown"),
                        "Recordkeeper vendor": summary_dict.get("rk_vendor", "Unknown"),
                        "Deferrals (Payroll)": summary_dict.get("total_deferrals_payroll", 0),
                        "Deferrals (RK)": summary_dict.get("total_deferrals_rk", 0),
                        "Deferral mismatches": summary_dict.get("deferral_mismatch_count", 0),
                        "Loan mismatches": summary_dict.get("loan_mismatch_count", 0),
                        "Late deferrals (rows)": summary_dict.get("late_deferral_count", 0),
                        "Risk": risk,
                        "Evidence pack": summary_dict.get("evidence_pack_path", "N/A"),
                    }
                )
            except requests.RequestException as e:
                st.warning(f"Failed to process batch item {idx}: {e}")
                continue

    if not batch_rows:
        st.warning("No runs completed. Check batch inputs.")
        return

    batch_df = pd.DataFrame(batch_rows)

    st.success("Batch reconciliation completed.")
    st.subheader("Batch Run Dashboard")
    st.dataframe(batch_df, use_container_width=True)

    # Download CSV summary
    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download batch_summary.csv",
        data=csv_bytes,
        file_name="batch_summary.csv",
        mime="text/csv",
    )

def render_reconciliation_tab():
    # Initialize variables to avoid UnboundLocalError
    results = None
    stdout = ""
    
    st.title("🔗 ProofLink")
    st.write(
        "Upload payroll and recordkeeper CSVs to analyze plan compliance and generate reports."
    )

    # ---------- Upload Section ----------
    st.header("1. Upload Files")
    
    # Load Demo checkbox
    use_demo = st.checkbox("📁 Load Demo Files", value=False, help="Load demo files from demo/ folder (no upload required)")
    
    # Demo files preset button (legacy - runs immediately)
    col_demo, col_upload = st.columns([1, 3])
    with col_demo:
        demo_button = st.button("▶️ Run Demo Files Now", type="secondary", use_container_width=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Upload Payroll CSV")
        payroll_file = st.file_uploader(" ", type=["csv"], key="recon_payroll", disabled=use_demo)

    with col2:
        st.caption("Upload Recordkeeper CSV")
        rk_file = st.file_uploader("  ", type=["csv"], key="recon_rk", disabled=use_demo)
    
    # Load demo files if checkbox is checked
    if use_demo:
        repo_root = Path(__file__).resolve().parent
        demo_payroll_path = repo_root / "demo" / "demo_payroll.csv"
        demo_rk_path = repo_root / "demo" / "demo_recordkeeper.csv"
        
        if demo_payroll_path.exists() and demo_rk_path.exists():
            # Read demo files and create file-like objects for compatibility
            with open(demo_payroll_path, "rb") as f:
                payroll_bytes = f.read()
            with open(demo_rk_path, "rb") as f:
                rk_bytes = f.read()
            
            # Create BytesIO objects to simulate uploaded files
            payroll_file = io.BytesIO(payroll_bytes)
            payroll_file.name = "demo_payroll.csv"
            rk_file = io.BytesIO(rk_bytes)
            rk_file.name = "demo_recordkeeper.csv"
            
            st.info("✅ Demo files loaded from demo/ folder")
        else:
            st.error(f"Demo files not found. Expected:\n- {demo_payroll_path}\n- {demo_rk_path}")
            use_demo = False

    # Plan name input
    plan_name = st.text_input("Plan name", value="Demo Plan")
    
    # Handle demo button click - run directly
    if demo_button:
        # Set demo file paths in session state
        repo_root = Path(__file__).resolve().parent
        demo_payroll_path = repo_root / "data" / "raw" / "demo_broken_payroll.csv"
        demo_rk_path = repo_root / "data" / "raw" / "demo_clean_rk.csv"
        demo_mapping_path = repo_root / "mapping_example.yaml"
        
        if not demo_payroll_path.exists() or not demo_rk_path.exists():
            st.error("Demo files not found. Please check that demo files exist in data/raw/")
            st.stop()
        
        # Read demo files into bytes
        with open(demo_payroll_path, "rb") as f:
            payroll_bytes = f.read()
        payroll_filename = demo_payroll_path.name
        
        with open(demo_rk_path, "rb") as f:
            rk_bytes = f.read()
        rk_filename = demo_rk_path.name
        
        # Validate recordkeeper headers
        with open(demo_rk_path, "rb") as f:
            is_valid, error_msg = validate_recordkeeper_headers(f)
            if not is_valid:
                st.error(error_msg)
                st.stop()
        
        # Run directly with demo files
        try:
            spinner_text = "Running ProofLink analysis with demo files..." if USE_API_BACKEND else "Running ProofLink analysis with demo files..."
            
            # Use demo mapping if available, otherwise use default
            demo_mapping_path = repo_root / "demo" / "demo_mapping.yaml"
            original_mapping_path = os.environ.get("MAPPING_YAML_PATH")
            if demo_mapping_path.exists():
                os.environ["MAPPING_YAML_PATH"] = str(demo_mapping_path)
            
            try:
                with st.spinner(spinner_text):
                    api_result = create_run(
                        payroll_bytes=payroll_bytes,
                        payroll_filename=payroll_filename,
                        rk_bytes=rk_bytes,
                        rk_filename=rk_filename,
                        plan_name=plan_name,
                        plan_rules=plan_rules,
                        payroll_vendor_hint=payroll_vendor_hint,
                        rk_vendor_hint=rk_vendor_hint,
                    )
            finally:
                # Restore original mapping path
                if demo_mapping_path.exists():
                    if original_mapping_path:
                        os.environ["MAPPING_YAML_PATH"] = original_mapping_path
                    else:
                        os.environ.pop("MAPPING_YAML_PATH", None)
            
            run_id = api_result.get("run_id")
            summary_dict = api_result.get("summary", {})
            status = api_result.get("status", "unknown")
            
            if not run_id:
                st.error("Run did not return a run_id.")
            else:
                # Store run_id and summary in session state for later use
                st.session_state["current_run_id"] = run_id
                st.session_state["current_summary"] = summary_dict
                st.session_state["current_status"] = status
                st.success("Analysis completed with demo files.")
        except requests.RequestException as e:
            if USE_API_BACKEND:
                st.error(f"Failed to contact ProofLink API: {e}")
            else:
                st.error(f"Failed to run ProofLink engine: {e}")
        except Exception as e:
            st.error(f"Error running ProofLink analysis: {e}")
            st.exception(e)

    st.markdown("---")

    # Initialize plan_rules with defaults
    plan_rules = {
        "eligibility_rule": "immediate",
        "service_days_required": None,
        "age_required": None,
        "align_first_month": False,
    }

    # ---------- Vendor Hints ----------
    with st.expander("Advanced Options: Vendor Hints"):
        payroll_vendor_hint = st.selectbox(
            "Payroll Vendor",
            ["Auto-detect", "ADP", "Paychex", "Paylocity", "VENDOR_RK_1", "Generic"],
            index=0,
        )
        rk_vendor_hint = st.selectbox(
            "Recordkeeper Vendor",
            ["Auto-detect", "VENDOR_RK_1", "Fidelity", "Vanguard", "Generic"],
            index=0,
        )

        payroll_vendor_hint = (
            None if payroll_vendor_hint == "Auto-detect" else payroll_vendor_hint
        )
        rk_vendor_hint = (
            None if rk_vendor_hint == "Auto-detect" else rk_vendor_hint
        )
        
        # Eligibility Rules
        st.markdown("### Eligibility Rules")
        rule_type = st.selectbox(
            "Eligibility rule",
            ["immediate", "age21_and_1year", "service_only", "age_only"],
            index=0,
        )
        
        service_days = None
        age_required = None
        
        if rule_type == "service_only":
            service_days = st.number_input(
                "Service requirement (days)",
                min_value=0,
                max_value=1095,
                value=0,
            )
        
        if rule_type == "age_only":
            age_required = st.number_input(
                "Age requirement",
                min_value=18,
                max_value=75,
                value=21,
            )
        
        align_first_month = st.checkbox(
            "Align eligibility to first of month",
            value=False,
        )
        
        plan_rules = {
            "eligibility_rule": rule_type,
            "service_days_required": service_days,
            "age_required": age_required,
            "align_first_month": align_first_month,
        }

    # ---------- Mapping Readiness Check ----------
    st.header("2. Mapping Readiness")
    
    preflight_safe = False
    preflight_report = None
    preflight_blocked_reasons = []
    
    # Check preflight if files are uploaded
    if payroll_file and rk_file:
        # Save files temporarily for preflight check
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            payroll_temp_path = temp_path / (payroll_file.name or "payroll.csv")
            rk_temp_path = temp_path / (rk_file.name or "rk.csv")
            
            # Use demo mapping if demo files are loaded, otherwise use default
            if use_demo:
                mapping_path = BASE_DIR / "demo" / "demo_mapping.yaml"
            else:
                mapping_path = BASE_DIR / "mapping_example.yaml"
            
            # Write uploaded files to temp location
            # Handle both BytesIO objects and uploaded file objects
            if hasattr(payroll_file, 'getvalue'):
                payroll_data = payroll_file.getvalue()
            else:
                payroll_file.seek(0)
                payroll_data = payroll_file.read()
                payroll_file.seek(0)
            
            if hasattr(rk_file, 'getvalue'):
                rk_data = rk_file.getvalue()
            else:
                rk_file.seek(0)
                rk_data = rk_file.read()
                rk_file.seek(0)
            
            with open(payroll_temp_path, "wb") as f:
                f.write(payroll_data)
            with open(rk_temp_path, "wb") as f:
                f.write(rk_data)
            
            # Run preflight check
            try:
                preflight_safe, preflight_report = run_preflight(
                    payroll_csv_path=str(payroll_temp_path),
                    recordkeeper_csv_path=str(rk_temp_path),
                    mapping_yaml_path=str(mapping_path)
                )
            except Exception as e:
                st.error(f"Preflight check failed: {e}")
                preflight_safe = False
                preflight_report = {"warnings": [f"Preflight error: {e}"]}
    
    # Display Mapping Readiness checklist
    checklist_col1, checklist_col2 = st.columns([3, 1])
    
    with checklist_col1:
        st.markdown("**Mapping Readiness Checklist:**")
        
        # Files uploaded
        files_ok = payroll_file is not None and rk_file is not None
        files_status = "✅" if files_ok else "❌"
        st.markdown(f"{files_status} Files uploaded")
        
        # Mapping loaded
        if not payroll_file or not rk_file:
            mapping_status = "⏳"
        elif preflight_report is None:
            mapping_status = "⏳"
        elif preflight_report and "warnings" in preflight_report:
            # Check if mapping failed to load
            mapping_failed = any("Failed to load mapping file" in str(w) for w in preflight_report.get("warnings", []))
            mapping_status = "❌" if mapping_failed else "✅"
        else:
            mapping_status = "✅"
        st.markdown(f"{mapping_status} Mapping loaded")
        
        # Headers match mapping
        if not payroll_file or not rk_file:
            headers_status = "⏳"
        elif preflight_report is None:
            headers_status = "⏳"
        elif preflight_report and "missing_mapped_headers" in preflight_report:
            missing_payroll = preflight_report["missing_mapped_headers"].get("payroll", [])
            missing_rk = preflight_report["missing_mapped_headers"].get("recordkeeper", [])
            headers_status = "❌" if (missing_payroll or missing_rk) else "✅"
        else:
            headers_status = "✅"
        st.markdown(f"{headers_status} Headers match mapping")
        
        # Join key coverage OK
        if not payroll_file or not rk_file:
            join_key_status = "⏳"
        elif preflight_report is None:
            join_key_status = "⏳"
        elif preflight_report:
            if (preflight_report.get("join_key_not_mapped", {}).get("payroll", False) or
                preflight_report.get("join_key_not_mapped", {}).get("recordkeeper", False) or
                preflight_report.get("join_key_empty_file", {}).get("payroll", False) or
                preflight_report.get("join_key_empty_file", {}).get("recordkeeper", False)):
                join_key_status = "❌"
            else:
                join_key_status = "✅"
        else:
            join_key_status = "✅"
        st.markdown(f"{join_key_status} Join key coverage OK")
    
    with checklist_col2:
        # Overall status
        if not files_ok:
            st.markdown("**Status:** ⏳ Waiting for files")
        elif preflight_safe:
            st.markdown("**Status:** ✅ Ready")
        else:
            st.markdown("**Status:** ❌ Blocked")
    
    # Show BLOCKED reasons if preflight failed
    if preflight_report and not preflight_safe:
        st.error("**BLOCKED:**")
        
        # Check for blocked files
        if "blocked_files" in preflight_report and preflight_report["blocked_files"]:
            for file_type, file_path in preflight_report["blocked_files"]:
                if file_type == "payroll":
                    st.error(f"- Payroll file not found: {file_path}")
                elif file_type == "recordkeeper":
                    st.error(f"- Recordkeeper file not found: {file_path}")
        
        # Check for missing mapped headers
        if "missing_mapped_headers" in preflight_report:
            missing_payroll = preflight_report["missing_mapped_headers"].get("payroll", [])
            missing_rk = preflight_report["missing_mapped_headers"].get("recordkeeper", [])
            if missing_payroll:
                st.error(f"- Missing payroll headers referenced by mapping: {', '.join(missing_payroll)}")
            if missing_rk:
                st.error(f"- Missing recordkeeper headers referenced by mapping: {', '.join(missing_rk)}")
        
        # Check for join key issues
        if preflight_report.get("join_key_not_mapped", {}).get("payroll", False):
            st.error("- Payroll join key (employee_id) is not mapped or not present in headers")
        if preflight_report.get("join_key_not_mapped", {}).get("recordkeeper", False):
            st.error("- Recordkeeper join key (employee_id) is not mapped or not present in headers")
        if preflight_report.get("join_key_empty_file", {}).get("payroll", False):
            st.error("- Payroll file has 0 rows (cannot compute join-key coverage)")
        if preflight_report.get("join_key_empty_file", {}).get("recordkeeper", False):
            st.error("- Recordkeeper file has 0 rows (cannot compute join-key coverage)")
    
    st.markdown("---")
    
    # Run button - disabled if preflight fails
    run_button = st.button(
        "▶️ Run Analysis", 
        type="primary", 
        use_container_width=True,
        disabled=not preflight_safe or not files_ok
    )

    # ---------- Run Reconciliation ----------
    if run_button:
        # Use uploaded files
        if not payroll_file or not rk_file:
            st.error("Please upload BOTH a payroll CSV and a recordkeeper CSV.")
            return
        
        # Block if preflight failed
        if not preflight_safe:
            st.error("Cannot run analysis: Preflight checks failed. Please fix the issues shown above.")
            return
        
        # Validate recordkeeper headers before running
        is_valid, error_msg = validate_recordkeeper_headers(rk_file)
        if not is_valid:
            st.error(error_msg)
            st.stop()
        
        # Handle both BytesIO objects and uploaded file objects
        if hasattr(payroll_file, 'getvalue'):
            payroll_bytes = payroll_file.getvalue()
        else:
            payroll_file.seek(0)
            payroll_bytes = payroll_file.read()
            payroll_file.seek(0)
        
        if hasattr(rk_file, 'getvalue'):
            rk_bytes = rk_file.getvalue()
        else:
            rk_file.seek(0)
            rk_bytes = rk_file.read()
            rk_file.seek(0)
        
        payroll_filename = getattr(payroll_file, 'name', 'payroll.csv')
        rk_filename = getattr(rk_file, 'name', 'rk.csv')

        try:
            spinner_text = "Running ProofLink analysis via API..." if USE_API_BACKEND else "Running ProofLink analysis..."
            
            # Set mapping path for demo files
            original_mapping_path = os.environ.get("MAPPING_YAML_PATH")
            if use_demo:
                demo_mapping_path = BASE_DIR / "demo" / "demo_mapping.yaml"
                os.environ["MAPPING_YAML_PATH"] = str(demo_mapping_path)
            
            try:
                with st.spinner(spinner_text):
                    api_result = create_run(
                        payroll_bytes=payroll_bytes,
                        payroll_filename=payroll_filename,
                        rk_bytes=rk_bytes,
                        rk_filename=rk_filename,
                        plan_name=plan_name,
                        plan_rules=plan_rules,
                        payroll_vendor_hint=payroll_vendor_hint,
                        rk_vendor_hint=rk_vendor_hint,
                    )
            finally:
                # Restore original mapping path
                if use_demo:
                    if original_mapping_path:
                        os.environ["MAPPING_YAML_PATH"] = original_mapping_path
                    else:
                        os.environ.pop("MAPPING_YAML_PATH", None)

            run_id = api_result.get("run_id")
            summary_dict = api_result.get("summary", {})
            status = api_result.get("status", "unknown")

            if not run_id:
                st.error("Run did not return a run_id.")
            else:
                # Store run_id and summary in session state for later use
                st.session_state["current_run_id"] = run_id
                st.session_state["current_summary"] = summary_dict
                st.session_state["current_status"] = status
                st.success("Analysis completed.")
        except requests.RequestException as e:
            if USE_API_BACKEND:
                st.error(f"Failed to contact ProofLink API: {e}")
            else:
                st.error(f"Failed to run ProofLink engine: {e}")
            return
        except Exception as e:
            st.error(f"Error running ProofLink analysis: {e}")
            st.exception(e)
            return

    # Display results if available
    summary_dict = st.session_state.get("current_summary")
    run_id = st.session_state.get("current_run_id")
    
    if summary_dict and run_id:
        # Get full run details from API to access manifest and other data
        try:
            run_details = api_get_run(run_id)
            summary_dict = run_details.get("summary", summary_dict)
            manifest = run_details.get("manifest", {})
        except requests.RequestException:
            # Fall back to cached summary if API call fails
            manifest = {}
        
        # Use summary_dict directly (it's already a dict from the API)
        results_dict = {"vendor_detection": summary_dict.get("vendor_detection", {})}
            
        # Extract coverage gap counts from summary (if available in mismatches)
        # Note: API summary may not have mismatches dict, so we'll use summary fields directly
        only_in_payroll = 0  # Will be populated from summary if available
        only_in_rk = 0
        coverage_gaps = only_in_payroll + only_in_rk
        
        # Check for missing columns warning
        all_totals_zero = (
            summary_dict.get("total_deferrals_payroll", 0) == 0
            and summary_dict.get("total_deferrals_rk", 0) == 0
            and summary_dict.get("total_loans_payroll", 0) == 0
        )
        
        if all_totals_zero:
            st.warning(
                "The uploaded files may be missing required deferral/loan columns "
                "(e.g., def_amount, loan_amount), so metrics are 0. "
                "This is a file-format issue, not a ProofLink engine error."
            )
        
        # Plan Health / Plan Summary (shown first as hero section)
        plan_health = summary_dict.get("plan_health") if isinstance(summary_dict, dict) else None
        if plan_health:
            st.subheader("Plan Health")
            
            cols = st.columns(3)
            cols[0].metric("Health Score", f"{plan_health.get('score', 0)}")
            cols[1].metric("Grade", plan_health.get("grade", "N/A"))
            cols[2].metric("Risk Level", plan_health.get("risk_level", "N/A"))
            
            with st.expander("Plan health details", expanded=False):
                st.write("Weighted violations:", plan_health.get("weighted_violations"))
                st.write("Total participants:", plan_health.get("total_participants"))
                by_cat = plan_health.get("by_category") or {}
                if by_cat:
                    st.write("Issues by category:")
                    st.json(by_cat)
        
        st.divider()
        
        # Contribution Timing Analysis
        st.markdown("### Contribution Timing Analysis")
        timing_metrics = summary_dict.get("timing_metrics", {})
        
        if not timing_metrics:
            st.write("No timing metrics available for this run.")
        else:
            timing_risk = timing_metrics.get("timing_risk", "N/A")
            total_rows = timing_metrics.get("total_rows", 0)
            late_rows = timing_metrics.get("late_rows", 0)
            missing_deposits = timing_metrics.get("missing_deposits", 0)
            
            cols = st.columns(4)
            cols[0].metric("Timing Risk", timing_risk)
            cols[1].metric("Total Rows Analyzed", total_rows)
            cols[2].metric("Late Rows", late_rows)
            cols[3].metric("Missing Deposits", missing_deposits)
        
        # Secure 2.0 Exceptions
        st.divider()
        st.markdown("### Secure 2.0 Exceptions")
        
        # Read unified Secure 2.0 summary
        secure20 = summary_dict.get("secure20") or {}
        if not isinstance(secure20, dict):
            secure20 = {}
        
        total_secure20 = secure20.get("total_violations", 0)
        
        if total_secure20 == 0:
            st.success("No Secure 2.0 exceptions detected for this plan.")
        else:
            st.metric("Total Secure 2.0 exceptions", total_secure20)
            
            # Show breakdown by violation type
            by_type = secure20.get("by_type", {})
            if by_type:
                st.caption("Breakdown by exception type:")
                for vtype, count in sorted(by_type.items()):
                    # Map violation types to human-readable labels
                    label = vtype
                    if vtype == "HCE catch-up not coded as Roth":
                        label = "HCE catch-up not coded as Roth"
                    elif vtype == "Potential catch-up coded in base deferral source":
                        label = "Potential catch-up coded in base deferral source"
                    st.write(f"- **{label}**: {count}")
            
            # Show detailed violations table
            violations = secure20.get("violations", [])
            if violations:
                with st.expander("View Secure 2.0 exception detail", expanded=False):
                    df_secure20 = pd.DataFrame(violations)
                    if not df_secure20.empty:
                        # Select key columns for display
                        display_cols = []
                        for col in ["employee_id", "violation_type", "age", "is_hce", 
                                   "deferral_pretax", "deferral_roth", "catchup_amount",
                                   "issue_category", "severity", "correction_hint"]:
                            if col in df_secure20.columns:
                                display_cols.append(col)
                        
                        if display_cols:
                            st.dataframe(df_secure20[display_cols], use_container_width=True)
                        else:
                            st.dataframe(df_secure20, use_container_width=True)
                    else:
                        st.write("No detailed rows available.")
            
            # Show CSV path if available
            secure20_csv_path = secure20.get("csv_path")
            if secure20_csv_path:
                st.caption(f"Secure 2.0 violations CSV: {secure20_csv_path}")
        
        # Eligibility Drift Detection
        st.divider()
        st.markdown("### Eligibility Drift Detection")
        
        drift_summary = summary_dict.get("eligibility_drift") or {}
        if not isinstance(drift_summary, dict):
            drift_summary = {}
        
        drift_count = drift_summary.get("eligibility_drift_count", 0)
        drift_csv_path = drift_summary.get("csv_path")
        
        st.markdown(f"**Eligibility drift rows:** {drift_count}")
        
        if drift_count > 0 and drift_csv_path and os.path.exists(drift_csv_path):
            run_id = summary_dict.get("run_id", "unknown")
            with open(drift_csv_path, "rb") as f:
                drift_data = f.read()
            st.download_button(
                label="Download eligibility drift CSV",
                data=drift_data,
                file_name="eligibility_drift.csv",
                mime="text/csv",
                key=f"eligibility_drift_{run_id}",
            )
        
        # Reconciliation Summary (compact)
        st.divider()
        st.markdown("### Reconciliation Summary")
        
        col3, col4, col5 = st.columns(3)
        col3.metric("Deferral Mismatches", f"{summary_dict.get('deferral_mismatch_count', 0):,}")
        col4.metric("Loan Mismatches", f"{summary_dict.get('loan_mismatch_count', 0):,}")
        col5.metric("Late Deferral Rows", f"{summary_dict.get('late_deferral_count', 0):,}")
        
        # Output Files
        st.divider()
        st.subheader("Output Files")
        
        if run_id:
            if st.button("Download Evidence Pack"):
                with st.spinner("Fetching evidence pack from API..."):
                    data = api_download_evidence_pack(run_id)
                if data is None:
                    st.error("Evidence pack not available for this run.")
                else:
                    st.download_button(
                        label="📥 Download Evidence Pack (ZIP)",
                        data=data,
                        file_name=f"evidence_{run_id}.zip",
                        mime="application/zip",
                    )
        
        # Plan Exception Summary download
        plan_ex_summary = summary_dict.get("plan_exceptions") if isinstance(summary_dict, dict) else None
        plan_ex_csv_path = None
        if isinstance(plan_ex_summary, dict):
            plan_ex_csv_path = plan_ex_summary.get("csv_path")
        
        if plan_ex_csv_path and os.path.exists(plan_ex_csv_path):
            st.markdown("### Plan Exception Summary")
            with open(plan_ex_csv_path, "rb") as f:
                plan_ex_data = f.read()
            st.download_button(
                label="Download Plan Exception Summary (CSV)",
                data=plan_ex_data,
                file_name="plan_exception_summary.csv",
                mime="text/csv",
                key="download_plan_exception_summary",
            )
        
        if not run_id:
            st.info("Run an analysis first to generate an evidence pack.")
    else:
        st.info("Run an analysis to see results.")

        # mismatches_detail = results.get("mismatches_detail")
        # if mismatches_detail:
        #     with st.expander("Deferral Mismatches"):
        #         def_df = mismatches_detail.get("deferrals_df")
        #         if def_df is not None:
        #             st.dataframe(def_df)
        #     with st.expander("Loan Mismatches"):
        #         loan_df = mismatches_detail.get("loans_df")
        #         if loan_df is not None:
        #             st.dataframe(loan_df)


    # Note: Run History is now in a separate tab

    try:
        with st.spinner("Loading recent runs from ProofLink API..."):
            runs_data = api_list_runs(limit=10)
        items = runs_data.get("items", [])
    except requests.RequestException as e:
        st.error(f"Failed to load run history from API: {e}")
        items = []

    if not items:
        st.info("No runs found yet. Run a reconciliation to see history here.")
    else:
        st.write(f"Found {len(items)} recent run(s) from ProofLink API")

        # Show the most recent runs
        for r in items:
            run_id = r.get("run_id")
            plan_name = r.get("plan_name") or "Unnamed"
            created_at = r.get("created_at")
            status = r.get("status")
            has_evidence = r.get("has_evidence_pack", False)
            summary = r.get("summary", {}) or {}

            with st.expander(f"Run: {plan_name} — {run_id} | Created: {created_at}"):
                st.write(f"**Run ID:** `{run_id}`")
                st.write(f"**Status:** {status}")
                st.write(f"**Created:** {created_at}")
                st.write(f"**Payroll File:** {r.get('payroll_filename')}")
                st.write(f"**Recordkeeper File:** {r.get('rk_filename')}")

                # Show key metrics
                if summary:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Deferral Mismatches", summary.get("deferral_mismatch_count", 0))
                    col2.metric("Loan Mismatches", summary.get("loan_mismatch_count", 0))
                    col3.metric("Late Deferrals", summary.get("late_deferral_count", 0))

                if has_evidence:
                    if st.button(f"Download Evidence Pack for {run_id}", key=f"dl_recon_{run_id}"):
                        with st.spinner("Fetching evidence pack from API..."):
                            data = api_download_evidence_pack(run_id)
                        if data is None:
                            st.error("Evidence pack not available for this run.")
                        else:
                            st.download_button(
                                label="Download Evidence Pack ZIP",
                                data=data,
                                file_name=f"evidence_{run_id}.zip",
                                mime="application/zip",
                                key=f"dl_btn_recon_{run_id}",
                            )
                else:
                    st.info("Evidence pack not available for this run.")


def render_run_history_tab():
    st.header("Run History")
    st.markdown("View summary of all previous reconciliation runs.")
    
    try:
        with st.spinner("Loading recent runs from ProofLink API..."):
            runs_data = api_list_runs(limit=50)
        items = runs_data.get("items", [])
    except requests.RequestException as e:
        st.error(f"Failed to load run history from API: {e}")
        items = []
    
    if not items:
        st.info("No runs found yet. Run an analysis to see history here.")
    else:
        # Summary table
        table_rows = []
        for r in items:
            summary = r.get("summary", {}) or {}
            # Try to pick out a risk indicator if present
            risk = summary.get("timing_risk") or summary.get("run_risk_level") or "N/A"
            
            # Calculate total issues from summary
            deferral_mismatches = summary.get("deferral_mismatch_count", 0)
            loan_mismatches = summary.get("loan_mismatch_count", 0)
            late_deferrals = summary.get("late_deferral_count", 0)
            total_issues = deferral_mismatches + loan_mismatches + late_deferrals
            
            # Compute risk level
            if total_issues == 0:
                risk_label = "Low"
            elif 1 <= total_issues <= 20:
                risk_label = "Medium"
            else:
                risk_label = "High"
            
            table_rows.append({
                "Run ID": r.get("run_id", "N/A"),
                "Plan": r.get("plan_name", "Unknown"),
                "Status": r.get("status", "unknown"),
                "Created": r.get("created_at", "N/A"),
                "Payroll File": r.get("payroll_filename", "N/A"),
                "RK File": r.get("rk_filename", "N/A"),
                "Risk": risk_label,
                "Deferral Mismatches": deferral_mismatches,
                "Loan Mismatches": loan_mismatches,
                "Late Deferrals": late_deferrals,
                "Total Issues": total_issues,
            })
        
        if table_rows:
            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Detailed per-run view with download buttons
        st.markdown("### Detailed Run Information")
        
        for r in items:
            run_id = r.get("run_id")
            plan_name = r.get("plan_name") or "Unnamed"
            created_at = r.get("created_at")
            status = r.get("status")
            has_evidence = r.get("has_evidence_pack", False)
            summary = r.get("summary", {}) or {}
            
            with st.expander(f"{plan_name} — {run_id}"):
                st.write(f"**Run ID:** `{run_id}`")
                st.write(f"**Status:** {status}")
                st.write(f"**Created:** {created_at}")
                st.write(f"**Payroll File:** {r.get('payroll_filename')}")
                st.write(f"**Recordkeeper File:** {r.get('rk_filename')}")
                
                # Show summary metrics
                if summary:
                    st.markdown("#### Summary Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Deferral Mismatches", summary.get("deferral_mismatch_count", 0))
                    col2.metric("Loan Mismatches", summary.get("loan_mismatch_count", 0))
                    col3.metric("Late Deferrals", summary.get("late_deferral_count", 0))
                    
                    col4, col5 = st.columns(2)
                    col4.metric("Total Payroll Deferrals", f"${summary.get('total_deferrals_payroll', 0):,.2f}")
                    col5.metric("Total RK Deferrals", f"${summary.get('total_deferrals_rk', 0):,.2f}")
                
                # Show full summary JSON in expander
                with st.expander("View Full Summary JSON"):
                    st.json(summary)
                
                if has_evidence:
                    if st.button(f"Download Evidence Pack for {run_id}", key=f"dl_{run_id}"):
                        with st.spinner("Fetching evidence pack from API..."):
                            data = api_download_evidence_pack(run_id)
                        if data is None:
                            st.error("Evidence pack not available for this run.")
                        else:
                            st.download_button(
                                label="Download Evidence Pack ZIP",
                                data=data,
                                file_name=f"evidence_{run_id}.zip",
                                mime="application/zip",
                                key=f"dl_btn_{run_id}",
                            )
                else:
                    st.info("Evidence pack not available for this run.")


# ==============================
#  Main entrypoint with tabs
# ==============================
def main():
    tab_analyze, tab_history = st.tabs(
        ["Analyze Plan", "Run History"]
    )

    with tab_analyze:
        render_reconciliation_tab()
    
    with tab_history:
        render_run_history_tab()



if __name__ == "__main__":
    main()
