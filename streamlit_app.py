import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import re
from typing import Optional
import sys
import io
import traceback
from datetime import datetime
import json

from main import RunSummary, run_reconciliation_with_summary


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


# ==============================
#  Contribution Timing TAB
# ==============================

def run_reconciliation_with_stdout_capture(
    payroll_file, rk_file, plan_name: str = "Demo Plan",
    payroll_vendor_hint: Optional[str] = None,
    rk_vendor_hint: Optional[str] = None,
    output_dir: Path = None
) -> tuple[bool, str, dict]:
    """
    Run reconciliation and capture stdout.
    
    Returns:
        tuple: (success: bool, stdout: str, results: dict with 'summary' RunSummary)
    """
    if output_dir is None:
        output_dir = BASE_DIR.parent / "data" / "processed"
    
    # Create run folder with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create inputs subfolder
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files to inputs subfolder
    payroll_path = inputs_dir / f"payroll_{payroll_file.name}"
    rk_path = inputs_dir / f"rk_{rk_file.name}"
    
    with open(payroll_path, "wb") as f:
        f.write(payroll_file.getbuffer())
    
    with open(rk_path, "wb") as f:
        f.write(rk_file.getbuffer())
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        # run_reconciliation_with_summary() now always returns a dict
        results = run_reconciliation_with_summary(
            payroll_csv=payroll_path,
            rk_csv=rk_path,
            output_dir=run_dir,
            plan_name=plan_name,
            payroll_vendor_hint=payroll_vendor_hint,
            rk_vendor_hint=rk_vendor_hint,
        )
        
        # Ensure results is a dict (defensive check)
        if not isinstance(results, dict):
            # This should never happen, but handle gracefully
            results = {
                "summary": None,
                "results_dict": {},
                "error": f"run_reconciliation_with_summary returned {type(results)}, expected dict"
            }
        
        # Check if there's an error in the results
        if results.get("error"):
            success = False
        else:
            success = True
    except Exception as e:
        success = False
        # Include full traceback in error for debugging
        error_text = traceback.format_exc()
        results = {
            "summary": None,
            "results_dict": {},
            "error": error_text
        }
        print(f"Error: {e}")
    finally:
        stdout_text = captured_output.getvalue()
        sys.stdout = old_stdout
    
    return success, stdout_text, results


def load_run_history() -> list[dict]:
    """
    Load run history from run_summary.json files in run folders.
    
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


def compute_run_risk_level(summary: RunSummary) -> tuple[str, str]:
    """
    Return (label, color) based on mismatch/late counts.
    
    Rules:
    - Low: all counts == 0
    - Medium: total issues between 1 and 20
    - High: total issues > 20
    """
    total_issues = (
        summary.deferral_mismatch_count 
        + summary.loan_mismatch_count 
        + summary.late_deferral_count
    )
    
    if total_issues == 0:
        return ("Low", "green")
    elif 1 <= total_issues <= 20:
        return ("Medium", "orange")
    else:
        return ("High", "red")


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
            "python", str(BASE_DIR / "contribution_timing_analyzer_v2.py"),
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
    st.header("ProofLink Analysis")

    st.markdown(
        "Upload exactly two CSV files to run the contribution timing analyzer end-to-end."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        payroll_file = st.file_uploader(
            "Payroll CSV",
            type=["csv"],
            key="timing_payroll",
        )

    with col_right:
        rk_file = st.file_uploader(
            "Recordkeeper CSV",
            type=["csv"],
            key="timing_rk",
        )

    late_threshold = st.number_input(
        "Late threshold (days)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        key="timing_late_threshold"
    )

    run_button = st.button("Run ProofLink Analysis", type="primary")

    if not run_button:
        return

    if payroll_file is None or rk_file is None:
        st.warning("Please upload both Payroll CSV and Recordkeeper CSV files.")
        return

    # Run the analysis
    with st.spinner("Running ProofLink Analysis..."):
        success, stdout, stderr = run_prooflink_analysis(payroll_file, rk_file, late_threshold=late_threshold)

    # Display run status
    if success:
        st.success("ProofLink analysis completed successfully.")
        
        # Parse the analyzer output
        metrics = parse_analyzer_output(stdout)
        
        # Display metrics if we successfully parsed them
        if any(v is not None for v in metrics.values()):
            st.subheader("Analysis Summary")
            
            # Vendor detection metrics
            if metrics["payroll_vendor"] or metrics["rk_vendor"]:
                col1, col2 = st.columns(2)
                if metrics["payroll_vendor"]:
                    col1.metric("Payroll Vendor", metrics["payroll_vendor"])
                if metrics["rk_vendor"]:
                    col2.metric("Recordkeeper", metrics["rk_vendor"])
            
            # Contribution timing metrics
            if metrics["total_rows"] is not None or metrics["late_contributions"] is not None or metrics["missing_deposits"] is not None:
                st.divider()
                
                # Display Timing Risk prominently at the top
                if metrics["timing_risk"]:
                    risk = metrics["timing_risk"]
                    if risk == "Low":
                        emoji = "🟢"
                    elif risk == "Medium":
                        emoji = "🟠"
                    elif risk == "High":
                        emoji = "🔴"
                    else:  # N/A
                        emoji = "⚪"
                    
                    st.markdown(f"**Timing Risk:** {emoji} {risk}")
                
                col3, col4, col5 = st.columns(3)
                
                if metrics["total_rows"] is not None:
                    col3.metric("Total Payroll Rows", f"{metrics['total_rows']:,}")
                
                if metrics["late_contributions"] is not None and metrics["late_threshold"] is not None:
                    col4.metric(
                        f"Late Contributions (> {metrics['late_threshold']} days)",
                        f"{metrics['late_contributions']:,}"
                    )
                elif metrics["late_contributions"] is not None:
                    col4.metric("Late Contributions", f"{metrics['late_contributions']:,}")
                
                if metrics["missing_deposits"] is not None:
                    col5.metric("Missing Deposits", f"{metrics['missing_deposits']:,}")
        
        # Display raw output
        st.subheader("Analyzer Output")
        st.code(stdout, language="text")
    else:
        st.error("ProofLink analysis failed")
        if stderr:
            st.code(stderr, language="text")
        else:
            st.error("No error details available.")

    # Check for output files
    output_dir = BASE_DIR.parent / "data" / "processed"
    late_contributions_path = output_dir / "late_contributions.csv"

    if late_contributions_path.exists():
        st.subheader("Output Files")
        st.info(f"Late contributions CSV available at: {late_contributions_path}")

        # Provide download link
        try:
            with open(late_contributions_path, "rb") as f:
                st.download_button(
                    label="📥 Download late_contributions.csv",
                    data=f.read(),
                    file_name="late_contributions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not read output file: {e}")
def classify_run_risk(summary: RunSummary) -> str:
    """
    Simple risk classifier based on mismatch + late counts.
    We’ll upgrade later with dollar thresholds / percentages.
    """
    if summary.late_deferral_count > 0:
        return "High"
    if summary.deferral_mismatch_count > 0 or summary.loan_mismatch_count > 0:
        return "Medium"
    return "Low"


def render_run_summary(summary: RunSummary) -> None:
    st.subheader("Run Summary")

    risk = classify_run_risk(summary)
    risk_icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(risk, "⚪")
    st.markdown(f"**Run Risk Level:** {risk_icon} {risk}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Plan", summary.plan_name)
        st.text(f"Payroll vendor: {summary.payroll_vendor}")
        st.text(f"Recordkeeper: {summary.rk_vendor}")

    with col2:
        st.metric(
            "Deferrals (Payroll)",
            f"{summary.total_deferrals_payroll:,.2f}",
        )
        st.metric(
            "Deferrals (RK)",
            f"{summary.total_deferrals_rk:,.2f}",
        )

    with col3:
        st.metric("Deferral mismatches", summary.deferral_mismatch_count)
        st.metric("Loan mismatches", summary.loan_mismatch_count)

    st.divider()

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Late deferrals (rows)", summary.late_deferral_count)

    with col5:
        st.write("Evidence Pack")
        st.download_button(
            label="Download Evidence Pack",
            data=open(summary.evidence_pack_path, "rb").read(),
            file_name=summary.evidence_pack_path.name,
            mime="application/zip",
        )


def build_anomaly_narrative(summary: RunSummary, results_dict: dict) -> str:
    """
    Turn the run metrics into a one-line narrative an auditor / committee can consume.
    
    Args:
        summary: RunSummary object with high-level KPIs
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

    with st.spinner("Running batch reconciliation..."):
        for idx, (p_file, rk_file) in enumerate(
            zip(batch_payroll_files, batch_rk_files), start=1
        ):
            # Persist each pair to RAW_DIR
            payroll_path = RAW_DIR / f"batch_{idx}_{p_file.name}"
            rk_path = RAW_DIR / f"batch_{idx}_{rk_file.name}"

            payroll_path.write_bytes(p_file.getbuffer())
            rk_path.write_bytes(rk_file.getbuffer())

            # Plan name for this run
            plan_name = f"{plan_prefix} #{idx}"

            # For now, no batch-level vendor hints; rely on detection/Unknown
            results = run_reconciliation_with_summary(
                payroll_csv=payroll_path,
                rk_csv=rk_path,
                output_dir=PROCESSED_DIR,
                plan_name=plan_name,
                payroll_vendor_hint=None,
                rk_vendor_hint=None,
            )
            
            # Extract summary from results dict
            if not isinstance(results, dict) or not results.get("summary"):
                # Skip this run if it failed
                continue
            
            summary: RunSummary = results["summary"]
            risk = classify_run_risk(summary)

            batch_rows.append(
                {
                    "Run #": idx,
                    "Plan": summary.plan_name,
                    "Payroll file": p_file.name,
                    "Recordkeeper file": rk_file.name,
                    "Payroll vendor": summary.payroll_vendor,
                    "Recordkeeper vendor": summary.rk_vendor,
                    "Deferrals (Payroll)": summary.total_deferrals_payroll,
                    "Deferrals (RK)": summary.total_deferrals_rk,
                    "Deferral mismatches": summary.deferral_mismatch_count,
                    "Loan mismatches": summary.loan_mismatch_count,
                    "Late deferrals (rows)": summary.late_deferral_count,
                    "Risk": risk,
                    "Evidence pack": str(summary.evidence_pack_path),
                }
            )

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
    
    st.title("🔗 ProofLink – Payroll ↔ Recordkeeper Reconciliation")
    st.write(
        "Upload payroll and recordkeeper CSVs, run reconciliation, and download reports + manifests."
    )

    # ---------- Upload Section ----------
    st.header("1. Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Upload Payroll CSV")
        payroll_file = st.file_uploader(" ", type=["csv"], key="recon_payroll")

    with col2:
        st.caption("Upload Recordkeeper CSV")
        rk_file = st.file_uploader("  ", type=["csv"], key="recon_rk")

    # Plan name input
    plan_name = st.text_input("Plan name", value="Demo Plan")

    st.markdown("---")

    # ---------- Vendor Hints ----------
    with st.expander("Advanced Options: Vendor Hints"):
        payroll_vendor_hint = st.selectbox(
            "Payroll Vendor",
            ["Auto-detect", "ADP", "Paychex", "Paylocity", "Empower", "Generic"],
            index=0,
        )
        rk_vendor_hint = st.selectbox(
            "Recordkeeper Vendor",
            ["Auto-detect", "Empower", "Fidelity", "Vanguard", "Generic"],
            index=0,
        )

        payroll_vendor_hint = (
            None if payroll_vendor_hint == "Auto-detect" else payroll_vendor_hint
        )
        rk_vendor_hint = (
            None if rk_vendor_hint == "Auto-detect" else rk_vendor_hint
        )

    run_button = st.button(
        "▶️ Run Reconciliation", type="primary", use_container_width=True
    )

    # ---------- Run Reconciliation ----------
    if run_button:
        if not payroll_file or not rk_file:
            st.error("Please upload BOTH a payroll CSV and a recordkeeper CSV.")
            return

        # Run reconciliation with stdout capture
        with st.spinner("Running reconciliation..."):
            success, stdout, results = run_reconciliation_with_stdout_capture(
                payroll_file=payroll_file,
                rk_file=rk_file,
                plan_name=plan_name,
                payroll_vendor_hint=payroll_vendor_hint,
                rk_vendor_hint=rk_vendor_hint,
            )

        # Display run status
        if success and isinstance(results, dict) and results.get("summary"):
            st.success("Reconciliation run completed.")
            
            summary: RunSummary = results["summary"]
            results_dict = results.get("results_dict", {})
            
            # Parse stdout for additional metrics
            parsed_metrics = parse_reconciliation_output(stdout)
            
            # Check for missing columns warning
            all_totals_zero = (
                summary.total_deferrals_payroll == 0
                and summary.total_deferrals_rk == 0
                and summary.total_loans_payroll == 0
            )
            missing_columns_detected = "missing columns. Payroll missing" in stdout
            
            if all_totals_zero and missing_columns_detected:
                st.warning(
                    "The uploaded files are missing required deferral/loan columns "
                    "(e.g., def_amount, loan_amount), so metrics are 0. "
                    "This is a file-format issue, not a ProofLink engine error."
                )
            
            # Display metrics from RunSummary (primary source - these are the real values)
            st.subheader("Reconciliation Summary")
            
            # Risk level indicator
            if all_totals_zero and missing_columns_detected:
                st.markdown("**Run Risk Level:** :grey_question: N/A (invalid file format)")
            else:
                label, color = compute_run_risk_level(summary)
                st.markdown(f"**Run Risk Level:** :{color}_circle: {label}")
            
            st.divider()
            
            # Vendor info with confidence warnings
            col1, col2 = st.columns(2)
            col1.metric("Payroll Vendor", summary.payroll_vendor)
            if summary.payroll_vendor_confidence < 0.65:
                col1.warning(f"Low confidence: {summary.payroll_vendor_confidence:.2f}")
            col2.metric("Recordkeeper", summary.rk_vendor)
            if summary.rk_vendor_confidence < 0.65:
                col2.warning(f"Low confidence: {summary.rk_vendor_confidence:.2f}")
            
            st.divider()
            
            # Deferrals section
            st.markdown("### Deferrals")
            col3, col4, col5 = st.columns(3)
            col3.metric("Total Payroll Deferrals", f"${summary.total_deferrals_payroll:,.2f}")
            col4.metric("Total RK Deferrals", f"${summary.total_deferrals_rk:,.2f}")
            col5.metric("Deferral Mismatches", f"{summary.deferral_mismatch_count:,}")
            
            # Loans section
            st.markdown("### Loans")
            col6, col7, col8 = st.columns(3)
            col6.metric("Total Payroll Loans", f"${summary.total_loans_payroll:,.2f}")
            col7.metric("Total RK Loans", f"${summary.total_loans_rk:,.2f}")
            col8.metric("Loan Mismatches", f"{summary.loan_mismatch_count:,}")
            
            # Late contributions
            st.markdown("### Late Contributions")
            col9 = st.columns(1)[0]
            col9.metric("Late Deferral Rows", f"{summary.late_deferral_count:,}")
            
            # Display raw stdout
            st.divider()
            st.subheader("Raw Reconciliation Output")
            st.code(stdout, language="text")
            
            # Output files and evidence pack
            if summary.evidence_pack_path and summary.evidence_pack_path.exists():
                st.subheader("Output Files")
                st.info(f"Evidence pack available at: {summary.evidence_pack_path}")
                
                try:
                    with open(summary.evidence_pack_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Evidence Pack",
                            data=f.read(),
                            file_name=summary.evidence_pack_path.name,
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"Could not read evidence pack: {e}")
        else:
            st.error("Reconciliation run failed.")

            # Safe error inspection based only on `results`
            if isinstance(results, dict) and results.get("error"):
                st.error(f"Error: {results['error']}")
            elif isinstance(results, list):
                st.error(
                    f"Internal error: reconciliation returned a list instead of a dict (len={len(results)})."
                )
            elif results is not None:
                st.error(f"Internal error: unexpected results type: {type(results).__name__}")

            # Optionally, if stdout is non-empty, show it:
            if stdout:
                st.code(stdout, language="text")

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


    # ---------- Run History ----------
    st.markdown("---")
    st.header("2. View Run History")

    manifest_files = sorted(PROOFS_DIR.glob("proof_manifest_*.json"), reverse=True)

    if not manifest_files:
        st.info("No proof manifests found yet. Run a reconciliation to generate one.")
    else:
        st.write(f"Found {len(manifest_files)} proof manifest(s) in {PROOFS_DIR}")

        # Show the 10 most recent runs
        for path in manifest_files[:10]:
            # Derive a simple timestamp label from the filename
            ts_label = path.stem.replace("proof_manifest_", "")

            with st.expander(f"Run: {ts_label}  |  File: {path.name}"):
                st.code(str(path), language="text")

                # Download button for the manifest file
                try:
                    with open(path, "rb") as f:
                        st.download_button(
                            label="Download manifest JSON",
                            data=f,
                            file_name=path.name,
                            mime="application/json",
                            key=f"download_{path.name}",
                        )
                except Exception as e:
                    st.error(f"Unable to read {path.name}: {e}")


def render_run_history_tab():
    st.header("Run History")
    st.markdown("View summary of all previous reconciliation runs.")
    
    runs = load_run_history()
    
    if not runs:
        st.info("No prior runs found yet.")
        return
    
    # Prepare data for display
    display_data = []
    for run in runs:
        display_data.append({
            "Run Time": run["timestamp"],
            "Plan": run["plan_name"],
            "Risk": f":{run['risk_color']}_circle: {run['risk_label']}",
            "Deferral mismatches": run["deferral_mismatch_count"],
            "Loan mismatches": run["loan_mismatch_count"],
            "Late deferrals": run["late_deferral_count"],
            "Total issues": run["total_issues"],
            "Evidence pack": run["evidence_pack_path"],
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ==============================
#  Main entrypoint with tabs
# ==============================
def main():
    tab_recon, tab_timing, tab_batch, tab_history = st.tabs(
        ["Reconciliation", "Contribution Timing", "Batch Reconciliation", "Run History"]
    )

    with tab_recon:
        render_reconciliation_tab()

    with tab_timing:
        render_contribution_timing_tab()

    with tab_batch:
        render_batch_reconciliation_tab()
    
    with tab_history:
        render_run_history_tab()



if __name__ == "__main__":
    main()
