# main.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from datetime import datetime
import hashlib
import json
import os
import numpy as np
import re
import traceback
import zipfile

import pandas as pd

from vendors import (
    PAYROLL_VENDOR_SIGNATURES,
    RK_VENDOR_SIGNATURES,
    apply_vendor_column_mapping,
)

from vendor_detection import detect_vendors, VendorDetectionResult

from contribution_timing_analyzer_v2 import run_timing_analysis

from eligibility_drift import detect_eligibility_drift

from secure20_catchup_analyzer import analyze_secure20_catchup

from eligibility_drift_analyzer import analyze_eligibility_drift

from comp_402g_analyzer import analyze_comp_402g_limits

from match_reasonableness_analyzer import analyze_match_reasonableness

from plan_exception_summary import build_plan_exception_summary


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
    eligibility_drift_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RunSummary to a JSON-serializable dict."""
        return {
            "plan_name": self.plan_name,
            "payroll_vendor": self.payroll_vendor,
            "rk_vendor": self.rk_vendor,
            "payroll_vendor_confidence": self.payroll_vendor_confidence,
            "rk_vendor_confidence": self.rk_vendor_confidence,
            "total_deferrals_payroll": self.total_deferrals_payroll,
            "total_deferrals_rk": self.total_deferrals_rk,
            "total_loans_payroll": self.total_loans_payroll,
            "total_loans_rk": self.total_loans_rk,
            "deferral_mismatch_count": self.deferral_mismatch_count,
            "loan_mismatch_count": self.loan_mismatch_count,
            "late_deferral_count": self.late_deferral_count,
            "eligibility_drift_count": self.eligibility_drift_count,
            "evidence_pack_path": str(self.evidence_pack_path) if self.evidence_pack_path else "",
            "run_id": self.run_id,
        }


@dataclass
class EngineConfig:
    """Configuration for the ProofLink engine."""
    plan_name: str
    late_threshold_days: int = 5
    secure2_enabled: bool = True
    payroll_vendor_hint: Optional[str] = None
    rk_vendor_hint: Optional[str] = None
    output_dir: str = "data/processed"
    proofs_dir: str = "proofs"


@dataclass
class EngineResult:
    """Result from running the ProofLink engine."""
    run_id: str
    summary: Dict[str, Any]
    evidence_pack_path: str
    manifest: Dict[str, Any]


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
            eligibility_drift_count=int(reconciliation_results.get("eligibility_drift_count", 0)),
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
            "eligibility_drift_count": run_summary.eligibility_drift_count,
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


def _find_timing_summary(run_root: str) -> Optional[Dict[str, Any]]:
    """
    Search for timing_summary.json under the given run_root directory.
    Returns the parsed dict if found and valid, otherwise None.
    """
    if not run_root or not os.path.isdir(run_root):
        return None

    for root, dirs, files in os.walk(run_root):
        if "timing_summary.json" in files:
            timing_path = os.path.join(root, "timing_summary.json")
            try:
                with open(timing_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                return None

    return None


def _infer_plan_year_from_payroll(payroll_df: pd.DataFrame) -> Optional[int]:
    """
    Infer plan year from payroll dataframe by finding the latest year in pay_date column.
    
    Args:
        payroll_df: Payroll dataframe with pay_date column
        
    Returns:
        Latest year found in pay_date, or None if cannot be determined
    """
    if "pay_date" not in payroll_df.columns or payroll_df.empty:
        return None
    dates = pd.to_datetime(payroll_df["pay_date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return None
    # Use the latest year present in the data as the plan_year
    return int(dates.dt.year.max())


def run_prooflink_engine(
    payroll_path: str,
    rk_path: str,
    config: EngineConfig,
    *,
    run_id: Optional[str] = None,
) -> EngineResult:
    """
    Core ProofLink engine entrypoint.

    Responsibilities:
    - Load input files
    - Run reconciliation
    - Run contribution timing analysis
    - Run Secure 2.0 checks
    - Build evidence pack ZIP
    - Assemble a JSON-serializable summary and manifest
    - Return EngineResult with:
        - run_id
        - summary dict
        - evidence pack filesystem path
        - manifest dict

    Args:
        payroll_path: Path to payroll CSV file
        rk_path: Path to recordkeeper CSV file
        config: EngineConfig with plan name, thresholds, and other settings
        run_id: Optional run ID (if not provided, will be generated)

    Returns:
        EngineResult containing run_id, summary dict, evidence_pack_path, and manifest dict
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Generate run_id if not provided
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert string paths to Path objects
    payroll_csv = Path(payroll_path)
    rk_csv = Path(rk_path)
    
    # Validate input files exist
    if not payroll_csv.exists():
        raise FileNotFoundError(f"Payroll CSV not found: {payroll_csv}")
    if not rk_csv.exists():
        raise FileNotFoundError(f"Recordkeeper CSV not found: {rk_csv}")
    
    # Run reconciliation (includes Secure 2.0 checks, timing analysis, evidence pack)
    reconciliation_results = run_reconciliation(
        payroll_csv=str(payroll_csv),
        rk_csv=str(rk_csv),
        payroll_vendor_hint=config.payroll_vendor_hint,
        rk_vendor_hint=config.rk_vendor_hint,
        output_dir=config.output_dir,
        proofs_dir=config.proofs_dir,
    )
    
    # Get the evidence pack path from results
    evidence_pack_path = reconciliation_results.get("evidence_pack", "")
    if not evidence_pack_path:
        raise ValueError("Evidence pack was not created by run_reconciliation")
    
    # Get manifest path and load it
    manifest_path = Path(reconciliation_results.get("manifest", ""))
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        logger.warning(f"Manifest file not found at {manifest_path}")
    
    # Build summary dict from reconciliation_results
    vendor_detection = reconciliation_results.get("vendor_detection", {})
    totals = reconciliation_results.get("totals", {})
    mismatches = reconciliation_results.get("mismatches", {})
    timing = reconciliation_results.get("timing", {})
    
    # Extract Secure 2.0 exceptions
    secure20_exceptions_raw = reconciliation_results.get("secure20_exceptions", [])
    secure20_exceptions = secure20_exceptions_raw if isinstance(secure20_exceptions_raw, list) else []
    secure20_exception_count = len(secure20_exceptions)
    secure20_exceptions_csv = reconciliation_results.get("secure20_exceptions_csv")
    
    # Run contribution timing analysis directly to ensure it executes and we capture metrics
    # Derive output directory from evidence_pack_path to ensure all outputs go to the same directory
    # evidence_pack_path example: "api_uploads\\<run_id>\\output\\prooflink_evidence_pack.zip"
    # output_dir should be: "api_uploads\\<run_id>\\output" (same directory as reconciliation outputs)
    if evidence_pack_path:
        output_dir_path = Path(evidence_pack_path).parent
    else:
        output_dir_path = Path(config.output_dir)
    output_dir = str(output_dir_path)  # Keep as string for timing analysis compatibility
    
    # Debug instrumentation for timing analysis
    timing_debug: Dict[str, Any] = {
        "called": False,
        "exception": None,
        "result_type": None,
        "result_keys": None,
        "payroll_path": str(payroll_csv),
        "rk_path": str(rk_csv),
        "output_dir": output_dir,
    }
    
    # Safe defaults for timing analysis results
    real_timing_metrics: Dict[str, Any] = {
        "total_rows": 0,
        "late_rows": 0,
        "missing_deposits": 0,
        "timing_risk": "N/A",
    }
    timing_result: Dict[str, Any] = {
        "late_contributions_path": "",
        "timing_summary_path": "",
        "total_rows": 0,
        "late_rows": 0,
        "missing_deposits": 0,
        "timing_risk": "N/A",
    }
    
    try:
        timing_debug["called"] = True
        timing_result = run_timing_analysis(
            payroll_path=str(payroll_csv),
            rk_path=str(rk_csv),
            output_dir=output_dir,
            late_threshold_days=5,  # Default Secure 2.0 threshold
        )
        
        timing_debug["result_type"] = str(type(timing_result))
        
        if isinstance(timing_result, dict):
            timing_debug["result_keys"] = list(timing_result.keys())
            real_timing_metrics = {
                "total_rows": timing_result.get("total_rows", 0),
                "late_rows": timing_result.get("late_rows", 0),
                "missing_deposits": timing_result.get("missing_deposits", 0),
                "timing_risk": timing_result.get("timing_risk", "N/A"),
            }
        else:
            timing_debug["result_keys"] = None
            # If result is not a dict, use defaults
            timing_result = {
                "late_contributions_path": "",
                "timing_summary_path": "",
                "total_rows": 0,
                "late_rows": 0,
                "missing_deposits": 0,
                "timing_risk": "N/A",
            }
            
    except Exception as exc:
        timing_debug["exception"] = repr(exc)
        # Log the error but continue - timing analysis failure should not crash the engine
        logger.warning(f"Timing analysis failed in run_prooflink_engine: {exc}")
        # timing_result and real_timing_metrics already have safe defaults set above
    
    # If timing analysis didn't produce results, try to get from reconciliation_results
    if not real_timing_metrics or real_timing_metrics.get("total_rows", 0) == 0:
        timing_metrics_raw = reconciliation_results.get("timing_metrics")
        if isinstance(timing_metrics_raw, dict) and timing_metrics_raw:
            real_timing_metrics = timing_metrics_raw.copy()
            # Ensure all keys exist
            real_timing_metrics.setdefault("total_rows", 0)
            real_timing_metrics.setdefault("late_rows", 0)
            real_timing_metrics.setdefault("missing_deposits", 0)
            real_timing_metrics.setdefault("timing_risk", "N/A")
    
    # Run Secure 2.0 catch-up analysis
    # Use the fully processed payroll dataframe from reconciliation (includes all derived columns)
    # Fall back to lightly-normalized version if processed df is not available
    processed_payroll_df = reconciliation_results.get("payroll_processed_df")
    if processed_payroll_df is None:
        # Fallback: load and normalize if processed df is missing
        payroll_df = pd.read_csv(payroll_csv)
        payroll_df = normalize_column_names(payroll_df)
        processed_payroll_df = payroll_df
    
    # Ensure output_dir_path exists (same directory as reconciliation, timing, and evidence pack)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Call Secure 2.0 catch-up analyzer - writes secure20_violations.csv to the same output directory
    # Uses fully processed payroll dataframe with derived columns (is_hce, catchup_pretax, catchup_roth, etc.)
    secure20_summary = analyze_secure20_catchup(
        payroll_df=processed_payroll_df,
        output_dir=output_dir_path,
    )
    
    # Run eligibility drift analysis
    # Uses the same fully processed payroll dataframe from reconciliation
    eligibility_summary = analyze_eligibility_drift(
        payroll_df=processed_payroll_df,
        output_dir=output_dir_path,
    )
    
    # Run 402(g) compensation limit analysis
    # Initialize default summary
    comp_402g_summary = {
        "total_participants": 0,
        "excess_violation_count": 0,
        "csv_path": None,
    }
    
    # Infer plan year for 402(g) analysis and exception summary
    inferred_plan_year = _infer_plan_year_from_payroll(processed_payroll_df)
    
    try:
        if inferred_plan_year is not None and not processed_payroll_df.empty:
            comp_402g_summary, comp_402g_csv_path = analyze_comp_402g_limits(
                processed_payroll_df,
                output_dir_path,
                inferred_plan_year,
            )
            # Normalize path to string if needed
            if comp_402g_csv_path is not None and "csv_path" in comp_402g_summary:
                comp_402g_summary["csv_path"] = str(comp_402g_csv_path)
            elif comp_402g_csv_path is not None:
                comp_402g_summary["csv_path"] = str(comp_402g_csv_path)
        else:
            comp_402g_summary = {
                "total_participants": 0,
                "excess_violation_count": 0,
                "csv_path": None,
                "warning": "402(g) analysis skipped because plan year could not be inferred from payroll data.",
            }
    except Exception as exc:
        print(f"[WARN] 402(g) analysis failed: {exc}")
        comp_402g_summary = {
            "total_participants": 0,
            "excess_violation_count": 0,
            "csv_path": None,
            "warning": f"402(g) analysis failed: {exc}",
        }
    
    # Run employer match reasonableness analysis
    # Initialize default summary
    match_summary = {
        "total_rows": 0,
        "under_match_count": 0,
        "over_match_count": 0,
        "csv_path": None,
    }
    
    # Default match configuration
    default_match_config = {
        "match_type": "percent_of_comp",
        "match_rate": 0.50,          # 50% match
        "match_cap_pct": 0.06,       # on first 6% of comp
        "absolute_tolerance": 5.00,  # $5
        "relative_tolerance_pct": 0.15,  # 15%
    }
    
    try:
        if processed_payroll_df is not None and not processed_payroll_df.empty:
            match_summary, match_csv_path = analyze_match_reasonableness(
                processed_payroll_df,
                output_dir_path,
                plan_config=default_match_config,
            )
            # Normalize csv_path into the summary if needed
            if match_csv_path is not None:
                match_summary["csv_path"] = str(match_csv_path)
        else:
            match_summary = {
                "total_rows": 0,
                "under_match_count": 0,
                "over_match_count": 0,
                "csv_path": None,
                "warning": "Match analysis skipped because processed payroll data is empty.",
            }
    except Exception as exc:
        print(f"[WARN] Match analysis failed: {exc}")
        match_summary = {
            "total_rows": 0,
            "under_match_count": 0,
            "over_match_count": 0,
            "csv_path": None,
            "warning": f"Match analysis failed: {exc}",
        }
    
    # Build plan exception summary from all analyzer CSV outputs
    # Extract CSV paths from each analyzer summary
    issue_csv_paths = {
        "secure20": secure20_summary.get("csv_path"),
        "eligibility": eligibility_summary.get("csv_path"),
        "comp_402g": comp_402g_summary.get("csv_path"),
        "match": match_summary.get("csv_path"),
    }
    
    try:
        plan_ex_summary, plan_ex_csv_path = build_plan_exception_summary(
            output_dir=output_dir_path,
            run_id=str(run_id),
            plan_name=config.plan_name,
            plan_year=inferred_plan_year,
            issue_csv_paths=issue_csv_paths,
        )
        if plan_ex_csv_path is not None:
            plan_ex_summary["csv_path"] = str(plan_ex_csv_path)
    except Exception as exc:
        print(f"[WARN] Plan exception summary failed: {exc}")
        plan_ex_summary = {
            "total_issues": 0,
            "by_category": {},
            "csv_path": None,
            "warning": f"Plan exception summary failed: {exc}",
        }
    
    summary = {
        "plan_name": config.plan_name,
        "payroll_vendor": vendor_detection.get("payroll_vendor", "Unknown / Generic"),
        "rk_vendor": vendor_detection.get("rk_vendor", "Unknown / Generic"),
        "payroll_vendor_confidence": float(vendor_detection.get("payroll_vendor_confidence", 0.0)),
        "rk_vendor_confidence": float(vendor_detection.get("rk_vendor_confidence", 0.0)),
        "total_deferrals_payroll": float(totals.get("deferrals_payroll", 0.0)),
        "total_deferrals_rk": float(totals.get("deferrals_rk", 0.0)),
        "total_loans_payroll": float(totals.get("loans_payroll", 0.0)),
        "total_loans_rk": float(totals.get("loans_rk", 0.0)),
        "deferral_mismatch_count": int(mismatches.get("deferral_count", 0)),
        "loan_mismatch_count": int(mismatches.get("loan_count", 0)),
        "late_deferral_count": int(timing.get("late_deferral_count", 0)),
        "eligibility_drift_count": int(eligibility_summary.get("eligibility_drift_count", reconciliation_results.get("eligibility_drift_count", 0))),
        "evidence_pack_path": evidence_pack_path,
        "run_id": run_id,
    }
    
    # Always include timing_metrics with real values (or defaults if not found)
    summary["timing_metrics"] = real_timing_metrics
    summary["timing_debug"] = timing_debug
    
    # Always include timing dict (even if empty, to ensure consistent summary structure)
    # Use timing from reconciliation_results if available, otherwise use empty dict with defaults
    if timing:
        summary["timing"] = timing
    else:
        # Provide default timing structure when timing analysis fails or is unavailable
        summary["timing"] = {
            "late_deferral_count": 0,
            "late_loan_count": 0,
        }
    
    # Include timing file paths if available (prefer from direct call, fallback to reconciliation_results)
    late_contributions_path = ""
    timing_summary_path = ""
    if timing_result and isinstance(timing_result, dict):
        late_contributions_path = timing_result.get("late_contributions_path", "")
        timing_summary_path = timing_result.get("timing_summary_path", "")
    if not late_contributions_path or not timing_summary_path:
        late_contributions_path = reconciliation_results.get("late_contributions", late_contributions_path)
        timing_summary_path = reconciliation_results.get("timing_summary", timing_summary_path)
    if late_contributions_path or timing_summary_path:
        summary["timing_files"] = {
            "timing_summary_json": timing_summary_path if timing_summary_path else None,
            "late_contributions_csv": late_contributions_path if late_contributions_path else None,
        }
    
    # Always include Secure 2.0 exceptions data
    summary["secure20_exceptions"] = secure20_exceptions
    summary["secure20_exception_count"] = secure20_exception_count
    
    # Include Secure 2.0 catch-up analysis summary
    summary["secure20"] = secure20_summary
    
    # Include Secure 2.0 file paths if available
    if secure20_exceptions_csv:
        summary["secure20_files"] = {
            "exceptions_csv": secure20_exceptions_csv,
        }
    
    # Include eligibility drift analysis summary
    summary["eligibility_drift"] = eligibility_summary
    
    # Include 402(g) compensation limit analysis summary
    summary["comp_402g"] = comp_402g_summary
    
    # Include employer match reasonableness analysis summary
    summary["match_checks"] = match_summary
    
    # Include plan exception summary (aggregated from all analyzers)
    summary["plan_exceptions"] = plan_ex_summary
    
    return EngineResult(
        run_id=run_id,
        summary=summary,
        evidence_pack_path=evidence_pack_path,
        manifest=manifest,
    )


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

    Returns:
        dict containing:
        - File paths (reconciliation_report, deferral_mismatches, loan_mismatches, etc.)
        - Metrics (totals, mismatches, timing, vendor_detection, etc.)
        - payroll_processed_df: pd.DataFrame with fully processed payroll data including
          derived columns (deferral_amount, loan_amount, is_hce, catchup_pretax, catchup_roth, etc.)
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
    
    # -------------------------------
    # Eligibility drift detection (run on raw payroll before any normalization)
    # -------------------------------
    try:
        drift_df = detect_eligibility_drift(payroll_df.copy(), grace_days=0)
    except Exception as e:
        print(f"[WARN] Eligibility drift detection failed: {e}")
        drift_df = None

    eligibility_drift_path = None
    eligibility_drift_count = 0

    if drift_df is not None and not drift_df.empty:
        os.makedirs(output_dir, exist_ok=True)
        eligibility_drift_path = os.path.join(output_dir, "eligibility_drift.csv")
        drift_df.to_csv(eligibility_drift_path, index=False)
        eligibility_drift_count = int(len(drift_df))
        print(f"[INFO] Eligibility drift report written to: {eligibility_drift_path}")
    else:
        print("[INFO] No eligibility drift rows detected.")
    
    rk_df = pd.read_csv(rk_path)
    
    # Normalize column names to handle flexible deferral/roth variants
    payroll_df = normalize_column_names(payroll_df)
    rk_df = normalize_column_names(rk_df)

    # =========================
    # Vendor detection with confidence
    # =========================
    vendor_detection_result = detect_vendors(
        payroll_df=payroll_df,
        rk_df=rk_df,
        payroll_vendor_hint=payroll_vendor_hint,
        rk_vendor_hint=rk_vendor_hint,
    )
    
    # Extract values for use in the rest of the function
    payroll_vendor = vendor_detection_result.payroll_vendor
    payroll_confidence = vendor_detection_result.payroll_confidence
    rk_vendor = vendor_detection_result.rk_vendor
    rk_confidence = vendor_detection_result.rk_confidence

    print("\n=== Vendor Detection ===")
    print(f"Detected payroll vendor:     {payroll_vendor} (confidence: {payroll_confidence:.2f})")
    print(f"Detected recordkeeper:       {rk_vendor} (confidence: {rk_confidence:.2f})")
    
    if payroll_confidence < 0.65 and not payroll_vendor_hint:
        print(f"[WARN] Low confidence ({payroll_confidence:.2f}) for payroll vendor detection. Manual verification recommended.")
    if rk_confidence < 0.65 and not rk_vendor_hint:
        print(f"[WARN] Low confidence ({rk_confidence:.2f}) for recordkeeper vendor detection. Manual verification recommended.")
    
    # Apply vendor-specific column mapping
    payroll_df = apply_vendor_column_mapping(payroll_df, payroll_vendor, PAYROLL_VENDOR_SIGNATURES)
    rk_df = apply_vendor_column_mapping(rk_df, rk_vendor, RK_VENDOR_SIGNATURES)

    # Fallback: if canonical deferral columns are missing, but demo headers exist, copy them over
    if "EE Deferral $" not in payroll_df.columns and "payroll_pretax" in payroll_df.columns:
        payroll_df["EE Deferral $"] = payroll_df["payroll_pretax"]

    if "EE Roth $" not in payroll_df.columns and "payroll_roth" in payroll_df.columns:
        payroll_df["EE Roth $"] = payroll_df["payroll_roth"]

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
    rk_mapping_safe = rk_cols or {}
    try:
        if hasattr(rk_mapping_safe, "items"):
            for src, dest in rk_mapping_safe.items():
                print(f"  {src} -> {dest}")
        else:
            for pair in rk_mapping_safe:
                try:
                    src, dest = pair
                    print(f"  {src} -> {dest}")
                except Exception:
                    print(f"  {pair}")
    except Exception as e:
        print(f"[WARN] Failed to print recordkeeper column mapping: {e}")

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

    # =========================
    # Secure 2.0 fields (payroll side only)
    # =========================
    secure2_fields_missing = []

    def _get_secure2_source(col_name: str) -> str | None:
        """
        Prefer a direct normalized column (e.g. 'is_hce', 'catchup_pretax', 'catchup_roth').
        If not present, fall back to the vendor mapping in payroll_cols.
        """
        # Direct column present?
        if col_name in p.columns:
            return col_name

        # Fall back to vendor-specific mapping
        mapped = payroll_cols.get(col_name)
        if mapped and mapped in p.columns:
            return mapped

        return None

    # is_hce (boolean)
    is_hce_src = _get_secure2_source("is_hce")
    if is_hce_src:
        p["is_hce"] = (
            p[is_hce_src]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "y": True,
                    "n": False,
                    "yes": True,
                    "no": False,
                }
            )
            .astype(bool)
        )
    else:
        p["is_hce"] = False
        secure2_fields_missing.append("is_hce")

    # catchup_pretax (amount)
    catchup_pretax_src = _get_secure2_source("catchup_pretax")
    if catchup_pretax_src:
        p["catchup_pretax"] = parse_amount(p[catchup_pretax_src])
    else:
        p["catchup_pretax"] = 0.0
        secure2_fields_missing.append("catchup_pretax")

    # catchup_roth (amount)
    catchup_roth_src = _get_secure2_source("catchup_roth")
    if catchup_roth_src:
        p["catchup_roth"] = parse_amount(p[catchup_roth_src])
    else:
        p["catchup_roth"] = 0.0
        secure2_fields_missing.append("catchup_roth")

    # Warn if any Secure 2.0 fields are missing
    if secure2_fields_missing:
        print(
            f"[WARN] Secure 2.0 fields not found in payroll file "
            f"({', '.join(secure2_fields_missing)}). Secure 2.0 checks will be limited."
        )

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
    # Secure 2.0 compliance checks
    # =========================
    secure20_exceptions = check_secure20_catchup(p)
    
    # Write Secure 2.0 exceptions to CSV if any violations found
    secure20_exceptions_path = output_dir_path / "secure20_exceptions.csv"
    if secure20_exceptions:
        secure20_df = pd.DataFrame(secure20_exceptions)
        secure20_df.to_csv(secure20_exceptions_path, index=False)
        print(f"Secure 2.0 exceptions written to: {secure20_exceptions_path} ({len(secure20_exceptions)} violations)")
    else:
        # Don't create file if no exceptions
        secure20_exceptions_path = None

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
    # secure20_exceptions_path is set earlier if exceptions exist

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

    # Use the same values that are printed in the console output
    # This ensures vendor_detection dict matches what's displayed in === Vendor Detection ===
    # Use the VendorDetectionResult that was already computed above
    vendor_detection = {
        "payroll_vendor": vendor_detection_result.payroll_vendor,
        "rk_vendor": vendor_detection_result.rk_vendor,
        "payroll_vendor_confidence": vendor_detection_result.payroll_confidence,
        "rk_vendor_confidence": vendor_detection_result.rk_confidence,
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
        "secure20_exceptions": secure20_exceptions,
        "eligibility_drift_count": eligibility_drift_count,
    }
    
    # Add Secure 2.0 exceptions CSV path if file was created
    if secure20_exceptions_path is not None:
        results["secure20_exceptions_csv"] = str(secure20_exceptions_path)
    
    # Add eligibility drift CSV path if file was created
    results["eligibility_drift"] = str(eligibility_drift_path) if eligibility_drift_path else None

    # Run timing analysis and add results
    try:
        timing_result = run_timing_analysis(
            payroll_path=str(payroll_path),
            rk_path=str(rk_path),
            output_dir=str(output_dir_path),
            late_threshold_days=5,  # Default Secure 2.0 threshold
        )
        # Add timing file paths to results
        results["late_contributions"] = timing_result.get("late_contributions_path", "")
        results["timing_summary"] = timing_result.get("timing_summary_path", "")
        # Store timing metrics for UI
        results["timing_metrics"] = {
            "total_rows": timing_result.get("total_rows", 0),
            "late_rows": timing_result.get("late_rows", 0),
            "missing_deposits": timing_result.get("missing_deposits", 0),
            "timing_risk": timing_result.get("timing_risk", "N/A"),
        }
    except Exception as e:
        print(f"[WARN] Timing analysis failed: {e}")
        # Continue without timing results - don't break reconciliation
        results["late_contributions"] = ""
        results["timing_summary"] = ""
        results["timing_metrics"] = {}

    # Build consolidated evidence pack ZIP
    evidence_zip = build_evidence_pack(results)
    results["evidence_pack"] = str(evidence_zip)

    print("\nRun complete. Key outputs:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # Add the fully processed payroll dataframe to results
    # This includes all derived columns (deferral_amount, loan_amount, is_hce, catchup_pretax, catchup_roth, etc.)
    results["payroll_processed_df"] = p

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
      - Secure 2.0 exceptions CSV (if present)
      - manifest JSON (if present)
      - eligibility drift CSV (if present)
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
        "secure20_exceptions_csv",
        "manifest",
        "late_contributions",
        "timing_summary",
        "eligibility_drift",
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
        # Note: "Employee_ID" and "Participant_ID" will match via case-insensitive matching
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
        # Note: "Pay_Date" will match via case-insensitive matching
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
        # Note: "Deposit_Date" will match via case-insensitive matching
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
        "ee_pretax_def",  # New: EE_PreTax_Def
        "payroll_pretax",  # Canonical name alias
    ],
    "payroll_roth": [
        "roth_defl",
        "ee roth $",
        "ee roth",  # Without $ sign
        "EE Roth $",  # Exact normalized name
        "roth contribution",
        "roth",
        "457b_ee_roth_amt",
        "ee_roth_def",  # New: EE_Roth_Def
        "payroll_roth",  # Canonical name alias
    ],
    "payroll_loan": [
        "loan_pmt",
        "loan repay $",
        "loan repayment",
        "457b_loan_repay_amt",         # <-- add this line
        "loan_repayment",  # New: Loan_Repayment
    ],

    # Secure 2.0 fields (payroll side only)
    "is_hce": [
        "is_hce",
        "is hce",
        "hce_flag",
        "hce flag",
        "hce",
        "highly_compensated",
        "highly compensated",
        "highly compensated employee",
    ],
    "catchup_pretax": [
        "catchup_pretax",
        "catchup pretax",
        "catch_up_pretax",
        "catch up pretax",
        "pretax catchup",
        "pretax catch-up",
        "catchup_contribution_pretax",
    ],
    "catchup_roth": [
        "catchup_roth",
        "catchup roth",
        "catch_up_roth",
        "catch up roth",
        "roth catchup",
        "roth catch-up",
        "catchup_contribution_roth",
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
        "ee_pretax_def",  # New: EE_PreTax_Def
    ],
    "rk_roth": [
        "ee_roth",
        "ee roth $",
        "ee roth",  # Without $ sign
        "EE Roth $",  # Exact normalized name
        "roth contribution",
        "roth cont",
        "roth",
        "ee_roth_def",  # New: EE_Roth_Def
    ],
    "rk_loan": [
        "loan_contr",
        "loan repayment",
        "loan_repayment",  # New: Loan_Repayment
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
        "total_deposit_amount",  # New: Total_Deposit_Amount
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
            # Secure 2.0 fields (optional)
            "is_hce": ["Is_HCE", "HCE_Flag", "Highly_Compensated"],
            "catchup_pretax": ["Catchup_Pretax", "Catch_Up_Pretax"],
            "catchup_roth": ["Catchup_Roth", "Catch_Up_Roth"],
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
        "ee_pretax_def",  # New variant: EE_PreTax_Def
    }
    
    # Roth column variants
    roth_variants = {
        "roth_amount",
        "roth_deferral",
        "roth",
        "roth_contribution",
        "ee_roth_def",  # New variant: EE_Roth_Def
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


def check_secure20_catchup(payroll_df: pd.DataFrame) -> list[dict]:
    """
    Basic Secure 2.0 check on the payroll side.
    
    Checks for HCEs (Highly Compensated Employees) with pre-tax catch-up contributions.
    Secure 2.0 requires that HCEs use Roth catch-up contributions, not pre-tax.
    
    Args:
        payroll_df: Payroll dataframe with normalized columns:
            - employee_id
            - pay_date
            - is_hce (boolean)
            - catchup_pretax (float)
            - catchup_roth (float)
    
    Returns:
        List of exception dicts for HCEs with pre-tax catch-up. Each dict contains:
            - employee_id
            - pay_date
            - catchup_pretax
            - catchup_roth
            - issue_code: "HCE_PRETAX_CATCHUP"
            - description: Human-readable message
        
        Returns empty list if no violations found.
    """
    exceptions = []
    
    for idx, row in payroll_df.iterrows():
        hce_flag = str(row.get("is_hce", "")).strip().upper()
        is_hce = hce_flag in ("1", "Y", "YES", "TRUE")
        
        age = row.get("age", None)
        try:
            age = int(age) if age not in (None, "") else None
        except Exception:
            age = None
        
        pretax = float(row.get("EE Deferral $", 0.0) or 0.0)
        roth = float(row.get("EE Roth $", 0.0) or 0.0)
        
        # DEMO-SAFE SECURE 2.0 RULE:
        # HCE, age >= 50, some pre-tax, no Roth
        if is_hce and age is not None and age >= 50 and pretax > 0 and roth == 0:
            # Employee ID: prefer normalized 'employee_id', fall back to 'EmpNumber'
            if "employee_id" in payroll_df.columns:
                employee_id_val = row.get("employee_id", "")
            elif "EmpNumber" in payroll_df.columns:
                employee_id_val = row.get("EmpNumber", "")
            else:
                employee_id_val = ""

            # Pay date: prefer normalized 'pay_date', fall back to 'Payroll_Run_Date'
            if "pay_date" in payroll_df.columns:
                pay_date_val = row.get("pay_date", "")
            elif "Payroll_Run_Date" in payroll_df.columns:
                pay_date_val = row.get("Payroll_Run_Date", "")
            else:
                pay_date_val = ""

            if pd.isna(employee_id_val):
                employee_id_val = ""
            if pd.isna(pay_date_val):
                pay_date_val = ""
            
            exceptions.append(
                {
                    "employee_id": str(employee_id_val),
                    "age": age,
                    "is_hce": hce_flag,
                    "pretax_amount": pretax,
                    "roth_amount": roth,
                    "reason": "HCE age 50+ with pre-tax deferrals and no Roth catch-up (demo rule)",
                }
            )
    
    return exceptions

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
    # Normalize join key types to avoid pandas merge dtype errors
    if "employee_id" in payroll_norm.columns:
        payroll_norm["employee_id"] = payroll_norm["employee_id"].astype(str)
    if "employee_id" in rk_norm.columns:
        rk_norm["employee_id"] = rk_norm["employee_id"].astype(str)

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
    """
    Legacy wrapper function that reads config and calls run_reconciliation_with_summary.
    
    This function maintains backward compatibility for callers that expect it to:
    - Read file paths from config
    - Use default output directories
    - Return None (no return value)
    
    All reconciliation logic has been moved to run_reconciliation_with_summary().
    """
    # Load config and extract parameters
    cfg = load_config()
    global MAX_BUSINESS_DAYS_LAG
    
    # Set global MAX_BUSINESS_DAYS_LAG from config (run_reconciliation also does this, but set it here for consistency)
    MAX_BUSINESS_DAYS_LAG = cfg.get("max_business_days_lag", MAX_BUSINESS_DAYS_LAG)
    
    # Get file names from config and convert to full paths
    payroll_filename = cfg.get("payroll_file", PAYROLL_FILE)
    rk_filename = cfg.get("recordkeeper_file", RECORDKEEPER_FILE)
    payroll_csv = DATA_RAW / payroll_filename
    rk_csv = DATA_RAW / rk_filename
    
    # Extract optional parameters from config
    plan_name = cfg.get("plan_name", "Unknown Plan")
    payroll_vendor_hint = cfg.get("payroll_vendor_hint")
    rk_vendor_hint = cfg.get("rk_vendor_hint")
    
    # Use default output directory (DATA_OUT)
    output_dir = DATA_OUT
    
    # Call the canonical reconciliation function
    # Note: We ignore the return value to maintain backward compatibility (returns None)
    run_reconciliation_with_summary(
        payroll_csv=payroll_csv,
        rk_csv=rk_csv,
        output_dir=output_dir,
        plan_name=plan_name,
        payroll_vendor_hint=payroll_vendor_hint,
        rk_vendor_hint=rk_vendor_hint,
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
    import sys
    import argparse

    # Check if any CLI arguments were provided
    # If --payroll_csv is present, assume user wants CLI mode
    has_cli_args = len(sys.argv) > 1 and "--payroll_csv" in sys.argv

    if has_cli_args:
        # CLI mode: parse arguments and run engine
        parser = argparse.ArgumentParser(description="ProofLink Reconciliation Engine")
        parser.add_argument("--payroll_csv", required=True, help="Path to payroll CSV file")
        parser.add_argument("--rk_csv", required=True, help="Path to recordkeeper CSV file")
        parser.add_argument("--plan_name", default="Unknown Plan", help="Plan name")
        parser.add_argument("--output_dir", default="data/processed", help="Output directory")
        parser.add_argument("--proofs_dir", default="proofs", help="Proofs directory")
        parser.add_argument("--late_threshold_days", type=int, default=5, help="Late contribution threshold in days")
        parser.add_argument("--payroll_vendor_hint", help="Optional payroll vendor hint")
        parser.add_argument("--rk_vendor_hint", help="Optional recordkeeper vendor hint")
        args = parser.parse_args()

        # Create engine config
        config = EngineConfig(
            plan_name=args.plan_name,
            late_threshold_days=args.late_threshold_days,
            secure2_enabled=True,
            payroll_vendor_hint=args.payroll_vendor_hint,
            rk_vendor_hint=args.rk_vendor_hint,
            output_dir=args.output_dir,
            proofs_dir=args.proofs_dir,
        )

        # Run engine
        result = run_prooflink_engine(
            payroll_path=args.payroll_csv,
            rk_path=args.rk_csv,
            config=config,
        )

        # Print results
        print("\n" + "="*60)
        print("ProofLink Engine Run Complete")
        print("="*60)
        print(f"Run ID: {result.run_id}")
        print(f"Evidence Pack: {result.evidence_pack_path}")
        print("\nSummary:")
        for key, value in result.summary.items():
            if key != "run_id":  # Already printed
                print(f"  {key}: {value}")
        print(f"\nManifest: {len(result.manifest)} keys")
    else:
        # No CLI args: use config-based wrapper (legacy behavior)
        reconcile_payroll_vs_recordkeeper()


    

