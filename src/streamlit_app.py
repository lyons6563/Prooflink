import streamlit as st
from pathlib import Path
import pandas as pd

from main import run_reconciliation

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
st.set_page_config(page_title="ProofLink Reconciliation", layout="wide")

st.title("🔗 ProofLink – Payroll ↔ Recordkeeper Reconciliation")
st.write("Upload payroll and recordkeeper CSVs, run reconciliation, and download reports + manifests.")

# ---------- Upload Section ----------
st.header("1. Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.caption("Upload Payroll CSV")
    payroll_file = st.file_uploader(" ", type=["csv"], key="payroll_csv")

with col2:
    st.caption("Upload Recordkeeper CSV")
    rk_file = st.file_uploader("  ", type=["csv"], key="rk_csv")

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

    payroll_vendor_hint = None if payroll_vendor_hint == "Auto-detect" else payroll_vendor_hint
    rk_vendor_hint = None if rk_vendor_hint == "Auto-detect" else rk_vendor_hint

run_button = st.button("▶️ Run Reconciliation", type="primary", use_container_width=True)

# ---------- Run Reconciliation ----------
if run_button:
    if not payroll_file or not rk_file:
        st.error("Please upload BOTH a payroll CSV and a recordkeeper CSV.")
    else:
        with st.spinner("Running reconciliation..."):
            # Save uploaded files to RAW_DIR
            payroll_path = RAW_DIR / f"uploaded_{payroll_file.name}"
            rk_path = RAW_DIR / f"uploaded_{rk_file.name}"

            with open(payroll_path, "wb") as f:
                f.write(payroll_file.getbuffer())

            with open(rk_path, "wb") as f:
                f.write(rk_file.getbuffer())

                        # Call core engine
            results = run_reconciliation(
                payroll_csv=str(payroll_path),
                rk_csv=str(rk_path),
                payroll_vendor_hint=payroll_vendor_hint,
                rk_vendor_hint=rk_vendor_hint,
                output_dir=str(PROCESSED_DIR),
                proofs_dir=str(PROOFS_DIR),
            )

            # Attach latest proof manifest (if any) from proofs/
            manifest_files = sorted(
                PROOFS_DIR.glob("proof_manifest_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if manifest_files:
                # Most recently modified manifest
                results["latest_manifest"] = str(manifest_files[0])

        st.success("Reconciliation complete!")


        # ---------- KPI DASHBOARD ----------
        st.subheader("Key Metrics")

        def safe_len(path_key: str):
            """Helper: return row count of a CSV path in results dict, or 0."""
            p = Path(results.get(path_key, ""))
            if p.exists() and p.is_file():
                try:
                    df = pd.read_csv(p)
                    return len(df)
                except Exception:
                    return 0
            return 0

        late_deferrals_count = safe_len("late_deferrals")
        late_loans_count = safe_len("late_loans")
        deferral_mismatches_count = safe_len("deferral_mismatches")
        loan_mismatches_count = safe_len("loan_mismatches")
        only_in_payroll_count = safe_len("only_in_payroll")
        only_in_rk_count = safe_len("only_in_recordkeeper")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Late deferral rows", late_deferrals_count)
            st.metric("Late loan rows", late_loans_count)

        with col_b:
            st.metric("Deferral mismatches", deferral_mismatches_count)
            st.metric("Loan mismatches", loan_mismatches_count)

        with col_c:
            st.metric("Only-in-payroll rows", only_in_payroll_count)
            st.metric("Only-in-recordkeeper rows", only_in_rk_count)

        st.markdown("---")

        # ---------- Raw Summary Output ----------
        st.subheader("Summary Output (Paths)")
        st.json(results)

        # ---------- Downloads ----------
        st.subheader("Download Artifacts")

        for key, value in results.items():
            path = Path(value)
            if path.exists() and path.is_file():
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {key}",
                        data=f,
                        file_name=path.name,
                        mime="application/octet-stream",
                    )

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

