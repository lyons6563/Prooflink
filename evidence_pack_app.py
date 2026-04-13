"""
ProofLink – Evidence Pack Builder (Standalone)

Streamlit UI wrapper for the canonical Evidence Pack runner.
This is a thin wrapper that calls src.runner.run_evidence_pack().

Can be run directly with: streamlit run evidence_pack_app.py
"""

import streamlit as st
import os
from pathlib import Path
import sys

# Ensure repo root is in path for imports
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.runner import run_evidence_pack
from src.run_context import create_run_context

# Page config
st.set_page_config(
    page_title="ProofLink – Evidence Pack Builder",
    layout="wide"
)

st.title("ProofLink – Evidence Pack Builder")
st.write("Upload payroll, recordkeeper, and mapping files to generate a frozen Evidence Pack with integrity hashing.")

# File upload section
st.header("Upload Files")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Payroll CSV")
    payroll_file = st.file_uploader(
        "Upload Payroll CSV",
        type=["csv"],
        key="ep_payroll"
    )

with col2:
    st.caption("Recordkeeper CSV")
    rk_file = st.file_uploader(
        "Upload Recordkeeper CSV",
        type=["csv"],
        key="ep_rk"
    )

with col3:
    st.caption("Mapping YAML")
    mapping_file = st.file_uploader(
        "Upload Mapping YAML",
        type=["yaml", "yml"],
        key="ep_mapping"
    )

st.markdown("---")

# Run button
run_button = st.button(
    "▶️ Run Evidence Pack",
    type="primary",
    use_container_width=True
)

if run_button:
    # Validate uploads
    if not payroll_file or not rk_file or not mapping_file:
        st.error("Please upload all three files: Payroll CSV, Recordkeeper CSV, and Mapping YAML.")
        st.stop()
    
    # Generate run_id
    run_context = create_run_context()
    run_id = run_context.run_id
    
    # Create upload directory
    upload_dir = Path("tmp_run_outputs") / "ui_uploads" / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    payroll_path = upload_dir / (payroll_file.name or "payroll.csv")
    rk_path = upload_dir / (rk_file.name or "recordkeeper.csv")
    mapping_path = upload_dir / (mapping_file.name or "mapping.yaml")
    
    try:
        with open(payroll_path, "wb") as f:
            f.write(payroll_file.getbuffer())
        
        with open(rk_path, "wb") as f:
            f.write(rk_file.getbuffer())
        
        with open(mapping_path, "wb") as f:
            f.write(mapping_file.getbuffer())
        
        # Run Evidence Pack generation
        with st.spinner("Generating Evidence Pack..."):
            result = run_evidence_pack(
                payroll_path=str(payroll_path),
                recordkeeper_path=str(rk_path),
                mapping_path=str(mapping_path),
                out_root="tmp_run_outputs"
            )
        
        # Store result in session state
        st.session_state["ep_result"] = result
        st.session_state["ep_run_id"] = run_id
        
        st.success("Evidence Pack generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating Evidence Pack: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Display results if available
if "ep_result" in st.session_state:
    result = st.session_state["ep_result"]
    manifest = result.get("manifest", {})
    
    st.header("Evidence Pack Results")
    
    # Run ID
    st.subheader("Run Information")
    st.write(f"**Run ID:** `{result.get('run_id', 'N/A')}`")
    st.write(f"**Output Directory:** `{result.get('output_dir', 'N/A')}`")
    
    # Input hashes
    st.subheader("Input Hashes")
    inputs = manifest.get("inputs", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        payroll_hash = inputs.get("payroll_file_hash", "N/A")
        st.text_area("Payroll File Hash", payroll_hash, height=50, disabled=True)
    
    with col2:
        rk_hash = inputs.get("recordkeeper_file_hash", "N/A")
        st.text_area("Recordkeeper File Hash", rk_hash, height=50, disabled=True)
    
    with col3:
        mapping_hash = inputs.get("mapping_config_hash", "N/A")
        st.text_area("Mapping Config Hash", mapping_hash, height=50, disabled=True)
    
    # Output hashes
    st.subheader("Output Hashes")
    outputs = manifest.get("outputs", {})
    
    col1, col2 = st.columns(2)
    with col1:
        results_hash = outputs.get("results_hash", "N/A")
        st.text_area("Results Hash", results_hash, height=50, disabled=True)
    
    with col2:
        violations_hash = outputs.get("violations_hash", "N/A")
        st.text_area("Violations Hash", violations_hash, height=50, disabled=True)
    
    # ZIP hash
    st.subheader("Evidence Pack ZIP Hash")
    zip_hash = manifest.get("evidence_pack_zip_hash", "N/A")
    st.text_area("ZIP Hash", zip_hash, height=50, disabled=True)
    
    # Download button
    st.markdown("---")
    st.subheader("Download Evidence Pack")
    
    evidence_pack_zip_path = result.get("evidence_pack_zip_path")
    if evidence_pack_zip_path and Path(evidence_pack_zip_path).exists():
        with open(evidence_pack_zip_path, "rb") as f:
            zip_data = f.read()
        
        zip_filename = os.path.basename(evidence_pack_zip_path)
        
        st.download_button(
            label="Download Evidence Pack",
            data=zip_data,
            file_name=zip_filename,
            mime="application/zip",
            key="download_ep_zip"
        )
        
        st.caption(f"File: {zip_filename}")
        st.caption(f"Path: {evidence_pack_zip_path}")
    else:
        st.error(f"Evidence Pack ZIP not found at: {evidence_pack_zip_path}")
    
    # Full manifest (expandable)
    with st.expander("View Full Manifest"):
        st.json(manifest)

else:
    st.info("Upload files and click 'Run Evidence Pack' to generate an Evidence Pack.")

