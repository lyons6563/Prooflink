"""
Evidence Pack assembly and freezing logic.

Assembles a frozen Evidence Pack containing run outputs, manifest, and
documentation, then creates a ZIP archive with integrity hashing.
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any

from .integrity import sha256_file


def build_evidence_pack(
    run_id: str,
    output_dir: str,
    manifest: dict,
    results_path: str | None = None,
    violations_path: str | None = None
) -> dict:
    """
    Assemble a frozen Evidence Pack for a run.
    Writes audit_summary.txt, README.txt, copies outputs if provided,
    zips the directory, hashes the ZIP, and returns updated manifest.
    
    Args:
        run_id: Unique run identifier
        output_dir: Directory containing the evidence pack contents
        manifest: Manifest dictionary to update with output hashes and ZIP hash
        results_path: Optional path to results CSV file
        violations_path: Optional path to violations CSV file
        
    Returns:
        Updated manifest dictionary with output hashes and evidence_pack_zip_hash
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure outputs dict exists in manifest
    if "outputs" not in manifest:
        manifest["outputs"] = {
            "results_hash": None,
            "violations_hash": None,
        }
    
    # Write audit_summary.txt
    execution_timestamp = manifest.get("execution_timestamp_utc", "unknown")
    audit_summary = f"""Evidence Pack Audit Summary
============================

Run ID: {run_id}
Execution Timestamp (UTC): {execution_timestamp}

System Status:
- Read-only: This Evidence Pack is a frozen snapshot of reconciliation results.
- Non-opinionated: The system performs rule-based analysis only. No compliance
  opinions, audit endorsements, or legal assertions are made.

This pack contains deterministic outputs from a single reconciliation run.
"""
    
    audit_summary_path = output_path / "audit_summary.txt"
    with audit_summary_path.open('w', encoding='utf-8') as f:
        f.write(audit_summary)
    
    # Write README.txt
    readme = """Evidence Pack README
===================

What is this Evidence Pack?
---------------------------
This Evidence Pack is a frozen, immutable snapshot of a single reconciliation
run. It contains:
- manifest.json: Run metadata and integrity hashes
- audit_summary.txt: Run identification and system status
- results.csv: Reconciliation results (if available)
- violations.csv: Exception/violation details (if available)

How to Verify Hashes
--------------------
All files in this pack have SHA-256 hashes recorded in manifest.json.

To verify a file's integrity:
1. Compute the SHA-256 hash of the file
2. Compare it to the hash in manifest.json
3. If hashes match, the file is unmodified

The evidence_pack_zip_hash in manifest.json is the hash of the entire ZIP
archive. Verify this to ensure the entire pack is intact.

What This Pack Does NOT Assert
-------------------------------
- Compliance verification or endorsement
- Audit readiness or audit approval
- Legal opinions or legal compliance
- Turnkey integration or production readiness
- Vendor-specific compatibility

This is a deterministic, rule-based analysis output only.
"""
    
    readme_path = output_path / "README.txt"
    with readme_path.open('w', encoding='utf-8') as f:
        f.write(readme)
    
    # Copy and hash results.csv if provided
    if results_path:
        results_file = Path(results_path)
        if results_file.exists():
            dest_results = output_path / "results.csv"
            shutil.copy2(results_file, dest_results)
            manifest["outputs"]["results_hash"] = sha256_file(str(dest_results))
    
    # Copy and hash violations.csv if provided
    if violations_path:
        violations_file = Path(violations_path)
        if violations_file.exists():
            dest_violations = output_path / "violations.csv"
            shutil.copy2(violations_file, dest_violations)
            manifest["outputs"]["violations_hash"] = sha256_file(str(dest_violations))
    
    # Write updated manifest.json to output_dir
    manifest_path = output_path / "manifest.json"
    import json
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # Create ZIP archive
    zip_filename = f"evidence_pack_{run_id}.zip"
    zip_path = output_path.parent / zip_filename
    
    # Create ZIP with all contents of output_dir
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_path.rglob('*'):
            if file_path.is_file():
                # Use relative path within the ZIP
                arcname = file_path.relative_to(output_path)
                zipf.write(file_path, arcname)
    
    # Hash the ZIP file
    manifest["evidence_pack_zip_hash"] = sha256_file(str(zip_path))
    
    return manifest

