"""
Canonical runner for producing Evidence Pack ZIPs from reconciliation inputs.

This module provides a single entrypoint that runs the existing reconciliation
engine and produces a frozen Evidence Pack with integrity hashing.
"""

from pathlib import Path
from typing import Optional
import sys

# Ensure repo root is in path for both execution modes (python -m src.runner and python src/runner.py)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.run_context import create_run_context
from src.manifest import build_manifest, attach_input_hashes, write_manifest
from src.evidence_pack import build_evidence_pack


def run_evidence_pack(
    payroll_path: str,
    recordkeeper_path: str,
    mapping_path: str,
    out_root: str = "tmp_run_outputs"
) -> dict:
    """
    Runs the existing reconciliation/preflight logic and produces a frozen Evidence Pack ZIP.
    
    Args:
        payroll_path: Path to payroll CSV file
        recordkeeper_path: Path to recordkeeper CSV file
        mapping_path: Path to mapping YAML file
        out_root: Root directory for output (default: "tmp_run_outputs")
        
    Returns:
        dict with keys:
          - run_id: str
          - output_dir: str
          - evidence_pack_zip_path: str
          - manifest: dict
    """
    # Create run context
    run_context = create_run_context()
    run_id = run_context.run_id
    
    # Create run-specific output directory
    output_dir = Path(out_root) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build initial manifest
    manifest = build_manifest(run_context)
    
    # Attach input hashes
    manifest = attach_input_hashes(
        manifest,
        payroll_path=payroll_path,
        recordkeeper_path=recordkeeper_path,
        mapping_path=mapping_path
    )
    
    # Write manifest to run directory
    write_manifest(manifest, str(output_dir))
    
    # Run the existing reconciliation engine
    # Import here to avoid circular dependencies
    # Note: main.py is at the repo root, not in src/
    from main import run_reconciliation
    
    # Call run_reconciliation with the output_dir set to our run directory
    reconciliation_results = run_reconciliation(
        payroll_csv=payroll_path,
        rk_csv=recordkeeper_path,
        output_dir=str(output_dir),
        proofs_dir=str(output_dir / "proofs"),
        mapping_yaml_path=mapping_path
    )
    
    # Extract output file paths from reconciliation results
    # The engine writes files to output_dir, so we look for them there
    results_path = None
    violations_path = None
    
    # Look for reconciliation results CSV (combine deferral and loan mismatches if available)
    # For Evidence Pack, we'll use deferral_mismatches as the primary results
    deferral_mismatches_path = output_dir / "deferral_mismatches.csv"
    loan_mismatches_path = output_dir / "loan_mismatches.csv"
    late_deferrals_path = output_dir / "late_deferrals_contributions.csv"
    
    # Combine violations into a single file if multiple exist
    # For now, use deferral_mismatches as results, and create a combined violations file
    if deferral_mismatches_path.exists():
        results_path = str(deferral_mismatches_path)
    
    # Create a combined violations file if we have multiple violation types
    violation_files = []
    if deferral_mismatches_path.exists():
        violation_files.append(("deferral_mismatches", deferral_mismatches_path))
    if loan_mismatches_path.exists():
        violation_files.append(("loan_mismatches", loan_mismatches_path))
    if late_deferrals_path.exists():
        violation_files.append(("late_deferrals", late_deferrals_path))
    
    if violation_files:
        # For Evidence Pack v2, we'll use the first available violation file
        # In a more sophisticated version, we could combine them
        violations_path = str(violation_files[0][1])
    
    # Build the Evidence Pack
    manifest = build_evidence_pack(
        run_id=run_id,
        output_dir=str(output_dir),
        manifest=manifest,
        results_path=results_path,
        violations_path=violations_path
    )
    
    # Get the ZIP path
    zip_filename = f"evidence_pack_{run_id}.zip"
    evidence_pack_zip_path = str(Path(out_root) / zip_filename)
    
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "evidence_pack_zip_path": evidence_pack_zip_path,
        "manifest": manifest
    }


if __name__ == "__main__":
    """Smoke test: run with demo files if available."""
    print("=== Evidence Pack Runner Smoke Test ===\n")
    
    repo_root = Path(__file__).resolve().parent.parent
    
    # Try to use demo paths
    payroll_path = repo_root / "data" / "raw" / "demo_broken_payroll.csv"
    recordkeeper_path = repo_root / "data" / "raw" / "demo_clean_rk.csv"
    mapping_path = repo_root / "mapping_example.yaml"
    
    # Convert to strings or None if files don't exist
    payroll_str = str(payroll_path) if payroll_path.exists() else None
    rk_str = str(recordkeeper_path) if recordkeeper_path.exists() else None
    mapping_str = str(mapping_path) if mapping_path.exists() else None
    
    if not all([payroll_str, rk_str, mapping_str]):
        print("Error: Demo files not found. Expected:")
        print(f"  Payroll: {payroll_path}")
        print(f"  Recordkeeper: {recordkeeper_path}")
        print(f"  Mapping: {mapping_path}")
        print("\nPlease provide valid file paths.")
        exit(1)
    
    print(f"Payroll file: {payroll_str}")
    print(f"Recordkeeper file: {rk_str}")
    print(f"Mapping file: {mapping_str}")
    print()
    
    try:
        result = run_evidence_pack(
            payroll_path=payroll_str,
            recordkeeper_path=rk_str,
            mapping_path=mapping_str
        )
        
        print("=== Evidence Pack Generated ===")
        print(f"Run ID: {result['run_id']}")
        print(f"Output directory: {result['output_dir']}")
        print(f"Evidence Pack ZIP: {result['evidence_pack_zip_path']}")
        print(f"ZIP Hash: {result['manifest'].get('evidence_pack_zip_hash', 'N/A')}")
        print()
        print("Smoke test complete.")
        
    except Exception as e:
        print(f"Error during Evidence Pack generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

