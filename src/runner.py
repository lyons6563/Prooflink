"""
Canonical runner for producing Evidence Pack ZIPs from reconciliation inputs.

This module provides a single entrypoint that runs the existing reconciliation
engine and produces a frozen Evidence Pack with integrity hashing.
"""

from pathlib import Path
from typing import Optional
import sys
import argparse

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
    """CLI entrypoint for Evidence Pack generation."""
    parser = argparse.ArgumentParser(
        description="Generate Evidence Pack ZIP from reconciliation inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use provided file paths
  python -m src.runner demo/demo_payroll.csv demo/demo_recordkeeper.csv demo/demo_mapping.yaml
  
  # Use default demo files (if available)
  python -m src.runner
        """
    )
    
    parser.add_argument(
        "payroll_csv_path",
        nargs="?",
        help="Path to payroll CSV file (default: data/raw/demo_broken_payroll.csv)"
    )
    parser.add_argument(
        "recordkeeper_csv_path",
        nargs="?",
        help="Path to recordkeeper CSV file (default: data/raw/demo_clean_rk.csv)"
    )
    parser.add_argument(
        "mapping_yaml_path",
        nargs="?",
        help="Path to mapping YAML file (default: mapping_example.yaml)"
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parent.parent
    
    # Use provided arguments or fall back to defaults
    # Priority: CLI arguments > defaults
    if args.payroll_csv_path is not None:
        payroll_path = Path(args.payroll_csv_path)
        if not payroll_path.is_absolute():
            payroll_path = repo_root / payroll_path
    else:
        payroll_path = repo_root / "data" / "raw" / "demo_broken_payroll.csv"
    
    if args.recordkeeper_csv_path is not None:
        recordkeeper_path = Path(args.recordkeeper_csv_path)
        if not recordkeeper_path.is_absolute():
            recordkeeper_path = repo_root / recordkeeper_path
    else:
        recordkeeper_path = repo_root / "data" / "raw" / "demo_clean_rk.csv"
    
    if args.mapping_yaml_path is not None:
        mapping_path = Path(args.mapping_yaml_path)
        if not mapping_path.is_absolute():
            mapping_path = repo_root / mapping_path
    else:
        mapping_path = repo_root / "mapping_example.yaml"
    
    # Validate files exist
    if not payroll_path.exists():
        print(f"Error: Payroll file not found: {payroll_path}")
        exit(1)
    
    if not recordkeeper_path.exists():
        print(f"Error: Recordkeeper file not found: {recordkeeper_path}")
        exit(1)
    
    if not mapping_path.exists():
        print(f"Error: Mapping file not found: {mapping_path}")
        exit(1)
    
    # Convert to absolute paths for consistent output
    payroll_str = str(payroll_path.resolve())
    rk_str = str(recordkeeper_path.resolve())
    mapping_str = str(mapping_path.resolve())
    
    print("=== Evidence Pack Runner ===\n")
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
        print("Evidence Pack generation complete.")
        
    except Exception as e:
        print(f"Error during Evidence Pack generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

