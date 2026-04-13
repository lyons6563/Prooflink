"""
Manifest builder for Evidence Pack v2.

Builds and writes manifest.json files containing run metadata.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .run_context import RunContext, create_run_context
from .integrity import sha256_file


def build_manifest(run_context: RunContext) -> Dict[str, Any]:
    """
    Build a manifest dictionary from a RunContext.
    
    Args:
        run_context: RunContext instance with execution metadata
        
    Returns:
        Dictionary containing manifest data
    """
    return {
        "run_id": run_context.run_id,
        "engine": run_context.engine,
        "engine_version": run_context.engine_version,
        "execution_timestamp_utc": run_context.execution_timestamp_utc.isoformat(),
        "inputs": {
            "payroll_file_hash": None,
            "recordkeeper_file_hash": None,
            "mapping_config_hash": None,
        },
        "outputs": {
            "results_hash": None,
            "violations_hash": None,
        },
        "evidence_pack_zip_hash": None,
    }


def attach_input_hashes(
    manifest: dict,
    payroll_path: str | None,
    recordkeeper_path: str | None,
    mapping_path: str | None
) -> dict:
    """
    Populate manifest['inputs'] hash fields for any provided file paths.
    If a path is None or the file does not exist, leave the field as null.
    
    Args:
        manifest: Manifest dictionary to update
        payroll_path: Path to payroll CSV file (optional)
        recordkeeper_path: Path to recordkeeper CSV file (optional)
        mapping_path: Path to mapping YAML file (optional)
        
    Returns:
        Updated manifest dictionary
    """
    # Ensure inputs dict exists
    if "inputs" not in manifest:
        manifest["inputs"] = {
            "payroll_file_hash": None,
            "recordkeeper_file_hash": None,
            "mapping_config_hash": None,
        }
    
    # Hash payroll file if provided and exists
    if payroll_path:
        payroll_file = Path(payroll_path)
        if payroll_file.exists():
            manifest["inputs"]["payroll_file_hash"] = sha256_file(str(payroll_file))
    
    # Hash recordkeeper file if provided and exists
    if recordkeeper_path:
        rk_file = Path(recordkeeper_path)
        if rk_file.exists():
            manifest["inputs"]["recordkeeper_file_hash"] = sha256_file(str(rk_file))
    
    # Hash mapping file if provided and exists
    if mapping_path:
        mapping_file = Path(mapping_path)
        if mapping_file.exists():
            manifest["inputs"]["mapping_config_hash"] = sha256_file(str(mapping_file))
    
    return manifest


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> Path:
    """
    Write manifest.json to the specified output directory.
    
    Creates the directory if it doesn't exist.
    Writes manifest.json with pretty JSON formatting.
    
    Args:
        manifest: Manifest dictionary to write
        output_dir: Directory path where manifest.json will be written
        
    Returns:
        Path to the written manifest.json file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    manifest_file = output_path / "manifest.json"
    
    with manifest_file.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest_file


if __name__ == "__main__":
    """CLI self-check: create run context and write sample manifest."""
    print("=== Manifest Module Self-Check ===\n")
    
    # Create a new run context
    run_context = create_run_context()
    print(f"Created RunContext:")
    print(f"  run_id: {run_context.run_id}")
    print(f"  engine: {run_context.engine}")
    print(f"  engine_version: {run_context.engine_version}")
    print(f"  execution_timestamp_utc: {run_context.execution_timestamp_utc.isoformat()}\n")
    
    # Build manifest
    manifest = build_manifest(run_context)
    print("Built manifest (before hashes):")
    print(json.dumps(manifest, indent=2))
    print()
    
    # Attach input hashes using demo paths
    repo_root = Path(__file__).resolve().parent.parent
    payroll_path = repo_root / "data" / "raw" / "demo_broken_payroll.csv"
    recordkeeper_path = repo_root / "data" / "raw" / "demo_clean_rk.csv"
    mapping_path = repo_root / "mapping_example.yaml"
    
    # Convert to strings or None if files don't exist
    payroll_str = str(payroll_path) if payroll_path.exists() else None
    rk_str = str(recordkeeper_path) if recordkeeper_path.exists() else None
    mapping_str = str(mapping_path) if mapping_path.exists() else None
    
    manifest = attach_input_hashes(
        manifest,
        payroll_path=payroll_str,
        recordkeeper_path=rk_str,
        mapping_path=mapping_str
    )
    
    print("Manifest after attaching input hashes:")
    print(json.dumps(manifest, indent=2))
    print()
    
    # Write manifest
    output_dir = repo_root / "tmp_run_outputs" / "sample_run"
    manifest_path = write_manifest(manifest, str(output_dir))
    print(f"Manifest written to: {manifest_path.absolute()}")
    print()
    
    # Create dummy results.csv in a temp location first, then copy to output_dir
    temp_dir = repo_root / "tmp_run_outputs" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_results_file = temp_dir / "results.csv"
    temp_results_file.write_text("employee_id,pay_date,def_amount,status\n12345,2025-01-15,100.00,matched\n", encoding='utf-8')
    print(f"Created dummy results.csv: {temp_results_file}")
    print()
    
    # Build evidence pack
    from .evidence_pack import build_evidence_pack
    manifest = build_evidence_pack(
        run_id=run_context.run_id,
        output_dir=str(output_dir),
        manifest=manifest,
        results_path=str(temp_results_file),
        violations_path=None
    )
    
    print("Final manifest after building evidence pack:")
    print(json.dumps(manifest, indent=2))
    print()
    
    zip_filename = f"evidence_pack_{run_context.run_id}.zip"
    zip_path = output_dir.parent / zip_filename
    if zip_path.exists():
        print(f"Evidence pack ZIP created: {zip_path.absolute()}")
        print(f"ZIP hash: {manifest.get('evidence_pack_zip_hash', 'N/A')}")
    
    print("\nSelf-check complete.")

