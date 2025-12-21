"""
Manifest builder for Evidence Pack v2.

Builds and writes manifest.json files containing run metadata.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .run_context import RunContext, create_run_context


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
    }


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
    print("Built manifest:")
    print(json.dumps(manifest, indent=2))
    print()
    
    # Write manifest
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "tmp_run_outputs" / "sample_run"
    manifest_path = write_manifest(manifest, str(output_dir))
    print(f"Manifest written to: {manifest_path.absolute()}")
    print("\nSelf-check complete.")

