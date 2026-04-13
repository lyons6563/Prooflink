#!/usr/bin/env python3
"""
Release packaging QA script.

Scans a target folder for forbidden runtime artifacts and build artifacts.
Exits with code 1 if any forbidden items are found, 0 if clean.
"""

import sys
from pathlib import Path
from fnmatch import fnmatch


def find_forbidden_artifacts(root_path: Path) -> list[Path]:
    """
    Scan directory tree for forbidden artifacts.
    
    Returns list of offending paths.
    """
    forbidden = []
    root_path = Path(root_path).resolve()
    
    if not root_path.exists():
        print(f"ERROR: Target path does not exist: {root_path}", file=sys.stderr)
        sys.exit(1)
    
    if not root_path.is_dir():
        print(f"ERROR: Target path is not a directory: {root_path}", file=sys.stderr)
        sys.exit(1)
    
    # Patterns to match
    forbidden_patterns = [
        # Directories (exact name match)
        ("api_uploads", True),
        ("streamlit_runs", True),
        ("__pycache__", True),
        ("output", True),
        (".venv", True),
        ("venv", True),
        
        # File patterns
        ("*.pyc", False),
        ("*.pyo", False),
        ("*.pyd", False),
        ("*.db", False),
        ("proof_manifest*.json", False),
        ("prooflink_evidence_pack*.zip", False),
        ("reconciliation_report*.xlsx", False),
    ]
    
    # Walk the directory tree
    for item in root_path.rglob("*"):
        if not item.exists():
            continue
            
        item_name = item.name
        item_path = item.relative_to(root_path)
        
        # Check directory patterns
        for pattern, is_dir in forbidden_patterns:
            if is_dir and item.is_dir():
                if item_name == pattern or fnmatch(item_name, pattern):
                    forbidden.append(item_path)
                    break
            elif not is_dir and item.is_file():
                if fnmatch(item_name, pattern):
                    forbidden.append(item_path)
                    break
    
    return sorted(set(forbidden))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Audit release folder for forbidden artifacts"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="./release_clean",
        help="Target folder to scan (default: ./release_clean)"
    )
    
    args = parser.parse_args()
    target_path = Path(args.target)
    
    forbidden = find_forbidden_artifacts(target_path)
    
    if forbidden:
        print("FAIL: Forbidden artifacts found in release folder:", file=sys.stderr)
        print("", file=sys.stderr)
        for path in forbidden:
            print(f"  {path}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Remove these artifacts before packaging.", file=sys.stderr)
        sys.exit(1)
    else:
        print("PASS: release folder clean")
        sys.exit(0)


if __name__ == "__main__":
    main()

