"""
Preflight checks for ProofLink reconciliation engine.

This module provides safety checks that must pass before the reconciliation
process can execute.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def normalize_column_name(col_name: str) -> str:
    """
    Normalize column name for matching: lowercase, strip whitespace, normalize internal whitespace to underscores.
    """
    if not col_name:
        return ""
    # Strip leading/trailing whitespace
    normalized = col_name.strip()
    # Replace any whitespace (spaces, tabs) with underscores
    normalized = re.sub(r'\s+', '_', normalized)
    # Convert to lowercase
    return normalized.lower()


def load_mapping_yaml(mapping_yaml_path: str) -> Dict[str, Any]:
    """
    Load mapping YAML file. Falls back to JSON if YAML not available.
    """
    path = Path(mapping_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_yaml_path}")
    
    if HAS_YAML:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Fallback to JSON
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def extract_examples_from_mapping(mapping_data: Dict[str, Any], file_type: str) -> Dict[str, List[str]]:
    """
    Extract example column names from mapping YAML for a given file type (payroll or recordkeeper).
    
    Returns dict mapping canonical field names to lists of example source column names.
    """
    examples_map = {}
    
    if file_type not in mapping_data:
        return examples_map
    
    file_section = mapping_data[file_type]
    if not isinstance(file_section, dict):
        return examples_map
    
    for field_key, field_data in file_section.items():
        if isinstance(field_data, dict) and 'examples' in field_data:
            canonical = field_data.get('canonical', field_key)
            examples = field_data.get('examples', [])
            if isinstance(examples, list):
                examples_map[canonical] = [normalize_column_name(ex) for ex in examples]
    
    return examples_map


def map_headers_to_canonical(
    source_headers: List[str],
    examples_map: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Map source headers to canonical field names using exact match (case-insensitive + whitespace normalized).
    
    Returns dict mapping canonical field names to source column names.
    """
    mapped = {}
    normalized_source = {normalize_column_name(h): h for h in source_headers}
    
    for canonical, examples in examples_map.items():
        for example_normalized in examples:
            if example_normalized in normalized_source:
                mapped[canonical] = normalized_source[example_normalized]
                break
    
    return mapped


def compute_join_key_coverage(csv_path: str, employee_id_col: str) -> float:
    """
    Compute percentage of rows where employee_id is non-empty.
    Reads the employee_id column efficiently.
    """
    try:
        # Read only the employee_id column for efficiency
        df = pd.read_csv(csv_path, usecols=[employee_id_col], dtype=str)
        if len(df) == 0:
            return 0.0
        
        # Count non-empty values (not NaN, not empty string after strip)
        non_empty = df[employee_id_col].astype(str).str.strip().ne('').sum()
        return (non_empty / len(df)) * 100.0
    except Exception as e:
        # If we can't read, return 0 (will be reported as warning)
        return 0.0


def run_preflight(
    payroll_csv_path: str,
    recordkeeper_csv_path: str,
    mapping_yaml_path: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run preflight checks on payroll and recordkeeper files.
    
    Args:
        payroll_csv_path: Path to payroll CSV file
        recordkeeper_csv_path: Path to recordkeeper CSV file
        mapping_yaml_path: Path to mapping YAML file
    
    Returns:
        Tuple of (safe: bool, report: dict)
        report contains:
            - missing_fields: dict with 'payroll' and 'recordkeeper' lists of missing required fields
            - mapped_fields: dict with 'payroll' and 'recordkeeper' dicts of canonical -> source mappings
            - join_key_coverage: dict with 'payroll' and 'recordkeeper' coverage percentages
            - warnings: list of warning messages
    """
    report = {
        'missing_fields': {'payroll': [], 'recordkeeper': []},
        'mapped_fields': {'payroll': {}, 'recordkeeper': {}},
        'join_key_coverage': {'payroll': 0.0, 'recordkeeper': 0.0},
        'warnings': []
    }
    
    # Load mapping YAML
    try:
        mapping_data = load_mapping_yaml(mapping_yaml_path)
    except Exception as e:
        report['warnings'].append(f"Failed to load mapping file: {e}")
        return (False, report)
    
    # Extract examples for payroll and recordkeeper
    payroll_examples = extract_examples_from_mapping(mapping_data, 'payroll')
    rk_examples = extract_examples_from_mapping(mapping_data, 'recordkeeper')
    
    # Read CSV headers only
    try:
        payroll_df = pd.read_csv(payroll_csv_path, nrows=0)
        payroll_headers = list(payroll_df.columns)
    except Exception as e:
        report['warnings'].append(f"Failed to read payroll file headers: {e}")
        return (False, report)
    
    try:
        rk_df = pd.read_csv(recordkeeper_csv_path, nrows=0)
        rk_headers = list(rk_df.columns)
    except Exception as e:
        report['warnings'].append(f"Failed to read recordkeeper file headers: {e}")
        return (False, report)
    
    # Map headers to canonical fields
    payroll_mapped = map_headers_to_canonical(payroll_headers, payroll_examples)
    rk_mapped = map_headers_to_canonical(rk_headers, rk_examples)
    
    report['mapped_fields']['payroll'] = payroll_mapped
    report['mapped_fields']['recordkeeper'] = rk_mapped
    
    # Check required fields
    # Payroll: employee_id required
    if 'employee_id' not in payroll_mapped:
        report['missing_fields']['payroll'].append('employee_id')
    
    # Recordkeeper: employee_id required
    if 'employee_id' not in rk_mapped:
        report['missing_fields']['recordkeeper'].append('employee_id')
    
    # Check reconciliation requirements: at least one amount field on both sides
    payroll_amount_fields = ['EE Deferral $', 'EE Roth $', 'loan_amount']
    rk_amount_fields = ['EE Deferral $', 'EE Roth $', 'loan_amount']
    
    payroll_has_amount = any(field in payroll_mapped for field in payroll_amount_fields)
    rk_has_amount = any(field in rk_mapped for field in rk_amount_fields)
    
    if not payroll_has_amount:
        report['missing_fields']['payroll'].append('at least one amount field (ee_deferral, ee_roth, or loan_amount)')
    if not rk_has_amount:
        report['missing_fields']['recordkeeper'].append('at least one amount field (ee_deferral, ee_roth, or loan_amount)')
    
    # Check timing requirements: pay_date + deposit_date
    if 'pay_date' not in payroll_mapped:
        report['warnings'].append('pay_date not found in payroll - timing analysis will be limited')
    if 'deposit_date' not in rk_mapped:
        report['warnings'].append('deposit_date not found in recordkeeper - timing analysis will be limited')
    
    # Compute join-key coverage
    if 'employee_id' in payroll_mapped:
        payroll_emp_id_col = payroll_mapped['employee_id']
        report['join_key_coverage']['payroll'] = compute_join_key_coverage(
            payroll_csv_path, payroll_emp_id_col
        )
        if report['join_key_coverage']['payroll'] < 100.0:
            report['warnings'].append(
                f"Payroll employee_id coverage: {report['join_key_coverage']['payroll']:.1f}% "
                f"({100.0 - report['join_key_coverage']['payroll']:.1f}% rows have empty employee_id)"
            )
    else:
        report['warnings'].append('Cannot compute payroll join-key coverage: employee_id not mapped')
    
    if 'employee_id' in rk_mapped:
        rk_emp_id_col = rk_mapped['employee_id']
        report['join_key_coverage']['recordkeeper'] = compute_join_key_coverage(
            recordkeeper_csv_path, rk_emp_id_col
        )
        if report['join_key_coverage']['recordkeeper'] < 100.0:
            report['warnings'].append(
                f"Recordkeeper employee_id coverage: {report['join_key_coverage']['recordkeeper']:.1f}% "
                f"({100.0 - report['join_key_coverage']['recordkeeper']:.1f}% rows have empty employee_id)"
            )
    else:
        report['warnings'].append('Cannot compute recordkeeper join-key coverage: employee_id not mapped')
    
    # Determine if safe to run
    # Core requirements:
    # 1. employee_id must be present on both sides
    # 2. At least one amount field must be present on both sides
    safe = (
        'employee_id' in payroll_mapped and
        'employee_id' in rk_mapped and
        payroll_has_amount and
        rk_has_amount
    )
    
    return (safe, report)


def print_preflight_report(report: Dict[str, Any]) -> None:
    """Print a human-readable preflight report."""
    print("\n=== Preflight Report ===\n")
    
    # Missing fields
    if report['missing_fields']['payroll'] or report['missing_fields']['recordkeeper']:
        print("MISSING REQUIRED FIELDS:")
        if report['missing_fields']['payroll']:
            print(f"  Payroll: {', '.join(report['missing_fields']['payroll'])}")
        if report['missing_fields']['recordkeeper']:
            print(f"  Recordkeeper: {', '.join(report['missing_fields']['recordkeeper'])}")
        print()
    else:
        print("✓ All required fields present\n")
    
    # Mapped fields
    print("MAPPED FIELDS:")
    print("  Payroll:")
    for canonical, source in report['mapped_fields']['payroll'].items():
        print(f"    {canonical} <- {source}")
    print("  Recordkeeper:")
    for canonical, source in report['mapped_fields']['recordkeeper'].items():
        print(f"    {canonical} <- {source}")
    print()
    
    # Join key coverage
    print("JOIN-KEY COVERAGE:")
    print(f"  Payroll employee_id: {report['join_key_coverage']['payroll']:.1f}%")
    print(f"  Recordkeeper employee_id: {report['join_key_coverage']['recordkeeper']:.1f}%")
    print()
    
    # Warnings
    if report['warnings']:
        print("WARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning}")
        print()
    else:
        print("✓ No warnings\n")


if __name__ == "__main__":
    """
    Manual test function for preflight checks.
    """
    import sys
    import os
    
    # Handle CLI arguments or use defaults
    if len(sys.argv) == 1:
        # No arguments: use environment variables or defaults
        payroll_path = os.getenv("PAYROLL_CSV_PATH")
        if not payroll_path:
            print("Error: PAYROLL_CSV_PATH environment variable not set.")
            print("Either set PAYROLL_CSV_PATH or provide payroll CSV path as first argument.")
            sys.exit(2)
        
        rk_path = os.getenv("RECORDKEEPER_CSV_PATH")
        if not rk_path:
            print("Error: RECORDKEEPER_CSV_PATH environment variable not set.")
            print("Either set RECORDKEEPER_CSV_PATH or provide recordkeeper CSV path as second argument.")
            sys.exit(2)
        
        mapping_path = os.getenv("MAPPING_YAML_PATH")
        if not mapping_path:
            # Default to mapping_example.yaml in repo root (same directory as this script)
            mapping_path = str(Path(__file__).parent / "mapping_example.yaml")
    
    elif len(sys.argv) == 4:
        # Three arguments provided: use them directly
        payroll_path = sys.argv[1]
        rk_path = sys.argv[2]
        mapping_path = sys.argv[3]
    
    else:
        # Invalid argument count
        print("Usage: python preflight.py <payroll_csv> <recordkeeper_csv> <mapping_yaml>")
        print("\nOr run with no arguments to use environment variables:")
        print("  PAYROLL_CSV_PATH - path to payroll CSV file")
        print("  RECORDKEEPER_CSV_PATH - path to recordkeeper CSV file")
        print("  MAPPING_YAML_PATH - path to mapping YAML file (optional, defaults to mapping_example.yaml)")
        sys.exit(1)
    
    # Print resolved paths
    print("=== Preflight Configuration ===")
    print(f"Payroll CSV: {payroll_path}")
    print(f"Recordkeeper CSV: {rk_path}")
    print(f"Mapping YAML: {mapping_path}")
    print()
    
    safe, report = run_preflight(payroll_path, rk_path, mapping_path)
    
    print_preflight_report(report)
    
    print(f"\nSAFE TO RUN: {'YES' if safe else 'NO'}\n")
    
    if not safe:
        sys.exit(1)
