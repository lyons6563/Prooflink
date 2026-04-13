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


def compute_join_key_coverage(csv_path: str, employee_id_col: str) -> Tuple[float, int]:
    """
    Compute percentage of rows where employee_id is non-empty.
    Reads the employee_id column efficiently.
    
    Returns:
        Tuple of (coverage_percentage: float, row_count: int)
        If file is empty, returns (0.0, 0)
    """
    try:
        # Read only the employee_id column for efficiency
        df = pd.read_csv(csv_path, usecols=[employee_id_col], dtype=str)
        row_count = len(df)
        
        if row_count == 0:
            return (0.0, 0)
        
        # Count non-empty values (not NaN, not empty string after strip)
        non_empty = df[employee_id_col].astype(str).str.strip().ne('').sum()
        coverage = (non_empty / row_count) * 100.0
        return (coverage, row_count)
    except Exception as e:
        # If we can't read, return 0 (will be reported as warning)
        return (0.0, 0)


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
        'join_key_row_count': {'payroll': 0, 'recordkeeper': 0},
        'join_key_empty_file': {'payroll': False, 'recordkeeper': False},
        'join_key_not_mapped': {'payroll': False, 'recordkeeper': False},
        'warnings': []
    }
    
    # Check file existence first
    payroll_path_obj = Path(payroll_csv_path)
    rk_path_obj = Path(recordkeeper_csv_path)
    
    blocked_files = []
    if not payroll_path_obj.exists():
        blocked_files.append(('payroll', payroll_csv_path))
    if not rk_path_obj.exists():
        blocked_files.append(('recordkeeper', recordkeeper_csv_path))
    
    if blocked_files:
        report['blocked_files'] = blocked_files
        return (False, report)
    
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
    # Check payroll join key
    if 'employee_id' not in payroll_mapped:
        report['join_key_not_mapped']['payroll'] = True
    else:
        payroll_emp_id_col = payroll_mapped['employee_id']
        coverage, row_count = compute_join_key_coverage(
            payroll_csv_path, payroll_emp_id_col
        )
        report['join_key_coverage']['payroll'] = coverage
        report['join_key_row_count']['payroll'] = row_count
        
        if row_count == 0:
            report['join_key_empty_file']['payroll'] = True
        elif coverage < 100.0:
            report['warnings'].append(
                f"Payroll employee_id coverage: {coverage:.1f}% "
                f"({100.0 - coverage:.1f}% rows have empty employee_id)"
            )
    
    # Check recordkeeper join key
    if 'employee_id' not in rk_mapped:
        report['join_key_not_mapped']['recordkeeper'] = True
    else:
        rk_emp_id_col = rk_mapped['employee_id']
        coverage, row_count = compute_join_key_coverage(
            recordkeeper_csv_path, rk_emp_id_col
        )
        report['join_key_coverage']['recordkeeper'] = coverage
        report['join_key_row_count']['recordkeeper'] = row_count
        
        if row_count == 0:
            report['join_key_empty_file']['recordkeeper'] = True
        elif coverage < 100.0:
            report['warnings'].append(
                f"Recordkeeper employee_id coverage: {coverage:.1f}% "
                f"({100.0 - coverage:.1f}% rows have empty employee_id)"
            )
    
    # Block if join key issues are detected
    if (report['join_key_not_mapped']['payroll'] or 
        report['join_key_not_mapped']['recordkeeper'] or
        report['join_key_empty_file']['payroll'] or 
        report['join_key_empty_file']['recordkeeper']):
        # Will be handled in print function and safe determination
        pass
    
    # Determine if safe to run
    # Core requirements:
    # 1. employee_id must be present on both sides
    # 2. At least one amount field must be present on both sides
    # 3. Join key must be mapped on both sides
    # 4. CSV files must have at least one row
    safe = (
        'employee_id' in payroll_mapped and
        'employee_id' in rk_mapped and
        payroll_has_amount and
        rk_has_amount and
        not report['join_key_not_mapped']['payroll'] and
        not report['join_key_not_mapped']['recordkeeper'] and
        not report['join_key_empty_file']['payroll'] and
        not report['join_key_empty_file']['recordkeeper']
    )
    
    return (safe, report)


def print_preflight_report(report: Dict[str, Any]) -> None:
    """Print a human-readable preflight report."""
    print("\n=== Preflight Report ===\n")
    
    # Check for blocked files first
    if 'blocked_files' in report and report['blocked_files']:
        print("BLOCKED:")
        for file_type, file_path in report['blocked_files']:
            if file_type == 'payroll':
                print(f"- Payroll file not found: {file_path}")
            elif file_type == 'recordkeeper':
                print(f"- Recordkeeper file not found: {file_path}")
        print()
        return
    
    # Check for missing mapped headers
    if 'missing_mapped_headers' in report and report['missing_mapped_headers']:
        print("BLOCKED:")
        missing_payroll = report['missing_mapped_headers'].get('payroll', [])
        missing_rk = report['missing_mapped_headers'].get('recordkeeper', [])
        if missing_payroll:
            print(f"- Missing payroll headers referenced by mapping: {', '.join(missing_payroll)}")
        if missing_rk:
            print(f"- Missing recordkeeper headers referenced by mapping: {', '.join(missing_rk)}")
        print()
        return
    
    # Check for join key not mapped (block immediately, don't show coverage)
    join_key_not_mapped = False
    if report.get('join_key_not_mapped', {}).get('payroll', False):
        print("BLOCKED:")
        print("- Payroll join key (employee_id) is not mapped or not present in headers")
        join_key_not_mapped = True
    if report.get('join_key_not_mapped', {}).get('recordkeeper', False):
        if not join_key_not_mapped:
            print("BLOCKED:")
        print("- Recordkeeper join key (employee_id) is not mapped or not present in headers")
        join_key_not_mapped = True
    
    if join_key_not_mapped:
        print()
        return
    
    # Check for empty files (will show in coverage section, but still block)
    join_key_empty = False
    if report.get('join_key_empty_file', {}).get('payroll', False):
        join_key_empty = True
    if report.get('join_key_empty_file', {}).get('recordkeeper', False):
        join_key_empty = True
    
    if join_key_empty:
        print("BLOCKED:")
        if report.get('join_key_empty_file', {}).get('payroll', False):
            print("- Payroll file has 0 rows (cannot compute join-key coverage)")
        if report.get('join_key_empty_file', {}).get('recordkeeper', False):
            print("- Recordkeeper file has 0 rows (cannot compute join-key coverage)")
        print()
    
    # Missing fields
    if report['missing_fields']['payroll'] or report['missing_fields']['recordkeeper']:
        print("MISSING REQUIRED FIELDS:")
        if report['missing_fields']['payroll']:
            print(f"  Payroll: {', '.join(report['missing_fields']['payroll'])}")
        if report['missing_fields']['recordkeeper']:
            print(f"  Recordkeeper: {', '.join(report['missing_fields']['recordkeeper'])}")
        print()
    else:
        print("[OK] All required fields present\n")
    
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
    
    # Payroll coverage
    if report.get('join_key_empty_file', {}).get('payroll', False):
        print("  Payroll employee_id: 0 rows (cannot compute)")
    elif report.get('join_key_not_mapped', {}).get('payroll', False):
        print("  Payroll employee_id: not mapped")
    else:
        print(f"  Payroll employee_id: {report['join_key_coverage']['payroll']:.1f}%")
    
    # Recordkeeper coverage
    if report.get('join_key_empty_file', {}).get('recordkeeper', False):
        print("  Recordkeeper employee_id: 0 rows (cannot compute)")
    elif report.get('join_key_not_mapped', {}).get('recordkeeper', False):
        print("  Recordkeeper employee_id: not mapped")
    else:
        print(f"  Recordkeeper employee_id: {report['join_key_coverage']['recordkeeper']:.1f}%")
    print()
    
    # Warnings
    if report['warnings']:
        print("WARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning}")
        print()
    else:
        print("[OK] No warnings\n")


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

