#!/usr/bin/env python3
"""
SECURE 2.0 Preflight Checker - CLI Entry Point

Performs Phase 1 compliance checks on payroll data to identify potential
SECURE 2.0 Act violations related to HCE catch-up contribution rules.
"""

import argparse
import csv
import sys
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional
import yaml


def load_config(config_path: Path) -> Dict:
    """
    Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        SystemExit(2): If config file cannot be read or is invalid
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in config file: {e}", file=sys.stderr)
        sys.exit(2)
    except IOError as e:
        print(f"Error: Cannot read config file: {e}", file=sys.stderr)
        sys.exit(2)
    
    if not isinstance(config, dict):
        print("Error: Config file must contain a YAML dictionary", file=sys.stderr)
        sys.exit(2)
    
    # Validate required keys
    required_keys = ['hce_threshold', 'catch_up', 'annualization']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required config key: {key}", file=sys.stderr)
            sys.exit(2)
    
    # Validate hce_threshold
    hce = config['hce_threshold']
    if 'current_year' not in hce or 'compensation_limit' not in hce:
        print("Error: hce_threshold must contain current_year and compensation_limit", file=sys.stderr)
        sys.exit(2)
    
    try:
        current_year = int(hce['current_year'])
        compensation_limit = float(hce['compensation_limit'])
        if current_year < 2000 or current_year > 2100:
            print("Error: current_year must be a valid year", file=sys.stderr)
            sys.exit(2)
        if compensation_limit <= 0:
            print("Error: compensation_limit must be positive", file=sys.stderr)
            sys.exit(2)
    except (ValueError, TypeError):
        print("Error: Invalid hce_threshold values (must be numeric)", file=sys.stderr)
        sys.exit(2)
    
    # Validate catch_up
    catch_up = config['catch_up']
    if 'roth_only_risk_year' not in catch_up:
        print("Error: catch_up must contain roth_only_risk_year", file=sys.stderr)
        sys.exit(2)
    
    try:
        roth_only_risk_year = int(catch_up['roth_only_risk_year'])
        if roth_only_risk_year < 2000 or roth_only_risk_year > 2100:
            print("Error: roth_only_risk_year must be a valid year", file=sys.stderr)
            sys.exit(2)
    except (ValueError, TypeError):
        print("Error: Invalid roth_only_risk_year (must be numeric)", file=sys.stderr)
        sys.exit(2)
    
    # Validate annualization
    annualization = config['annualization']
    if 'method' not in annualization:
        print("Error: annualization must contain method", file=sys.stderr)
        sys.exit(2)
    
    valid_methods = ['gross', 'ytd', 'gross_or_ytd']
    if annualization['method'] not in valid_methods:
        print(f"Error: annualization.method must be one of: {', '.join(valid_methods)}", file=sys.stderr)
        sys.exit(2)
    
    return config


def load_payroll_data(payroll_path: Path) -> List[Dict]:
    """
    Load and validate payroll CSV file.
    
    Args:
        payroll_path: Path to payroll CSV file
        
    Returns:
        List of dictionaries, each representing a payroll record
        
    Raises:
        SystemExit(2): If CSV file cannot be read or is invalid
    """
    required_columns = ['employee_id', 'employee_name', 'gross_pay', 
                       'ytd_gross_pay', 'pay_period_start', 'pay_period_end']
    
    try:
        with open(payroll_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check for required columns
            if not reader.fieldnames:
                print("Error: CSV file is empty or has no header row", file=sys.stderr)
                sys.exit(2)
            
            missing_columns = [col for col in required_columns if col not in reader.fieldnames]
            if missing_columns:
                print(f"Error: Missing required CSV columns: {', '.join(missing_columns)}", file=sys.stderr)
                sys.exit(2)
            
            records = []
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    # Validate and convert required fields
                    record = {
                        'employee_id': str(row['employee_id']).strip(),
                        'employee_name': str(row['employee_name']).strip(),
                        'gross_pay': _parse_decimal(row['gross_pay'], 'gross_pay', row_num),
                        'ytd_gross_pay': _parse_decimal(row['ytd_gross_pay'], 'ytd_gross_pay', row_num),
                        'pay_period_start': _parse_date(row['pay_period_start'], 'pay_period_start', row_num),
                        'pay_period_end': _parse_date(row['pay_period_end'], 'pay_period_end', row_num),
                    }
                    
                    # Validate optional fields
                    catch_up_contribution = row.get('catch_up_contribution', '').strip()
                    if catch_up_contribution:
                        record['catch_up_contribution'] = _parse_decimal(
                            catch_up_contribution, 'catch_up_contribution', row_num
                        )
                    else:
                        record['catch_up_contribution'] = Decimal('0')
                    
                    catch_up_type = row.get('catch_up_type', '').strip()
                    if catch_up_type:
                        if catch_up_type not in ['Roth', 'Traditional']:
                            print(f"Error: Row {row_num}: catch_up_type must be 'Roth' or 'Traditional'", file=sys.stderr)
                            sys.exit(2)
                        record['catch_up_type'] = catch_up_type
                    else:
                        record['catch_up_type'] = None
                    
                    # Validate dates are in order
                    if record['pay_period_start'] > record['pay_period_end']:
                        print(f"Error: Row {row_num}: pay_period_start must be <= pay_period_end", file=sys.stderr)
                        sys.exit(2)
                    
                    # Validate non-negative amounts
                    if record['gross_pay'] < 0:
                        print(f"Error: Row {row_num}: gross_pay must be non-negative", file=sys.stderr)
                        sys.exit(2)
                    if record['ytd_gross_pay'] < 0:
                        print(f"Error: Row {row_num}: ytd_gross_pay must be non-negative", file=sys.stderr)
                        sys.exit(2)
                    if record['catch_up_contribution'] < 0:
                        print(f"Error: Row {row_num}: catch_up_contribution must be non-negative", file=sys.stderr)
                        sys.exit(2)
                    
                    records.append(record)
                    
                except (ValueError, KeyError) as e:
                    print(f"Error: Row {row_num}: {e}", file=sys.stderr)
                    sys.exit(2)
            
            if not records:
                print("Error: CSV file contains no data rows", file=sys.stderr)
                sys.exit(2)
            
            return records
            
    except IOError as e:
        print(f"Error: Cannot read payroll file: {e}", file=sys.stderr)
        sys.exit(2)


def _parse_decimal(value: str, field_name: str, row_num: int) -> Decimal:
    """Parse a decimal value from CSV, raising clear errors on failure."""
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field_name} must be a valid number")


def _parse_date(value: str, field_name: str, row_num: int) -> date:
    """Parse a date value from CSV (YYYY-MM-DD format), raising clear errors on failure."""
    try:
        return datetime.strptime(value.strip(), '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"{field_name} must be in YYYY-MM-DD format")


def check_roth_only_catchup_hce(payroll_data: List[Dict], config: Dict) -> List[Dict]:
    """
    Check 1: Identify HCEs making Roth-only catch-up contributions.
    
    Rule: For the year specified in catch_up.roth_only_risk_year and beyond,
    HCEs cannot make catch-up contributions exclusively to Roth accounts.
    
    Args:
        payroll_data: List of payroll records
        config: Configuration dictionary
        
    Returns:
        List of violation dictionaries for exception CSV
    """
    violations = []
    current_year = config['hce_threshold']['current_year']
    roth_only_risk_year = config['catch_up']['roth_only_risk_year']
    
    # Only check if current year is >= roth_only_risk_year
    if current_year < roth_only_risk_year:
        return violations
    
    for record in payroll_data:
        # Filter employees with Roth catch-up contributions
        if (record.get('catch_up_contribution', Decimal('0')) > 0 and 
            record.get('catch_up_type') == 'Roth'):
            
            # Determine if employee is an HCE
            if is_hce(record, config):
                projected_comp = annualize_compensation(record, config)
                
                violation = {
                    'employee_id': record['employee_id'],
                    'employee_name': record['employee_name'],
                    'violation_type': 'ROTH_ONLY_CATCHUP_HCE',
                    'violation_description': (
                        f"HCE making Roth-only catch-up contributions. "
                        f"Projected annual compensation: ${projected_comp:,.2f}"
                    ),
                    'projected_annual_compensation': float(projected_comp),
                    'catch_up_amount': float(record['catch_up_contribution']),
                    'catch_up_type': record['catch_up_type'],
                    'pay_period_start': record['pay_period_start'].strftime('%Y-%m-%d'),
                    'pay_period_end': record['pay_period_end'].strftime('%Y-%m-%d'),
                }
                violations.append(violation)
    
    return violations


def check_potential_hce(payroll_data: List[Dict], config: Dict) -> List[Dict]:
    """
    Check 2: Identify potential HCEs based on projected annual compensation.
    
    Rule: Identify employees who may become HCEs based on annualized compensation.
    
    Args:
        payroll_data: List of payroll records
        config: Configuration dictionary
        
    Returns:
        List of potential HCE records for exception CSV (informational)
    """
    potential_hces = []
    threshold = Decimal(str(config['hce_threshold']['compensation_limit']))
    
    for record in payroll_data:
        # Calculate projected annual compensation
        projected_comp = annualize_compensation(record, config)
        
        # Flag if projected compensation >= threshold
        if projected_comp >= threshold:
            potential_hce = {
                'employee_id': record['employee_id'],
                'employee_name': record['employee_name'],
                'violation_type': 'POTENTIAL_HCE',
                'violation_description': (
                    f"Potential HCE based on projected annual compensation: ${projected_comp:,.2f}"
                ),
                'projected_annual_compensation': float(projected_comp),
                'catch_up_amount': float(record.get('catch_up_contribution', Decimal('0'))),
                'catch_up_type': record.get('catch_up_type') or '',
                'pay_period_start': record['pay_period_start'].strftime('%Y-%m-%d'),
                'pay_period_end': record['pay_period_end'].strftime('%Y-%m-%d'),
            }
            potential_hces.append(potential_hce)
    
    return potential_hces


def annualize_compensation(record: Dict, config: Dict) -> Decimal:
    """
    Calculate projected annual compensation for an employee.
    
    Args:
        record: Payroll record dictionary
        config: Configuration dictionary
        
    Returns:
        Projected annual compensation amount
    """
    method = config['annualization']['method']
    current_year = config['hce_threshold']['current_year']
    
    # Calculate days elapsed in year for YTD projection
    year_start = date(current_year, 1, 1)
    pay_period_end = record['pay_period_end']
    days_elapsed = (pay_period_end - year_start).days + 1  # +1 to include end date
    if days_elapsed <= 0:
        days_elapsed = 1  # Prevent division by zero
    
    # Method: ytd - always use YTD projection if available
    if method == 'ytd':
        if record['ytd_gross_pay'] > 0:
            return record['ytd_gross_pay'] * Decimal('365') / Decimal(str(days_elapsed))
        else:
            # Fallback to gross if YTD is zero/not available
            return _annualize_from_gross(record)
    
    # Method: gross_or_ytd - prefer YTD if available, otherwise use gross
    if method == 'gross_or_ytd':
        if record['ytd_gross_pay'] > 0:
            return record['ytd_gross_pay'] * Decimal('365') / Decimal(str(days_elapsed))
        else:
            return _annualize_from_gross(record)
    
    # Method: gross - always annualize from gross pay
    if method == 'gross':
        return _annualize_from_gross(record)
    
    # Should not reach here, but default to gross_or_ytd behavior
    return _annualize_from_gross(record)


def _annualize_from_gross(record: Dict) -> Decimal:
    """Annualize compensation from gross pay based on pay period frequency."""
    pay_period_start = record['pay_period_start']
    pay_period_end = record['pay_period_end']
    gross_pay = record['gross_pay']
    
    # Calculate pay period length in days
    pay_period_days = (pay_period_end - pay_period_start).days + 1  # +1 to include both dates
    if pay_period_days <= 0:
        pay_period_days = 1  # Prevent division by zero
    
    # Annualize: gross_pay * (365 / pay_period_days)
    return gross_pay * Decimal('365') / Decimal(str(pay_period_days))


def is_hce(record: Dict, config: Dict) -> bool:
    """
    Determine if an employee is a Highly Compensated Employee (HCE).
    
    Args:
        record: Payroll record dictionary
        config: Configuration dictionary
        
    Returns:
        True if employee is an HCE, False otherwise
    """
    projected_compensation = annualize_compensation(record, config)
    threshold = Decimal(str(config['hce_threshold']['compensation_limit']))
    return projected_compensation >= threshold


def write_exception_csv(exceptions: List[Dict], output_path: Path) -> None:
    """
    Write exception records to CSV file.
    
    Args:
        exceptions: List of exception dictionaries
        output_path: Path to output CSV file
    """
    fieldnames = [
        'employee_id',
        'employee_name',
        'violation_type',
        'violation_description',
        'projected_annual_compensation',
        'catch_up_amount',
        'catch_up_type',
        'pay_period_start',
        'pay_period_end'
    ]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(exceptions)
    except IOError as e:
        print(f"Error: Cannot write exception CSV file: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SECURE 2.0 Preflight Checker - Phase 1 compliance checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  secure20-preflight --payroll data/payroll.csv --config config.yaml
  secure20-preflight -p payroll.csv -c config.yaml -o exceptions.csv
        """
    )
    
    parser.add_argument(
        "--payroll", "-p",
        type=str,
        required=True,
        help="Path to payroll CSV file"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="exceptions.csv",
        help="Path to exception CSV output file (default: exceptions.csv)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="SECURE 2.0 Preflight Checker 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Validate input file paths
    payroll_path = Path(args.payroll)
    config_path = Path(args.config)
    output_path = Path(args.output)
    
    if not payroll_path.exists():
        print(f"Error: Payroll file not found: {payroll_path}", file=sys.stderr)
        sys.exit(2)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Load payroll data
        payroll_data = load_payroll_data(payroll_path)
        
        # Perform Phase 1 checks
        violations = []
        
        # Check 1: Roth-only catch-up risk for HCEs (actual violations)
        roth_violations = check_roth_only_catchup_hce(payroll_data, config)
        violations.extend(roth_violations)
        
        # Check 2: Potential HCE detection (informational warnings)
        potential_hces = check_potential_hce(payroll_data, config)
        violations.extend(potential_hces)
        
        # Count actual violations (exclude informational POTENTIAL_HCE)
        actual_violations = [v for v in violations if v['violation_type'] == 'ROTH_ONLY_CATCHUP_HCE']
        violation_count = len(actual_violations)
        
        # Write exception CSV if any exceptions found
        if violations:
            write_exception_csv(violations, output_path)
        
        # Print results
        if violation_count > 0:
            print("NOT SAFE", file=sys.stdout)
            print(f"Violations: {violation_count}", file=sys.stdout)
            
            # Get top 10 employee IDs with violations (prioritize actual violations)
            employee_ids = [v['employee_id'] for v in actual_violations]
            # Add potential HCE IDs if we have fewer than 10
            if len(employee_ids) < 10:
                potential_ids = [v['employee_id'] for v in potential_hces 
                               if v['employee_id'] not in employee_ids]
                employee_ids.extend(potential_ids[:10 - len(employee_ids)])
            
            top_10 = employee_ids[:10]
            if top_10:
                print(f"Top employee IDs: {', '.join(top_10)}", file=sys.stdout)
            
            sys.exit(2)  # NOT SAFE
        else:
            print("SAFE", file=sys.stdout)
            sys.exit(0)  # SAFE
            
    except KeyboardInterrupt:
        print("\nError: Interrupted by user", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

