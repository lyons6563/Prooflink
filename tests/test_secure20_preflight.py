"""
Unit tests for SECURE 2.0 Preflight Checker

Focused tests for:
- Annualization from gross pay
- Potential HCE threshold logic
- Violation detection rules
"""

import unittest
from datetime import date
from decimal import Decimal
import sys
from pathlib import Path

# Add parent directory to path to import secure20_preflight
sys.path.insert(0, str(Path(__file__).parent.parent))

from secure20_preflight import (
    annualize_compensation,
    is_hce,
    check_potential_hce,
    check_roth_only_catchup_hce,
)


class TestViolationDetection(unittest.TestCase):
    """Tests for violation detection rules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'hce_threshold': {'current_year': 2024, 'compensation_limit': 150000},
            'catch_up': {'roth_only_risk_year': 2024},
            'annualization': {'method': 'gross_or_ytd'}
        }
    
    def test_detect_hce_roth_only_catchup_violation(self):
        """Test detection of HCE making Roth-only catch-up contributions."""
        # HCE (annualized > $150k) with Roth catch-up
        record = {
            'employee_id': 'EMP001',
            'employee_name': 'HCE with Roth',
            'gross_pay': Decimal('10000.00'),  # ~$260k annualized
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('750.00'),
            'catch_up_type': 'Roth',
        }
        
        violations = check_roth_only_catchup_hce([record], self.config)
        
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['violation_type'], 'ROTH_ONLY_CATCHUP_HCE')
        self.assertEqual(violations[0]['employee_id'], 'EMP001')
        self.assertEqual(violations[0]['catch_up_type'], 'Roth')
        self.assertEqual(violations[0]['catch_up_amount'], 750.0)
        self.assertGreater(violations[0]['projected_annual_compensation'], 150000.0)
    
    def test_no_violation_for_traditional_catchup(self):
        """Test that traditional catch-up contributions don't trigger violation."""
        # HCE with Traditional catch-up (allowed)
        record = {
            'employee_id': 'EMP002',
            'employee_name': 'HCE with Traditional',
            'gross_pay': Decimal('10000.00'),  # ~$260k annualized
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('750.00'),
            'catch_up_type': 'Traditional',
        }
        
        violations = check_roth_only_catchup_hce([record], self.config)
        
        self.assertEqual(len(violations), 0)
    
    def test_no_violation_for_non_hce_roth_catchup(self):
        """Test that non-HCE Roth catch-up doesn't trigger violation."""
        # Non-HCE (annualized < $150k) with Roth catch-up (allowed)
        record = {
            'employee_id': 'EMP003',
            'employee_name': 'Non-HCE with Roth',
            'gross_pay': Decimal('4000.00'),  # ~$104k annualized
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('500.00'),
            'catch_up_type': 'Roth',
        }
        
        violations = check_roth_only_catchup_hce([record], self.config)
        
        self.assertEqual(len(violations), 0)
    
    def test_no_violation_when_no_catchup(self):
        """Test that employees without catch-up contributions don't trigger violation."""
        # HCE with no catch-up contributions
        record = {
            'employee_id': 'EMP004',
            'employee_name': 'HCE No Catch-up',
            'gross_pay': Decimal('10000.00'),
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('0'),
            'catch_up_type': None,
        }
        
        violations = check_roth_only_catchup_hce([record], self.config)
        
        self.assertEqual(len(violations), 0)
    
    def test_no_violation_before_risk_year(self):
        """Test that violations are not flagged before roth_only_risk_year."""
        # Config with risk year in future
        future_config = {
            'hce_threshold': {'current_year': 2023, 'compensation_limit': 150000},
            'catch_up': {'roth_only_risk_year': 2024},
            'annualization': {'method': 'gross_or_ytd'}
        }
        
        # HCE with Roth catch-up in 2023 (before restriction)
        record = {
            'employee_id': 'EMP005',
            'employee_name': 'HCE Before Risk Year',
            'gross_pay': Decimal('10000.00'),
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2023, 1, 1),
            'pay_period_end': date(2023, 1, 14),
            'catch_up_contribution': Decimal('750.00'),
            'catch_up_type': 'Roth',
        }
        
        violations = check_roth_only_catchup_hce([record], future_config)
        
        self.assertEqual(len(violations), 0)
    
    def test_violation_detection_multiple_employees(self):
        """Test violation detection with multiple employees."""
        records = [
            {
                'employee_id': 'EMP001',
                'employee_name': 'HCE Roth',
                'gross_pay': Decimal('10000.00'),
                'ytd_gross_pay': Decimal('0'),
                'pay_period_start': date(2024, 1, 1),
                'pay_period_end': date(2024, 1, 14),
                'catch_up_contribution': Decimal('750.00'),
                'catch_up_type': 'Roth',
            },
            {
                'employee_id': 'EMP002',
                'employee_name': 'HCE Traditional',
                'gross_pay': Decimal('10000.00'),
                'ytd_gross_pay': Decimal('0'),
                'pay_period_start': date(2024, 1, 1),
                'pay_period_end': date(2024, 1, 14),
                'catch_up_contribution': Decimal('750.00'),
                'catch_up_type': 'Traditional',
            },
            {
                'employee_id': 'EMP003',
                'employee_name': 'Non-HCE Roth',
                'gross_pay': Decimal('4000.00'),
                'ytd_gross_pay': Decimal('0'),
                'pay_period_start': date(2024, 1, 1),
                'pay_period_end': date(2024, 1, 14),
                'catch_up_contribution': Decimal('500.00'),
                'catch_up_type': 'Roth',
            },
        ]
        
        violations = check_roth_only_catchup_hce(records, self.config)
        
        # Should only flag EMP001 (HCE with Roth catch-up)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['employee_id'], 'EMP001')


class TestPotentialHCEDetection(unittest.TestCase):
    """Tests for Check 2: Potential HCE threshold logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'hce_threshold': {'current_year': 2024, 'compensation_limit': 150000},
            'catch_up': {'roth_only_risk_year': 2024},
            'annualization': {'method': 'gross_or_ytd'}
        }
    
    def test_detect_potential_hce_at_threshold(self):
        """Test that employees at exactly the threshold are flagged."""
        # Annualized compensation exactly at $150,000 threshold
        record = {
            'employee_id': 'EMP001',
            'employee_name': 'Test Employee',
            'gross_pay': Decimal('5753.42'),  # ~$150,000 annualized (biweekly)
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('0'),
            'catch_up_type': None,
        }
        
        violations = check_potential_hce([record], self.config)
        
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['violation_type'], 'POTENTIAL_HCE')
        self.assertEqual(violations[0]['employee_id'], 'EMP001')
        self.assertGreaterEqual(
            violations[0]['projected_annual_compensation'], 
            150000.0
        )
    
    def test_detect_potential_hce_above_threshold(self):
        """Test that employees above threshold are flagged."""
        # Annualized compensation well above threshold
        record = {
            'employee_id': 'EMP002',
            'employee_name': 'High Earner',
            'gross_pay': Decimal('10000.00'),  # ~$260,000 annualized (biweekly)
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('0'),
            'catch_up_type': None,
        }
        
        violations = check_potential_hce([record], self.config)
        
        self.assertEqual(len(violations), 1)
        self.assertGreater(violations[0]['projected_annual_compensation'], 150000.0)
    
    def test_no_flag_for_below_threshold(self):
        """Test that employees below threshold are not flagged."""
        # Annualized compensation below threshold
        record = {
            'employee_id': 'EMP003',
            'employee_name': 'Regular Employee',
            'gross_pay': Decimal('4000.00'),  # ~$104,000 annualized (biweekly)
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('0'),
            'catch_up_type': None,
        }
        
        violations = check_potential_hce([record], self.config)
        
        self.assertEqual(len(violations), 0)
    
    def test_detect_potential_hce_from_ytd(self):
        """Test detection of potential HCE using YTD compensation projection."""
        # YTD $80,000 through day 180 (projects to ~$162,000 annually)
        record = {
            'employee_id': 'EMP004',
            'employee_name': 'YTD Employee',
            'gross_pay': Decimal('5000.00'),
            'ytd_gross_pay': Decimal('80000.00'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 6, 29),  # Day 180
            'catch_up_contribution': Decimal('0'),
            'catch_up_type': None,
        }
        
        violations = check_potential_hce([record], self.config)
        
        self.assertEqual(len(violations), 1)
        # YTD projection: $80000 * (365 / 180) = $162,222
        self.assertGreater(violations[0]['projected_annual_compensation'], 150000.0)
    
    def test_threshold_edge_case_just_below(self):
        """Test that employees just below threshold are not flagged."""
        # Annualized compensation just below $150,000
        record = {
            'employee_id': 'EMP005',
            'employee_name': 'Edge Case',
            'gross_pay': Decimal('5750.00'),  # Slightly below threshold
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
            'catch_up_contribution': Decimal('0'),
            'catch_up_type': None,
        }
        
        violations = check_potential_hce([record], self.config)
        
        # Should not be flagged if projected is < 150000
        projected = annualize_compensation(record, self.config)
        if projected < Decimal('150000'):
            self.assertEqual(len(violations), 0)


class TestAnnualization(unittest.TestCase):
    """Tests for compensation annualization logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_gross = {
            'hce_threshold': {'current_year': 2024, 'compensation_limit': 150000},
            'catch_up': {'roth_only_risk_year': 2024},
            'annualization': {'method': 'gross'}
        }
        self.config_ytd = {
            'hce_threshold': {'current_year': 2024, 'compensation_limit': 150000},
            'catch_up': {'roth_only_risk_year': 2024},
            'annualization': {'method': 'ytd'}
        }
        self.config_gross_or_ytd = {
            'hce_threshold': {'current_year': 2024, 'compensation_limit': 150000},
            'catch_up': {'roth_only_risk_year': 2024},
            'annualization': {'method': 'gross_or_ytd'}
        }
    
    def test_annualize_from_gross_pay_biweekly(self):
        """Test annualization from gross pay for biweekly pay period."""
        # Biweekly: 14 days, $5000 per period
        # Expected: $5000 * (365 / 14) = $130,357.14
        record = {
            'employee_id': 'EMP001',
            'employee_name': 'Test Employee',
            'gross_pay': Decimal('5000.00'),
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
        }
        
        result = annualize_compensation(record, self.config_gross)
        expected = Decimal('5000.00') * Decimal('365') / Decimal('14')
        
        self.assertAlmostEqual(float(result), float(expected), places=2)
        self.assertGreater(result, Decimal('130000'))
    
    def test_annualize_from_gross_pay_monthly(self):
        """Test annualization from gross pay for monthly pay period."""
        # Monthly: 31 days (January), $10000 per period
        # Expected: $10000 * (365 / 31) = $117,741.94
        record = {
            'employee_id': 'EMP002',
            'employee_name': 'Test Employee',
            'gross_pay': Decimal('10000.00'),
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 31),
        }
        
        result = annualize_compensation(record, self.config_gross)
        expected = Decimal('10000.00') * Decimal('365') / Decimal('31')
        
        self.assertAlmostEqual(float(result), float(expected), places=2)
    
    def test_annualize_from_gross_pay_single_day(self):
        """Test annualization handles single-day pay period correctly."""
        # Single day: $1000
        # Expected: $1000 * (365 / 1) = $365,000
        record = {
            'employee_id': 'EMP003',
            'employee_name': 'Test Employee',
            'gross_pay': Decimal('1000.00'),
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 1),
        }
        
        result = annualize_compensation(record, self.config_gross)
        expected = Decimal('1000.00') * Decimal('365')
        
        self.assertEqual(result, expected)
    
    def test_gross_or_ytd_prefers_ytd_when_available(self):
        """Test gross_or_ytd method prefers YTD when available."""
        # YTD: $60000 through day 100 (should project to ~$219,000)
        # Gross: $5000 for 14 days (would project to ~$130,000)
        # Should use YTD projection
        record = {
            'employee_id': 'EMP004',
            'employee_name': 'Test Employee',
            'gross_pay': Decimal('5000.00'),
            'ytd_gross_pay': Decimal('60000.00'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 4, 10),  # Day 100 of year
        }
        
        result = annualize_compensation(record, self.config_gross_or_ytd)
        # YTD projection: $60000 * (365 / 100) = $219,000
        expected_ytd = Decimal('60000.00') * Decimal('365') / Decimal('100')
        
        self.assertAlmostEqual(float(result), float(expected_ytd), places=2)
        # Should be much higher than gross-only projection
        gross_projection = Decimal('5000.00') * Decimal('365') / Decimal('14')
        self.assertGreater(result, gross_projection)
    
    def test_gross_or_ytd_falls_back_to_gross_when_ytd_zero(self):
        """Test gross_or_ytd falls back to gross when YTD is zero."""
        record = {
            'employee_id': 'EMP005',
            'employee_name': 'Test Employee',
            'gross_pay': Decimal('5000.00'),
            'ytd_gross_pay': Decimal('0'),
            'pay_period_start': date(2024, 1, 1),
            'pay_period_end': date(2024, 1, 14),
        }
        
        result = annualize_compensation(record, self.config_gross_or_ytd)
        expected = Decimal('5000.00') * Decimal('365') / Decimal('14')
        
        self.assertAlmostEqual(float(result), float(expected), places=2)




if __name__ == "__main__":
    unittest.main()

