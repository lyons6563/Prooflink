"""
Vendor detection module for ProofLink.

This module provides a unified vendor detection interface that serves as the
single source of truth for vendor detection across the system.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from vendors import (
    PAYROLL_VENDOR_SIGNATURES,
    RK_VENDOR_SIGNATURES,
    detect_vendor_with_confidence,
)


@dataclass
class VendorDetectionResult:
    """Result of vendor detection for both payroll and recordkeeper."""
    payroll_vendor: str
    payroll_confidence: float
    rk_vendor: str
    rk_confidence: float


def detect_vendors(
    payroll_df: pd.DataFrame,
    rk_df: pd.DataFrame,
    payroll_vendor_hint: Optional[str] = None,
    rk_vendor_hint: Optional[str] = None,
    **kwargs,
) -> VendorDetectionResult:
    """
    Run vendor detection for both payroll and recordkeeper data.

    This function reuses the existing logic that currently prints:
    'Detected payroll vendor: ...' and 'Detected recordkeeper: ...'.

    This function MUST:
    - Return the same detected vendor names.
    - Return the same confidence scores.
    - Not print directly; instead, return the result.

    Args:
        payroll_df: Payroll dataframe (should be normalized)
        rk_df: Recordkeeper dataframe (should be normalized)
        payroll_vendor_hint: Optional hint to override payroll vendor detection
        rk_vendor_hint: Optional hint to override recordkeeper vendor detection

    Returns:
        VendorDetectionResult with detected vendors and confidence scores.

    Defaults:
        If detection fails or inputs are empty:
        - payroll_vendor = "Unknown / Generic"
        - rk_vendor = "Unknown / Generic"
        - confidences = 0.0
    """
    # Handle empty dataframes
    if payroll_df is None or payroll_df.empty:
        return VendorDetectionResult(
            payroll_vendor="Unknown / Generic",
            payroll_confidence=0.0,
            rk_vendor="Unknown / Generic",
            rk_confidence=0.0,
        )
    
    if rk_df is None or rk_df.empty:
        return VendorDetectionResult(
            payroll_vendor="Unknown / Generic",
            payroll_confidence=0.0,
            rk_vendor="Unknown / Generic",
            rk_confidence=0.0,
        )

    # Run detection with confidence scoring
    payroll_vendor, payroll_confidence = detect_vendor_with_confidence(
        payroll_df, PAYROLL_VENDOR_SIGNATURES, payroll_vendor_hint
    )
    rk_vendor, rk_confidence = detect_vendor_with_confidence(
        rk_df, RK_VENDOR_SIGNATURES, rk_vendor_hint
    )
    
    # If hint provided, use it (confidence still calculated)
    if payroll_vendor_hint:
        payroll_vendor = payroll_vendor_hint
    if rk_vendor_hint:
        rk_vendor = rk_vendor_hint

    # Apply defaults for None values (same as print statements)
    final_payroll_vendor = payroll_vendor or "Unknown / Generic"
    final_rk_vendor = rk_vendor or "Unknown / Generic"

    return VendorDetectionResult(
        payroll_vendor=final_payroll_vendor,
        payroll_confidence=payroll_confidence,
        rk_vendor=final_rk_vendor,
        rk_confidence=rk_confidence,
    )

