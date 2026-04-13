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

UNKNOWN_VENDOR = "Unknown / Generic"


@dataclass
class VendorDetectionResult:
    """Result of vendor detection for both payroll and recordkeeper."""

    payroll_vendor: str
    payroll_confidence: float
    rk_vendor: str
    rk_confidence: float


def _detect_single_vendor(
    df: Optional[pd.DataFrame],
    signatures: dict,
    vendor_hint: Optional[str],
) -> tuple[str, float]:
    """Detect a single vendor safely and return normalized defaults on failures."""
    if df is None or df.empty:
        # Honor explicit hints even when files are empty, with zero confidence.
        return (vendor_hint or UNKNOWN_VENDOR), 0.0

    vendor, confidence = detect_vendor_with_confidence(df, signatures, vendor_hint)

    # Preserve hint override behavior while still returning the computed confidence.
    if vendor_hint:
        vendor = vendor_hint

    return vendor or UNKNOWN_VENDOR, confidence


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
        - vendor = "Unknown / Generic"
        - confidence = 0.0
    """
    # Backwards compatible signature; kwargs intentionally ignored.
    _ = kwargs

    payroll_vendor, payroll_confidence = _detect_single_vendor(
        df=payroll_df,
        signatures=PAYROLL_VENDOR_SIGNATURES,
        vendor_hint=payroll_vendor_hint,
    )
    rk_vendor, rk_confidence = _detect_single_vendor(
        df=rk_df,
        signatures=RK_VENDOR_SIGNATURES,
        vendor_hint=rk_vendor_hint,
    )

    return VendorDetectionResult(
        payroll_vendor=payroll_vendor,
        payroll_confidence=payroll_confidence,
        rk_vendor=rk_vendor,
        rk_confidence=rk_confidence,
    )
