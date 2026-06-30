# issue_taxonomy.py

"""
Central taxonomy for ProofLink engine issue types.

This module serves as the single source of truth for:
- Issue categorization (Secure 2.0, Eligibility, Comp/402(g), Match, Timing, etc.)
- Severity levels (Low, Medium, High)
- Standardized correction hints

Analyzers should call get_issue_metadata(issue_type) when building their CSV rows
or exception records to ensure consistent categorization and messaging across
all analysis outputs.
"""

from __future__ import annotations

from typing import Dict, TypedDict


class IssueMetadata(TypedDict):
    """Metadata structure for issue types."""
    issue_category: str
    severity: str
    correction_hint: str


# Central mapping of issue_type strings to standardized metadata
ISSUE_METADATA_BY_TYPE: Dict[str, IssueMetadata] = {
    # Secure 2.0 issues
    "HCE catch-up not coded as Roth": {
        "issue_category": "Secure 2.0",
        "severity": "High",
        "correction_hint": (
            "Review HCE catch-up contribution configuration and ensure all HCE catch-up "
            "amounts are coded as Roth contributions per Secure 2.0 requirements. "
            "Coordinate with payroll and recordkeeper to correct source coding and "
            "consult plan document provisions as needed."
        ),
    },
    "Potential catch-up coded in base deferral source": {
        "issue_category": "Secure 2.0",
        "severity": "Medium",
        "correction_hint": (
            "Review payroll source mapping to confirm catch-up contributions are "
            "tracked in a separate source from base deferrals. Recode catch-up dollars "
            "into the appropriate catch-up source to ensure proper tracking and compliance."
        ),
    },
    
    # Population validation issues
    "EMPLOYMENT_STATUS_CONFLICT": {
        "issue_category": "Population Validation",
        "severity": "High",
        "correction_hint": (
            "Review payroll and recordkeeper census status values for the participant. "
            "Confirm current employment status and update source records or operational feeds as needed."
        ),
    },
    "POST_TERMINATION_COMPENSATION": {
        "issue_category": "Population Validation",
        "severity": "High",
        "correction_hint": (
            "Investigate compensation paid more than 30 days after the recordkeeper termination date. "
            "Confirm whether the payment, termination date, or census transmission requires operational follow-up."
        ),
    },
    
    # Eligibility issues
    "Late start after eligibility": {
        "issue_category": "Eligibility",
        "severity": "High",
        "correction_hint": (
            "Confirm eligibility tracking and enrollment processes. Make participants "
            "whole for missed deferrals and employer match where appropriate under plan "
            "provisions and EPCRS correction procedures."
        ),
    },
    "No contributions after eligibility": {
        "issue_category": "Eligibility",
        "severity": "High",
        "correction_hint": (
            "Review participant enrollment and eligibility tracking. Confirm whether "
            "participant should have been enrolled and make participants whole for "
            "missed deferrals and match where appropriate under plan provisions and "
            "EPCRS correction procedures."
        ),
    },
    
    # 402(g) compensation limit issues
    "402(g) excess deferrals": {
        "issue_category": "Comp/402(g)",
        "severity": "High",
        "correction_hint": (
            "Review year-to-date deferrals against IRS 402(g) limits. Process corrective "
            "distribution of excess deferrals (and associated earnings) by the applicable "
            "deadline to avoid plan disqualification. Adjust future payroll deferrals to "
            "stay within limits."
        ),
    },
    
    # Match calculation issues
    "Under-match": {
        "issue_category": "Match",
        "severity": "High",
        "correction_hint": (
            "Reconcile employer match calculations against plan formula and participant "
            "deferrals. Correct any under-credited match amounts and adjust ongoing "
            "match calculations to ensure compliance with plan provisions."
        ),
    },
    "Over-match": {
        "issue_category": "Match",
        "severity": "Medium",
        "correction_hint": (
            "Identify the cause of over-credited employer match. Coordinate with employer "
            "and recordkeeper to determine appropriate correction steps, which may include "
            "adjusting future match calculations or processing corrective distributions "
            "where applicable."
        ),
    },
    "Under-match from compensation definition variance": {
        "issue_category": "Compensation/Match",
        "severity": "High",
        "correction_hint": (
            "Review eligible compensation mapping and employer match calculations. "
            "Recalculate affected participant match using plan compensation rules and "
            "coordinate correction of any under-credited match amounts."
        ),
    },
    "Over-match from excluded compensation included": {
        "issue_category": "Compensation/Match",
        "severity": "Medium",
        "correction_hint": (
            "Review excluded compensation treatment in payroll and match calculation logic. "
            "Confirm whether over-credited match should be corrected under plan procedures."
        ),
    },
    "Match paid to excluded employee class": {
        "issue_category": "Compensation/Match",
        "severity": "High",
        "correction_hint": (
            "Review employee class eligibility and match source coding. Confirm whether the "
            "participant class is excluded under plan rules and correct any match paid in error."
        ),
    },
    "Potential annual true-up timing difference": {
        "issue_category": "Compensation/Match",
        "severity": "Low",
        "correction_hint": (
            "Review annual true-up provisions and year-to-date match calculations before "
            "treating this per-payroll variance as a correction item."
        ),
    },
    "Compensation source data incomplete": {
        "issue_category": "Compensation/Match",
        "severity": "Medium",
        "correction_hint": (
            "Review payroll and recordkeeper source files for missing compensation, deferral, "
            "match, or employee class fields before relying on match variance results."
        ),
    },
}


# Default metadata for unrecognized issue types
DEFAULT_METADATA: IssueMetadata = {
    "issue_category": "Other",
    "severity": "Medium",
    "correction_hint": (
        "Review this item with your TPA, recordkeeper, and plan document to determine "
        "the appropriate correction steps."
    ),
}


def get_issue_metadata(issue_type: str) -> IssueMetadata:
    """
    Return standardized metadata for a given issue_type.
    
    This function provides:
    - issue_category: Short category string (e.g., "Secure 2.0", "Eligibility", "Comp/402(g)")
    - severity: One of "Low", "Medium", "High"
    - correction_hint: 1-2 sentence standardized correction suggestion
    
    Args:
        issue_type: The issue type string as defined in analyzer modules
            (e.g., "HCE catch-up not coded as Roth", "Late start after eligibility")
    
    Returns:
        IssueMetadata dict with issue_category, severity, and correction_hint.
        Falls back to DEFAULT_METADATA if the issue_type is not recognized.
    
    Example:
        >>> metadata = get_issue_metadata("HCE catch-up not coded as Roth")
        >>> metadata["issue_category"]
        'Secure 2.0'
        >>> metadata["severity"]
        'High'
    """
    return ISSUE_METADATA_BY_TYPE.get(issue_type, DEFAULT_METADATA)

