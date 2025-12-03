"""
Vendor detection and column normalization subsystem.

Provides vendor signatures, column mappings, and confidence scoring
for automatic payroll and recordkeeper vendor identification.
"""

import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
from difflib import SequenceMatcher


# =========================
# VENDOR SIGNATURES
# =========================

PAYROLL_VENDOR_SIGNATURES = {
    "ADP": {
        "signature_keywords": ["emp id", "check date", "ee deferral", "ee roth"],
        "column_map": {
            "employee_id": ["emp id", "employee_id", "emp_id", "employee number"],
            "pay_date": ["check date", "pay date", "payroll date", "check_date"],
            "EE Deferral $": ["ee deferral", "ee deferral $", "deferral", "pretax deferral"],
            "EE Roth $": ["ee roth", "ee roth $", "roth deferral", "roth"],
            "loan_amount": ["loan repay", "loan payment", "loan repay $"],
        }
    },
    "Paychex": {
        "signature_keywords": ["employee number", "pay period end", "401k deferral", "roth 401k"],
        "column_map": {
            "employee_id": ["employee number", "emp number", "employee_id"],
            "pay_date": ["pay period end", "pay date", "check date"],
            "EE Deferral $": ["401k deferral", "pretax deferral", "deferral amount"],
            "EE Roth $": ["roth 401k", "roth deferral", "roth amount"],
            "loan_amount": ["loan repayment", "loan payment"],
        }
    },
    "Paylocity": {
        "signature_keywords": ["employee id", "pay date", "pretax", "roth contribution"],
        "column_map": {
            "employee_id": ["employee id", "employee_id", "emp id"],
            "pay_date": ["pay date", "payroll date", "check date"],
            "EE Deferral $": ["pretax", "pretax deferral", "401k pretax"],
            "EE Roth $": ["roth contribution", "roth deferral", "401k roth"],
            "loan_amount": ["loan payment", "loan repayment"],
        }
    },
    "QuickBooks": {
        "signature_keywords": ["employee", "payroll date", "401k employee", "401k roth"],
        "column_map": {
            "employee_id": ["employee", "employee id", "employee_id"],
            "pay_date": ["payroll date", "pay date", "check date"],
            "EE Deferral $": ["401k employee", "401k pretax", "deferral"],
            "EE Roth $": ["401k roth", "roth 401k", "roth"],
            "loan_amount": ["loan", "loan payment"],
        }
    },
    "Workday": {
        "signature_keywords": ["worker id", "pay period end", "pretax contribution", "roth contribution"],
        "column_map": {
            "employee_id": ["worker id", "employee id", "worker_id"],
            "pay_date": ["pay period end", "pay date", "period end date"],
            "EE Deferral $": ["pretax contribution", "pretax", "401k pretax"],
            "EE Roth $": ["roth contribution", "roth", "401k roth"],
            "loan_amount": ["loan payment", "loan repayment"],
        }
    },
    "UKG": {
        "signature_keywords": ["employee number", "pay date", "pretax deferral", "roth deferral"],
        "column_map": {
            "employee_id": ["employee number", "emp number", "employee_id"],
            "pay_date": ["pay date", "payroll date", "check date"],
            "EE Deferral $": ["pretax deferral", "pretax", "401k pretax"],
            "EE Roth $": ["roth deferral", "roth", "401k roth"],
            "loan_amount": ["loan payment", "loan repayment"],
        }
    },
    "BambooHR": {
        "signature_keywords": ["employee id", "pay date", "401k pretax", "401k roth"],
        "column_map": {
            "employee_id": ["employee id", "employee_id", "emp id"],
            "pay_date": ["pay date", "payroll date", "check date"],
            "EE Deferral $": ["401k pretax", "pretax", "pretax deferral"],
            "EE Roth $": ["401k roth", "roth", "roth deferral"],
            "loan_amount": ["loan payment", "loan repayment"],
        }
    },
    "TriNet": {
        "signature_keywords": ["employee id", "pay period end", "pretax 401k", "roth 401k"],
        "column_map": {
            "employee_id": ["employee id", "employee_id", "emp id"],
            "pay_date": ["pay period end", "pay date", "payroll date"],
            "EE Deferral $": ["pretax 401k", "pretax", "401k pretax"],
            "EE Roth $": ["roth 401k", "roth", "401k roth"],
            "loan_amount": ["loan payment", "loan repayment"],
        }
    },
}

RK_VENDOR_SIGNATURES = {
    "Empower": {
        "signature_keywords": ["part_id", "post_date", "ee_pretax", "ee_roth"],
        "column_map": {
            "employee_id": ["part_id", "participant id", "employee_id"],
            "deposit_date": ["post_date", "post date", "deposit date"],
            "EE Deferral $": ["ee_pretax", "ee pretax", "pretax contribution"],
            "EE Roth $": ["ee_roth", "ee roth", "roth contribution"],
            "loan_amount": ["loan_contr", "loan contribution", "loan payment"],
        }
    },
    "Fidelity": {
        "signature_keywords": ["participant id", "transaction date", "pretax contribution", "roth contribution"],
        "column_map": {
            "employee_id": ["participant id", "employee id", "participant_id"],
            "deposit_date": ["transaction date", "deposit date", "post date"],
            "EE Deferral $": ["pretax contribution", "pretax", "401k pretax"],
            "EE Roth $": ["roth contribution", "roth", "401k roth"],
            "loan_amount": ["loan contribution", "loan payment"],
        }
    },
    "Vanguard": {
        "signature_keywords": ["participant id", "transaction date", "pretax", "roth"],
        "column_map": {
            "employee_id": ["participant id", "employee id", "participant_id"],
            "deposit_date": ["transaction date", "deposit date", "post date"],
            "EE Deferral $": ["pretax", "pretax contribution", "401k pretax"],
            "EE Roth $": ["roth", "roth contribution", "401k roth"],
            "loan_amount": ["loan", "loan contribution"],
        }
    },
}


# =========================
# CONFIDENCE SCORING
# =========================

def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def calculate_vendor_confidence(
    df: pd.DataFrame,
    vendor_name: str,
    signatures: Dict[str, Dict]
) -> float:
    """
    Calculate confidence score for vendor detection (0.0 to 1.0).
    
    Based on:
    - Percent of signature_keywords matched
    - Strength of fuzzy matches
    - Column distribution patterns
    """
    if vendor_name not in signatures:
        return 0.0
    
    vendor_config = signatures[vendor_name]
    
    # Safeguard: ensure vendor_config is a dict
    if not isinstance(vendor_config, dict):
        return 0.0
    
    signature_keywords = vendor_config.get("signature_keywords", [])
    column_map = vendor_config.get("column_map", {})
    
    if not signature_keywords:
        return 0.0
    
    # Normalize column names for matching
    cols_normalized = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
    cols_lower = [col.lower() for col in df.columns]
    
    # 1. Signature keyword matching (40% weight)
    keyword_matches = 0
    for keyword in signature_keywords:
        keyword_norm = re.sub(r'\s+', '_', keyword.strip().lower())
        if any(keyword_norm in col or col in keyword_norm for col in cols_normalized):
            keyword_matches += 1
        elif any(keyword.lower() in col or col in keyword.lower() for col in cols_lower):
            keyword_matches += 1
    
    keyword_score = keyword_matches / len(signature_keywords) if signature_keywords else 0.0
    
    # 2. Column map fuzzy matching (40% weight)
    column_match_score = 0.0
    total_columns_to_match = 0
    
    for canonical_name, variants in column_map.items():
        total_columns_to_match += 1
        best_match = 0.0
        
        for variant in variants:
            variant_norm = re.sub(r'\s+', '_', variant.strip().lower())
            for col in cols_normalized:
                sim = similarity_score(variant_norm, col)
                if sim > best_match:
                    best_match = sim
        
        column_match_score += best_match
    
    column_score = column_match_score / total_columns_to_match if total_columns_to_match > 0 else 0.0
    
    # 3. Column distribution pattern (20% weight)
    # Check if we have expected number of columns (heuristic)
    expected_min_cols = len(column_map)
    pattern_score = min(1.0, len(df.columns) / max(expected_min_cols, 5))
    
    # Weighted combination
    confidence = (
        keyword_score * 0.4 +
        column_score * 0.4 +
        pattern_score * 0.2
    )
    
    return min(1.0, max(0.0, confidence))


# =========================
# VENDOR DETECTION
# =========================

def detect_vendor_with_confidence(
    df: pd.DataFrame,
    signatures: Dict[str, Dict],
    vendor_hint: Optional[str] = None
) -> Tuple[Optional[str], float]:
    """
    Detect vendor from dataframe with confidence scoring.
    
    Returns:
        Tuple of (vendor_name, confidence_score)
        confidence_score is 0.0 to 1.0
    """
    if vendor_hint and vendor_hint in signatures:
        # If hint provided, calculate confidence for that vendor
        confidence = calculate_vendor_confidence(df, vendor_hint, signatures)
        return (vendor_hint, confidence)
    
    # Normalize column names for matching
    cols_normalized = [re.sub(r'\s+', '_', col.strip().lower()) for col in df.columns]
    cols_lower = [col.lower() for col in df.columns]
    
    best_vendor = None
    best_confidence = 0.0
    
    # Try each vendor
    for vendor_name, vendor_config in signatures.items():
        # Safeguard: ensure vendor_config is a dict
        if not isinstance(vendor_config, dict):
            continue
        
        signature_keywords = vendor_config.get("signature_keywords", [])
        
        if not signature_keywords:
            continue
        
        # Check if all signature keywords match
        match_all = True
        for keyword in signature_keywords:
            keyword_norm = re.sub(r'\s+', '_', keyword.strip().lower())
            found = (
                any(keyword_norm in col or col in keyword_norm for col in cols_normalized) or
                any(keyword.lower() in col or col in keyword.lower() for col in cols_lower)
            )
            if not found:
                match_all = False
                break
        
        if match_all:
            # Calculate confidence for this vendor
            confidence = calculate_vendor_confidence(df, vendor_name, signatures)
            if confidence > best_confidence:
                best_confidence = confidence
                best_vendor = vendor_name
    
    return (best_vendor, best_confidence)


# =========================
# COLUMN MAPPING
# =========================

def apply_vendor_column_mapping(
    df: pd.DataFrame,
    vendor_name: Optional[str],
    signatures: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Apply vendor-specific column mapping to dataframe.
    
    Maps columns based on vendor's column_map configuration.
    """
    if not vendor_name or vendor_name not in signatures:
        return df
    
    vendor_config = signatures[vendor_name]
    
    # Safeguard: ensure vendor_config is a dict
    if not isinstance(vendor_config, dict):
        return df
    
    column_map = vendor_config.get("column_map", {})
    
    if not column_map:
        return df
    
    df = df.copy()
    rename_map = {}
    
    # Normalize column names for matching
    cols_normalized = {}
    for col in df.columns:
        normalized = re.sub(r'\s+', '_', col.strip().lower())
        cols_normalized[col] = normalized
    
    # Build rename mapping
    for canonical_name, variants in column_map.items():
        best_match_col = None
        best_similarity = 0.0
        threshold = 0.6  # Minimum similarity threshold
        
        for variant in variants:
            variant_norm = re.sub(r'\s+', '_', variant.strip().lower())
            for original_col, col_norm in cols_normalized.items():
                # Exact match first
                if variant_norm == col_norm:
                    best_match_col = original_col
                    best_similarity = 1.0
                    break
                # Fuzzy match
                sim = similarity_score(variant_norm, col_norm)
                if sim > best_similarity and sim >= threshold:
                    best_similarity = sim
                    best_match_col = original_col
        
        if best_match_col and best_match_col not in rename_map.values():
            rename_map[best_match_col] = canonical_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df

