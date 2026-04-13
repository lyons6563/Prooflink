import pandas as pd

from vendor_detection import UNKNOWN_VENDOR, detect_vendors


def test_detect_vendors_handles_each_side_independently():
    payroll_df = pd.DataFrame(
        {
            "Emp ID": [1001],
            "Check Date": ["2025-01-15"],
            "EE Deferral": [100],
            "EE Roth": [0],
        }
    )
    empty_rk_df = pd.DataFrame()

    result = detect_vendors(payroll_df=payroll_df, rk_df=empty_rk_df)

    assert result.payroll_vendor != UNKNOWN_VENDOR
    assert result.payroll_confidence > 0
    assert result.rk_vendor == UNKNOWN_VENDOR
    assert result.rk_confidence == 0.0


def test_detect_vendors_respects_hints_for_non_empty_inputs():
    payroll_df = pd.DataFrame(
        {
            "RandomCol": [1],
            "AnotherCol": [2],
        }
    )
    rk_df = pd.DataFrame(
        {
            "random": [1],
            "cols": [2],
        }
    )

    result = detect_vendors(
        payroll_df=payroll_df,
        rk_df=rk_df,
        payroll_vendor_hint="ADP",
        rk_vendor_hint="Empower",
    )

    assert result.payroll_vendor == "ADP"
    assert result.rk_vendor == "Empower"


def test_detect_vendors_uses_hints_even_when_inputs_are_empty():
    result = detect_vendors(
        payroll_df=pd.DataFrame(),
        rk_df=pd.DataFrame(),
        payroll_vendor_hint="ADP",
        rk_vendor_hint="Empower",
    )

    assert result.payroll_vendor == "ADP"
    assert result.rk_vendor == "Empower"
    assert result.payroll_confidence == 0.0
    assert result.rk_confidence == 0.0
