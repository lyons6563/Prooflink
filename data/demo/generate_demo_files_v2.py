import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Demo file generator (Balanced / Realistic)
# - Clean demo: perfect reconciliation, no timing, no Secure 2.0 issues
# - Messy demo: mismatches, missing deposits, late deposits
# - Secure 2.0 demo: clear HCE 50+ patterns for Secure 2.0 violations
# -------------------------------------------------------------------

OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)


def generate_employee_ids(n: int):
    return [f"E{1000 + i}" for i in range(n)]


def generate_pay_dates(n: int):
    """Generate n bi-weekly pay dates starting 2025-01-03"""
    base = datetime(2025, 1, 3)
    return [base + timedelta(days=14 * i) for i in range(n)]


# -------------------------------------------------------------------
# CLEAN DEMO
# -------------------------------------------------------------------


def generate_clean_demo():
    """
    Clean plan:
    - Perfect reconciliation
    - All deposits within 3 days
    - No Secure 2.0 violations
    """
    employees = generate_employee_ids(40)
    pay_dates = generate_pay_dates(6)

    payroll_rows = []
    rk_rows = []

    for ee in employees:
        for pd_ in pay_dates:
            def_amount = round(np.random.uniform(50, 200), 2)
            roth_amount = 0.0
            loan_amount = 0.0

            payroll_rows.append(
                [ee, pd_, def_amount, roth_amount, loan_amount, def_amount, roth_amount]
            )

            rk_rows.append(
                [ee, pd_ + timedelta(days=3), def_amount, roth_amount, loan_amount]
            )

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=[
            "employee_id", "pay_date", "def_amount", "roth_amount",
            "loan_amount", "deferral_pre", "deferral_roth",
        ],
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=["employee_id", "record_date", "def_amount", "roth_amount", "loan_amount"],
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "demo_clean_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "demo_clean_rk.csv"), index=False)

    print(f"[CLEAN] Payroll rows: {len(df_payroll)}, RK rows: {len(df_rk)}")


# -------------------------------------------------------------------
# MESSY DEMO (Balanced / Realistic)
# -------------------------------------------------------------------


def generate_messy_demo():
    """
    Messy plan (Balanced realism):
    - Deferral mismatches on ~20-25% of rows
    - ~10-15% missing RK deposits (missing deposits)
    - ~15-20% late deposits (> 5 days)
    """
    employees = generate_employee_ids(80)
    pay_dates = generate_pay_dates(8)

    payroll_rows = []
    rk_rows = []

    missing_count = 0
    late_count = 0
    mismatch_count = 0

    for ee in employees:
        for pd_ in pay_dates:
            def_amount = round(np.random.uniform(40, 220), 2)
            roth_amount = round(def_amount * np.random.uniform(0.0, 0.15), 2)
            loan_amount = 0.0

            payroll_rows.append(
                [ee, pd_, def_amount, roth_amount, loan_amount, def_amount, roth_amount]
            )

            if random.random() < 0.12:  # ~12% missing
                missing_count += 1
                continue

            rk_def = def_amount
            rk_roth = roth_amount
            rk_loan = loan_amount

            if random.random() < 0.22:
                delta = round(np.random.uniform(-25, 25), 2)
                rk_def = max(0.0, rk_def + delta)
                mismatch_count += 1

            if random.random() < 0.18:  # ~18% late
                offset_days = random.randint(6, 12)
                late_count += 1
            else:
                offset_days = random.randint(1, 4)

            record_date = pd_ + timedelta(days=offset_days)

            rk_rows.append([ee, record_date, rk_def, rk_roth, rk_loan])

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=[
            "employee_id", "pay_date", "def_amount", "roth_amount",
            "loan_amount", "deferral_pre", "deferral_roth",
        ],
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=["employee_id", "record_date", "def_amount", "roth_amount", "loan_amount"],
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "demo_messy_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "demo_messy_rk.csv"), index=False)

    print(
        f"[MESSY] Payroll rows: {len(df_payroll)}, RK rows: {len(df_rk)}, "
        f"missing deposits: {missing_count}, late deposits: {late_count}, "
        f"deferral mismatches (approx): {mismatch_count}"
    )


# -------------------------------------------------------------------
# SECURE 2.0 DEMO (Balanced / Realistic)
# -------------------------------------------------------------------


def generate_secure20_demo():
    """
    Secure 2.0 plan:
    - Mix of non-HCE / HCE
    - HCE age 50+ have elevated pre-tax deferrals with zero Roth
      to create clear "this looks like pre-tax catch-up" patterns
    - Non-HCEs and younger participants look normal / mixed
    """
    employees = generate_employee_ids(60)
    pay_dates = generate_pay_dates(8)

    payroll_rows = []
    rk_rows = []
    potential_violations = 0

    for ee in employees:
        age = random.choice([35, 42, 48, 50, 55, 60])
        is_hce = random.random() < 0.35  # ~35% HCEs

        for pd_ in pay_dates:
            base_def = round(np.random.uniform(40, 190), 2)
            def_amount = base_def
            roth_amount = 0.0

            if not is_hce or age < 50:
                if random.random() < 0.4:
                    roth_amount = round(base_def * np.random.uniform(0.2, 0.6), 2)
                    def_amount = round(base_def * np.random.uniform(0.5, 0.9), 2)

            if is_hce and age >= 50:
                catchup = round(np.random.uniform(120, 320), 2)
                def_amount = base_def + catchup
                roth_amount = 0.0
                potential_violations += 1

            loan_amount = 0.0

            payroll_rows.append(
                [ee, age, int(is_hce), pd_, def_amount, roth_amount, loan_amount,
                 def_amount, roth_amount]
            )

            record_date = pd_ + timedelta(days=2)
            rk_rows.append([ee, record_date, def_amount, roth_amount, loan_amount])

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=[
            "employee_id", "age", "is_hce", "pay_date", "def_amount", "roth_amount",
            "loan_amount", "deferral_pre", "deferral_roth",
        ],
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=["employee_id", "record_date", "def_amount", "roth_amount", "loan_amount"],
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "demo_secure20_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "demo_secure20_rk.csv"), index=False)

    print(
        f"[SECURE20] Payroll rows: {len(df_payroll)}, RK rows: {len(df_rk)}, "
        f"HCE 50+ potential violations: {potential_violations}"
    )


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating demo datasets (Balanced realism)...")
    generate_clean_demo()
    generate_messy_demo()
    generate_secure20_demo()
    print("All demo datasets generated successfully.")
