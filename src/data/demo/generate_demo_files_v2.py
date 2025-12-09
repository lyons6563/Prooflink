import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)


def generate_employee_ids(n: int):
    return [f"E{1000 + i}" for i in range(n)]


def generate_pay_dates(n: int):
    base = datetime(2025, 1, 1)
    return [base + timedelta(days=14 * i) for i in range(n)]


def generate_clean_demo():
    """
    Clean plan:
    - Perfect reconciliation
    - No late deposits
    - No Secure 2.0 issues
    """
    employees = generate_employee_ids(50)
    pay_dates = generate_pay_dates(10)

    payroll_rows = []
    rk_rows = []

    for ee in employees:
        for pd_ in pay_dates:
            # Simple fixed structure
            def_amount = round(np.random.uniform(50, 200), 2)
            roth_amount = 0.0
            loan_amount = 0.0

            payroll_rows.append(
                [
                    ee,
                    pd_,
                    def_amount,
                    roth_amount,
                    loan_amount,
                    def_amount,  # deferral_pre synonym
                    roth_amount,  # deferral_roth synonym
                ]
            )

            rk_rows.append(
                [
                    ee,
                    pd_ + timedelta(days=3),
                    def_amount,
                    roth_amount,
                    loan_amount,
                ]
            )

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=[
            "employee_id",
            "pay_date",
            "def_amount",
            "roth_amount",
            "loan_amount",
            "deferral_pre",   # synonym for engine normalization
            "deferral_roth",  # synonym for engine normalization
        ],
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=[
            "employee_id",
            "record_date",
            "def_amount",
            "roth_amount",
            "loan_amount",
        ],
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "demo_clean_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "demo_clean_rk.csv"), index=False)


def generate_messy_demo():
    """
    Messy plan:
    - Mismatches
    - Missing deposits
    - Late deposits
    """
    employees = generate_employee_ids(100)
    pay_dates = generate_pay_dates(12)

    payroll_rows = []
    rk_rows = []

    for ee in employees:
        for pd_ in pay_dates:
            base = round(np.random.uniform(40, 220), 2)
            roth = round(base * 0.05, 2)
            loan = 0.0

            # Payroll row (always present)
            payroll_rows.append(
                [
                    ee,
                    pd_,
                    base,
                    roth,
                    loan,
                    base,  # deferral_pre synonym
                    roth,  # deferral_roth synonym
                ]
            )

            # RK row (sometimes missing or mismatched)
            if random.random() > 0.10:  # 10% missing in RK
                rk_def = (
                    base
                    if random.random() > 0.15
                    else base + round(np.random.uniform(-10, 10), 2)
                )
                rk_roth = roth
                rk_loan = loan

                rk_rows.append(
                    [
                        ee,
                        pd_ + timedelta(days=random.randint(1, 10)),  # potentially late
                        rk_def,
                        rk_roth,
                        rk_loan,
                    ]
                )

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=[
            "employee_id",
            "pay_date",
            "def_amount",
            "roth_amount",
            "loan_amount",
            "deferral_pre",
            "deferral_roth",
        ],
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=[
            "employee_id",
            "record_date",
            "def_amount",
            "roth_amount",
            "loan_amount",
        ],
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "demo_messy_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "demo_messy_rk.csv"), index=False)


def generate_secure20_demo():
    """
    Secure 2.0 plan:
    - Forces obvious HCE violations age 50+
    - Non-HCEs remain compliant
    - Ensures demo always surfaces meaningful rule-breaks
    """
    employees = generate_employee_ids(50)
    pay_dates = generate_pay_dates(8)

    payroll_rows = []
    rk_rows = []

    for ee in employees:
        age = random.choice([30, 40, 50, 55, 60])
        is_hce = random.random() < 0.30  # 30% HCEs

        for pd_ in pay_dates:

            # Base deferral
            base_deferral = round(np.random.uniform(50, 180), 2)

            # --- SECURE 2.0 LOGIC DEMO ---
            # If HCE age 50+ → FORCE a violation by making all catch-up deferrals PRE-TAX
            if age >= 50 and is_hce:
                # Pretend they contributed catch-up but incorrectly as pre-tax
                catchup_amount = round(np.random.uniform(100, 300), 2)

                def_amount = base_deferral + catchup_amount  # all pre-tax (violation)
                roth_amount = 0.0                           # required method NOT used
            else:
                # Compliant or non-catch-up cases
                def_amount = base_deferral
                roth_amount = 0.0

            loan_amount = 0.0

            payroll_rows.append(
                [
                    ee,
                    age,
                    int(is_hce),
                    pd_,
                    def_amount,
                    roth_amount,
                    loan_amount,
                    def_amount,   # synonyms
                    roth_amount,
                ]
            )

            rk_rows.append(
                [
                    ee,
                    pd_ + timedelta(days=2),
                    def_amount,
                    roth_amount,
                    loan_amount,
                ]
            )

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=[
            "employee_id",
            "age",
            "is_hce",
            "pay_date",
            "def_amount",
            "roth_amount",
            "loan_amount",
            "deferral_pre",
            "deferral_roth",
        ],
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=["employee_id", "record_date", "def_amount", "roth_amount", "loan_amount"],
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "demo_secure20_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "demo_secure20_rk.csv"), index=False)



if __name__ == "__main__":
    print("Generating demo datasets...")
    generate_clean_demo()
    generate_messy_demo()
    generate_secure20_demo()
    print("All demo datasets generated successfully.")
