import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

OUTPUT_DIR = "data/raw/demo/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

def generate_employee_ids(n):
    return [f"E{1000+i}" for i in range(n)]

def generate_pay_dates(n):
    base = datetime(2025, 1, 1)
    return [base + timedelta(days=14*i) for i in range(n)]

def generate_clean_demo():
    employees = generate_employee_ids(50)
    pay_dates = generate_pay_dates(10)

    rows = []
    for ee in employees:
        for pd in pay_dates:
            amount = round(np.random.uniform(50, 200), 2)
            rows.append([ee, pd, amount, 0])  # no Roth

    df_payroll = pd.DataFrame(rows, columns=["employee_id", "pay_date", "deferral_pre", "deferral_roth"])

    df_rk = df_payroll.copy()
    df_rk["record_date"] = df_rk["pay_date"] + timedelta(days=3)

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "clean_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "clean_rk.csv"), index=False)


def generate_messy_demo():
    employees = generate_employee_ids(100)
    pay_dates = generate_pay_dates(12)

    payroll_rows = []
    rk_rows = []

    for ee in employees:
        for pd in pay_dates:
            base = round(np.random.uniform(40, 220), 2)

            payroll_rows.append([
                ee,
                pd,
                base,
                round(base * 0.05, 2)
            ])

            # RK introduces mismatches, missing records, late records
            if random.random() > 0.1:  # 10% missing in RK
                rk_amount = base if random.random() > 0.15 else base + round(np.random.uniform(-10, 10), 2)
                rk_rows.append([
                    ee,
                    pd + timedelta(days=random.randint(1, 10)),  # potentially late
                    rk_amount,
                    round(rk_amount * 0.05, 2)
                ])

    df_payroll = pd.DataFrame(payroll_rows, columns=["employee_id", "pay_date", "deferral_pre", "deferral_roth"])
    df_rk = pd.DataFrame(rk_rows, columns=["employee_id", "record_date", "deferral_pre", "deferral_roth"])

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "messy_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "messy_rk.csv"), index=False)


def generate_secure20_demo():
    employees = generate_employee_ids(50)
    pay_dates = generate_pay_dates(8)

    payroll_rows = []
    rk_rows = []

    for ee in employees:
        age = random.choice([30, 40, 50, 55, 60])
        is_hce = random.random() < 0.3  # ~30% HCEs

        for pd in pay_dates:
            pre = round(np.random.uniform(50, 180), 2)

            roth = 0
            if age >= 50:
                # generate some violations
                roth = round(np.random.uniform(50, 150), 2) if is_hce and random.random() < 0.5 else 0

            payroll_rows.append([ee, age, is_hce, pd, pre, roth])

            rk_rows.append([ee, pd + timedelta(days=2), pre, roth])

    df_payroll = pd.DataFrame(
        payroll_rows,
        columns=["employee_id", "age", "is_hce", "pay_date", "deferral_pre", "deferral_roth"]
    )

    df_rk = pd.DataFrame(
        rk_rows,
        columns=["employee_id", "record_date", "deferral_pre", "deferral_roth"]
    )

    df_payroll.to_csv(os.path.join(OUTPUT_DIR, "secure20_payroll.csv"), index=False)
    df_rk.to_csv(os.path.join(OUTPUT_DIR, "secure20_rk.csv"), index=False)


if __name__ == "__main__":
    print("Generating demo datasets...")
    generate_clean_demo()
    generate_messy_demo()
    generate_secure20_demo()
    print("Done.")
