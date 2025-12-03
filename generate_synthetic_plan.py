from pathlib import Path
import random
import pandas as pd
from pandas.tseries.offsets import BusinessDay

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def generate_synthetic_plan(
    num_emps: int = 400,
    num_periods: int = 6,
    start_date: str = "2025-01-03",
    payroll_filename: str = "payroll_adp_synthetic_400.csv",
    rk_filename: str = "rk_empower_synthetic_400.csv",
) -> None:
    """
    Generate a synthetic ADP-style payroll file and Empower-style recordkeeper file.

    - Biweekly payroll for num_periods
    - 400 employees
    - Pretax, Roth, After-tax, Loan repayments
    - 5% employer match
    - Business-day lags between payroll date and deposit (1–10 days)
    - A small percentage of amount mismatches on RK side
    """

    print("Starting synthetic plan generation...")

    start = pd.to_datetime(start_date)
    # Biweekly payroll periods
    periods = [start + pd.to_timedelta(14 * i, unit="D") for i in range(num_periods)]

    payroll_rows = []
    rk_rows = []

    for emp_idx in range(num_emps):
        emp_id = 100001 + emp_idx

        # Base employee profile
        base_salary = random.randint(40_000, 160_000)

        # Auto-enrollment style: most people at 3–6%, some higher, some at 0
        annual_def_pct = random.choices(
            [0, 3, 5, 6, 8, 10], weights=[0.15, 0.25, 0.30, 0.20, 0.05, 0.05]
        )[0]
        # Roth usage
        roth_pct = random.choices([0, 2, 4], weights=[0.6, 0.25, 0.15])[0]
        # After-tax
        aftertax_pct = random.choices([0, 2], weights=[0.8, 0.2])[0]

        # Loans: ~20% have one
        has_loan = random.random() < 0.2
        annual_loan_pmt = random.randint(600, 3_600) if has_loan else 0

        per_period_salary = base_salary / 26  # biweekly
        per_period_loan = annual_loan_pmt / 26 if has_loan else 0

        for p_date in periods:
            # Add some noise to wages
            gross = per_period_salary + random.uniform(-50, 50)

            pretax = round(gross * annual_def_pct / 100, 2)
            roth = round(gross * roth_pct / 100, 2)
            aftertax = round(gross * aftertax_pct / 100, 2)
            loan = round(per_period_loan, 2)

            # 5% match on total contributions (capped at 5% of pay)
            match_base = pretax + roth + aftertax
            match_cap = gross * 0.05
            match = round(min(match_base, match_cap), 2)

            # -------------------------
            # Payroll (ADP-style) row
            # -------------------------
            payroll_rows.append(
                {
                    "EmpNumber": emp_id,
                    "Payroll_Run_Date": p_date.strftime("%Y-%m-%d"),
                    "PreTax_Defl": pretax,
                    "Roth_Defl": roth,
                    "AfterTax_Defl": aftertax,
                    "Loan_Pmt": loan,
                    "Gross_Wages": round(gross, 2),
                    "Match_5pct": match,
                    "Paygroup": "REG",
                    "LocationCode": random.choice(["ATL", "SAV", "PHX", "DEN"]),
                    "Is_Offcycle": 0,
                    "Is_Prior_Period_Adj": 0,
                    "Years_of_Service": random.randint(0, 10),
                }
            )

            # -------------------------
            # RK (Empower-style) row
            # -------------------------
            # Business-day lag distribution
            lag_choice = random.random()
            if lag_choice < 0.70:
                lag_days = random.choice([1, 2, 3])  # 70% normal
            elif lag_choice < 0.95:
                lag_days = random.choice([4, 5])     # 25% slightly slow
            else:
                lag_days = random.randint(6, 10)     # 5% truly late

            post_date = p_date + BusinessDay(lag_days)

            # Simulate mismatches/corrections:
            # ~90% of rows match exactly, ~10% have small +/- 2–5% variance.
            if random.random() < 0.10:
                noise_factor = 1 + random.choice([0.02, -0.02, 0.05, -0.05])
            else:
                noise_factor = 1.0

            rk_pretax = round(pretax * noise_factor, 2)
            rk_roth = round(roth * noise_factor, 2)
            rk_aftertax = round(aftertax * noise_factor, 2)
            rk_loan = round(loan * noise_factor, 2)

            rk_rows.append(
                {
                    "Part_ID": emp_id,
                    "Post_Date": post_date.strftime("%Y-%m-%d"),
                    "EE_PreTax": rk_pretax,
                    "EE_Roth": rk_roth,
                    "EE_AfterTax": rk_aftertax,
                    "Loan_Contr": rk_loan,
                    "ER_Match": match,
                    "Source_PreTax": "EE PreTax" if rk_pretax > 0 else "",
                    "Source_Roth": "EE Roth" if rk_roth > 0 else "",
                    "Source_AfterTax": "EE AfterTax" if rk_aftertax > 0 else "",
                    "Source_Loan": "Loan" if rk_loan > 0 else "",
                    "Fund": random.choice(
                        [
                            "Target 2050",
                            "Target 2040",
                            "US Equity",
                            "Intl Equity",
                            "Stable Value",
                        ]
                    ),
                }
            )

    payroll_df = pd.DataFrame(payroll_rows)
    rk_df = pd.DataFrame(rk_rows)

    payroll_path = DATA_RAW / payroll_filename
    rk_path = DATA_RAW / rk_filename

    payroll_df.to_csv(payroll_path, index=False)
    rk_df.to_csv(rk_path, index=False)

    print(f"Generated payroll file:      {payroll_path}  ({len(payroll_df)} rows)")
    print(f"Generated recordkeeper file: {rk_path}  ({len(rk_df)} rows)")


if __name__ == "__main__":
    # When you run:  python generate_synthetic_plan.py
    # this will execute and actually write the files.
    generate_synthetic_plan()
