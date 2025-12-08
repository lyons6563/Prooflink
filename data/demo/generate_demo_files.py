import os
import csv
from datetime import datetime, timedelta

RAW_DIR = os.path.join("data", "raw")

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)

def write_csv(path, rows, headers):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")

# ------------------------------------------------------------
# CLEAN DATASET (everything matches)
# ------------------------------------------------------------
def generate_clean():
    payroll_headers = [
        "Employee_ID","Full_Name","Gross_Pay","HCE_Flag","Pay_Date",
        "EE_PreTax_Def","EE_Roth_Def","Loan_Repayment"
    ]
    rk_headers = [
        "Participant_ID","EE_PreTax_Def","EE_Roth_Def",
        "Loan_Repayment","Deposit_Date","Total_Deposit_Amount"
    ]

    payroll_rows = [
        {
            "Employee_ID": "1001",
            "Full_Name": "Alice Example",
            "Gross_Pay": "2500",
            "HCE_Flag": "N",
            "Pay_Date": "2025-11-26",
            "EE_PreTax_Def": "125",
            "EE_Roth_Def": "60",
            "Loan_Repayment": "30",
        }
    ]

    rk_rows = [
        {
            "Participant_ID": "1001",
            "EE_PreTax_Def": "125",
            "EE_Roth_Def": "60",
            "Loan_Repayment": "30",
            "Deposit_Date": "2025-11-27",
            "Total_Deposit_Amount": "215",
        }
    ]

    write_csv(os.path.join(RAW_DIR, "demo_clean_payroll.csv"), payroll_rows, payroll_headers)
    write_csv(os.path.join(RAW_DIR, "demo_clean_rk.csv"), rk_rows, rk_headers)

# ------------------------------------------------------------
# MESSY DATASET (missing deposits, mismatches, late deposits)
# ------------------------------------------------------------
def generate_messy():
    payroll_headers = [
        "Employee_ID","Full_Name","Gross_Pay","HCE_Flag","Pay_Date",
        "EE_PreTax_Def","EE_Roth_Def","Loan_Repayment"
    ]
    rk_headers = [
        "Participant_ID","EE_PreTax_Def","EE_Roth_Def",
        "Loan_Repayment","Deposit_Date","Total_Deposit_Amount"
    ]

    payroll_rows = [
        # Perfect
        {
            "Employee_ID": "2001",
            "Full_Name": "Bob Good",
            "Gross_Pay": "3200",
            "HCE_Flag": "N",
            "Pay_Date": "2025-11-26",
            "EE_PreTax_Def": "160",
            "EE_Roth_Def": "80",
            "Loan_Repayment": "40",
        },
        # Missing in RK
        {
            "Employee_ID": "2002",
            "Full_Name": "Charlie Missing",
            "Gross_Pay": "3000",
            "HCE_Flag": "N",
            "Pay_Date": "2025-11-26",
            "EE_PreTax_Def": "150",
            "EE_Roth_Def": "70",
            "Loan_Repayment": "0",
        },
        # Mismatched amounts
        {
            "Employee_ID": "2003",
            "Full_Name": "Danny Wrong",
            "Gross_Pay": "2800",
            "HCE_Flag": "N",
            "Pay_Date": "2025-11-26",
            "EE_PreTax_Def": "140",
            "EE_Roth_Def": "60",
            "Loan_Repayment": "20",
        },
    ]

    today = datetime(2025,11,26)

    rk_rows = [
        # Perfect
        {
            "Participant_ID": "2001",
            "EE_PreTax_Def": "160",
            "EE_Roth_Def": "80",
            "Loan_Repayment": "40",
            "Deposit_Date": "2025-11-27",
            "Total_Deposit_Amount": "280",
        },
        # Mismatch
        {
            "Participant_ID": "2003",
            "EE_PreTax_Def": "120",  # wrong
            "EE_Roth_Def": "50",     # wrong
            "Loan_Repayment": "10",  # wrong
            "Deposit_Date": "2025-12-10",  # late
            "Total_Deposit_Amount": "180",
        },
        # Only in RK
        {
            "Participant_ID": "9999",
            "EE_PreTax_Def": "90",
            "EE_Roth_Def": "40",
            "Loan_Repayment": "0",
            "Deposit_Date": "2025-11-27",
            "Total_Deposit_Amount": "130",
        }
    ]

    write_csv(os.path.join(RAW_DIR, "demo_messy_payroll.csv"), payroll_rows, payroll_headers)
    write_csv(os.path.join(RAW_DIR, "demo_messy_rk.csv"), rk_rows, rk_headers)

# ------------------------------------------------------------
# SECURE 2.0 DATASET (catch-up violations)
# ------------------------------------------------------------
def generate_secure20():
    payroll_headers = [
        "EmpNumber","Payroll_Run_Date","PreTax_Defl","Roth_Defl",
        "Loan_Pmt","is_hce","catchup_pretax","catchup_roth"
    ]
    rk_headers = [
        "employee_id","deposit_date","EE Deferral $","EE Roth $",
        "Loan_Repayment","Total_Deposit_Amount"
    ]

    payroll_rows = [
        # violation
        {"EmpNumber":"3001","Payroll_Run_Date":"2025-11-26","PreTax_Defl":"200","Roth_Defl":"0",
         "Loan_Pmt":"0","is_hce":"Y","catchup_pretax":"200","catchup_roth":"0"},

        # violation
        {"EmpNumber":"3002","Payroll_Run_Date":"2025-11-26","PreTax_Defl":"220","Roth_Defl":"0",
         "Loan_Pmt":"0","is_hce":"Y","catchup_pretax":"220","catchup_roth":"0"},

        # violation
        {"EmpNumber":"3003","Payroll_Run_Date":"2025-11-26","PreTax_Defl":"150","Roth_Defl":"0",
         "Loan_Pmt":"20","is_hce":"Y","catchup_pretax":"150","catchup_roth":"0"},

        # compliant HCE
        {"EmpNumber":"3004","Payroll_Run_Date":"2025-11-26","PreTax_Defl":"100","Roth_Defl":"200",
         "Loan_Pmt":"10","is_hce":"Y","catchup_pretax":"0","catchup_roth":"200"},

        # non-HCE
        {"EmpNumber":"3005","Payroll_Run_Date":"2025-11-26","PreTax_Defl":"90","Roth_Defl":"50",
         "Loan_Pmt":"0","is_hce":"N","catchup_pretax":"0","catchup_roth":"0"},
    ]

    rk_rows = [
        {"employee_id":"3001","deposit_date":"2025-11-27","EE Deferral $":"200","EE Roth $":"0",
         "Loan_Repayment":"0","Total_Deposit_Amount":"200"},
        {"employee_id":"3002","deposit_date":"2025-11-27","EE Deferral $":"220","EE Roth $":"0",
         "Loan_Repayment":"0","Total_Deposit_Amount":"220"},
        {"employee_id":"3003","deposit_date":"2025-12-05","EE Deferral $":"150","EE Roth $":"0",
         "Loan_Repayment":"20","Total_Deposit_Amount":"170"},
        {"employee_id":"3004","deposit_date":"2025-11-27","EE Deferral $":"100","EE Roth $":"200",
         "Loan_Repayment":"10","Total_Deposit_Amount":"310"},
        {"employee_id":"3005","deposit_date":"2025-11-27","EE Deferral $":"90","EE Roth $":"50",
         "Loan_Repayment":"0","Total_Deposit_Amount":"140"},
    ]

    write_csv(os.path.join(RAW_DIR, "demo_secure20_payroll.csv"), payroll_rows, payroll_headers)
    write_csv(os.path.join(RAW_DIR, "demo_secure20_rk.csv"), rk_rows, rk_headers)

# ------------------------------------------------------------

def main():
    ensure_dirs()
    generate_clean()
    generate_messy()
    generate_secure20()
    print("All demo datasets generated successfully.")

if __name__ == "__main__":
    main()
