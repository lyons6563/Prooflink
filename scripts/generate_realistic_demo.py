from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

DEFAULT_EMPLOYEES = 500
DEFAULT_YEAR = 2025
DEFAULT_SEED = 20250630

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "demo"

PAYROLL_COLUMNS = [
    "employee_id", "first_name", "last_name", "pay_date", "pay_period_number",
    "Employee Class", "department", "location", "Regular Compensation", "Overtime",
    "Bonus", "Fringe", "Reimbursement", "Gross Compensation", "Compensation",
    "Plan Compensation", "EE Deferral $", "EE Roth $", "ER Match $",
    "True-Up Enabled", "loan_amount", "is_hce", "catchup_pretax",
    "catchup_roth", "age", "dob", "hire_date", "rehire_date",
    "employment_status", "termination_date",
]

RK_COLUMNS = [
    "employee_id", "pay_date", "deposit_date", "EE Deferral $", "EE Roth $",
    "loan_amount", "recordkeeper_employment_status", "termination_date",
    "source_system", "transaction_type",
]

GROUND_TRUTH_COLUMNS = [
    "scenario_id", "employee_id", "exception_type", "issue_category",
    "engine_area", "source_file", "pay_period_number", "pay_date",
    "expected_detection", "future_only", "description", "planted_payroll_value",
    "planted_recordkeeper_value", "notes",
]

FIRST_NAMES = "Alex Jordan Taylor Morgan Casey Riley Avery Quinn Jamie Cameron Drew Sam Parker Reese Rowan Emerson Hayden Finley Skyler Kendall".split()
LAST_NAMES = "Adams Bennett Chen Diaz Ellis Foster Garcia Hughes Ibrahim Johnson Kaur Lee Miller Nguyen Ortiz Patel Robinson Singh Turner Walker".split()
DEPARTMENTS = ["Operations", "Finance", "Sales", "Engineering", "Clinical", "Logistics", "Customer Care", "HR"]
LOCATIONS = ["Austin", "Charlotte", "Chicago", "Denver", "Phoenix", "Raleigh", "Remote", "Seattle"]
CLASSES = ["Full-Time", "Part-Time", "Intern", "Union Excluded"]


@dataclass(frozen=True)
class Employee:
    employee_id: str
    first_name: str
    last_name: str
    department: str
    location: str
    employee_class: str
    annual_salary: float
    hire_date: date
    rehire_date: date | None
    termination_date: date | None
    dob: date
    is_hce: bool
    deferral_rate: float
    roth_share: float
    has_loan: bool
    loan_payment: float
    true_up_enabled: bool


def money(value: float) -> str:
    value = round(value + 1e-9, 2)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def iso(value: date | None) -> str:
    return value.isoformat() if value else ""


def business_day_add(start: date, days: int) -> date:
    current = start
    remaining = days
    while remaining:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def pay_periods_for_year(year: int, periods: int = 26) -> list[date]:
    first_payday = date(year, 1, 3)
    return [first_payday + timedelta(days=14 * i) for i in range(periods)]


def sample_ids(ids: list[str], start: int, count: int) -> set[str]:
    return set(ids[start:min(start + count, len(ids))])


def scenario_sets(ids: list[str]) -> dict[str, set[str]]:
    if len(ids) < 260:
        return {
            "deferral_mismatch": set(ids[4:5]),
            "loan_mismatch": set(ids[5:6]),
            "late_contribution": set(ids[6:7]),
            "only_in_payroll": set(ids[7:8]),
            "status_conflict": set(ids[8:9]),
            "post_term_comp": set(ids[9:10]),
            "excess_402g": set(ids[10:11]),
            "secure20_hce_pretax": set(ids[11:12]),
            "comp_match_under": set(ids[12:13]),
            "comp_match_over": set(ids[13:14]),
            "excluded_class_match": set(ids[14:15]),
        }
    return {
        "deferral_mismatch": sample_ids(ids, 4, 6),
        "loan_mismatch": sample_ids(ids, 20, 6),
        "late_contribution": sample_ids(ids, 40, 8),
        "only_in_payroll": sample_ids(ids, 65, 3),
        "status_conflict": sample_ids(ids, 90, 5),
        "post_term_comp": sample_ids(ids, 120, 4),
        "excess_402g": sample_ids(ids, 150, 4),
        "secure20_hce_pretax": sample_ids(ids, 180, 4),
        "comp_match_under": sample_ids(ids, 210, 4),
        "comp_match_over": sample_ids(ids, 230, 4),
        "excluded_class_match": sample_ids(ids, 250, 3),
    }


def truth(rows: list[dict[str, str]], **kwargs: object) -> None:
    row = {column: "" for column in GROUND_TRUTH_COLUMNS}
    for key, value in kwargs.items():
        if isinstance(value, date):
            row[key] = value.isoformat()
        elif isinstance(value, bool):
            row[key] = "true" if value else "false"
        else:
            row[key] = str(value)
    row.setdefault("future_only", "false")
    if not row["future_only"]:
        row["future_only"] = "false"
    rows.append(row)


def build_employees(count: int, year: int, rng: random.Random, scenarios: dict[str, set[str]]) -> list[Employee]:
    employees: list[Employee] = []
    ids = [f"E{100001 + i}" for i in range(count)]
    for i, employee_id in enumerate(ids):
        age = rng.randint(24, 62)
        if employee_id in scenarios["secure20_hce_pretax"]:
            age = rng.randint(52, 61)
        if employee_id in scenarios["excess_402g"]:
            age = rng.randint(35, 49)
        dob = date(year - age, rng.randint(1, 12), rng.randint(1, 28))

        hire_date = date(rng.randint(2010, year - 1), rng.randint(1, 12), rng.randint(1, 28))
        termination_date = None
        rehire_date = None
        if i % 41 == 0 and employee_id not in scenarios["status_conflict"] | scenarios["post_term_comp"]:
            termination_date = date(year, rng.choice([7, 8, 9, 10]), rng.choice([4, 11, 18, 25]))
        if i % 89 == 0 and termination_date:
            rehire_date = min(date(year, 12, 1), termination_date + timedelta(days=56))
        if employee_id in scenarios["post_term_comp"]:
            termination_date = date(year, 1, 1)
            rehire_date = None

        employee_class = rng.choices(CLASSES, weights=[82, 12, 3, 3], k=1)[0]
        if employee_id in scenarios["excluded_class_match"]:
            employee_class = "Intern"
        if employee_id in scenarios["comp_match_under"] | scenarios["comp_match_over"]:
            employee_class = "Full-Time"
        is_hce = rng.random() < 0.12 or employee_id in scenarios["excess_402g"] | scenarios["secure20_hce_pretax"]
        if is_hce:
            annual_salary = rng.uniform(155000, 275000)
        elif employee_class == "Part-Time":
            annual_salary = rng.uniform(32000, 76000)
        elif employee_class == "Intern":
            annual_salary = rng.uniform(28000, 42000)
        else:
            annual_salary = rng.uniform(56000, 148000)

        deferral_rate = rng.choices([0, .03, .04, .05, .06, .08, .10, .12], weights=[10, 10, 16, 20, 22, 13, 6, 3], k=1)[0]
        if employee_id in scenarios["excess_402g"]:
            deferral_rate = .18
        if employee_id in scenarios["comp_match_under"] | scenarios["comp_match_over"]:
            deferral_rate = .06
        employees.append(Employee(
            employee_id=employee_id,
            first_name=FIRST_NAMES[i % len(FIRST_NAMES)],
            last_name=LAST_NAMES[(i * 7) % len(LAST_NAMES)],
            department=DEPARTMENTS[(i * 5) % len(DEPARTMENTS)],
            location=LOCATIONS[(i * 3) % len(LOCATIONS)],
            employee_class=employee_class,
            annual_salary=annual_salary,
            hire_date=hire_date,
            rehire_date=rehire_date,
            termination_date=termination_date,
            dob=dob,
            is_hce=is_hce,
            deferral_rate=deferral_rate,
            roth_share=rng.choices([0, .25, .5, 1], weights=[68, 12, 12, 8], k=1)[0],
            has_loan=(rng.random() < .13 or employee_id in scenarios["loan_mismatch"]),
            loan_payment=rng.uniform(42, 185),
            true_up_enabled=(i % 17 == 0),
        ))
    return employees


def status_for(emp: Employee, pay_date: date) -> str:
    if pay_date < emp.hire_date:
        return "Not Yet Hired"
    if emp.termination_date and pay_date > emp.termination_date:
        if emp.rehire_date and pay_date >= emp.rehire_date:
            return "Active"
        return "Terminated"
    return "Active"


def compensation(emp: Employee, period: int, pay_date: date, rng: random.Random, scenarios: dict[str, set[str]]) -> tuple[float, float, float, float, float]:
    status = status_for(emp, pay_date)
    if status not in {"Active", "Terminated"}:
        return 0, 0, 0, 0, 0
    if status == "Terminated":
        if emp.employee_id in scenarios["post_term_comp"] and period == 14:
            return 1800, 0, 0, 0, 0
        return 0, 0, 0, 0, 0
    regular = emp.annual_salary / 26 * (1.015 if period in {11, 12, 23, 24} else 1)
    overtime = regular * rng.uniform(.03, .16) if emp.employee_class in {"Full-Time", "Union Excluded"} and rng.random() < .18 else 0
    bonus = regular * rng.uniform(.20, 1.25) if period in {6, 25} and rng.random() < (.35 if emp.is_hce else .16) else 0
    fringe = regular * rng.uniform(0, .025) if rng.random() < .08 else 0
    reimbursement = rng.uniform(25, 300) if rng.random() < .06 else 0
    return regular, overtime, bonus, fringe, reimbursement


def mapping_yaml() -> str:
    return """# Mapping for the deterministic realistic ProofLink demo dataset.

payroll:
  employee_id: {canonical: "employee_id", examples: ["employee_id"]}
  pay_date: {canonical: "pay_date", examples: ["pay_date"]}
  ee_deferral: {canonical: "EE Deferral $", examples: ["EE Deferral $"]}
  ee_roth: {canonical: "EE Roth $", examples: ["EE Roth $"]}
  loan_amount: {canonical: "loan_amount", examples: ["loan_amount"]}
  is_hce: {canonical: "is_hce", examples: ["is_hce"]}
  catchup_pretax: {canonical: "catchup_pretax", examples: ["catchup_pretax"]}
  catchup_roth: {canonical: "catchup_roth", examples: ["catchup_roth"]}
  employment_status: {canonical: "employment_status", examples: ["employment_status"]}
  termination_date: {canonical: "termination_date", examples: ["termination_date"]}

recordkeeper:
  employee_id: {canonical: "employee_id", examples: ["employee_id"]}
  deposit_date: {canonical: "deposit_date", examples: ["deposit_date"]}
  ee_deferral: {canonical: "EE Deferral $", examples: ["EE Deferral $"]}
  ee_roth: {canonical: "EE Roth $", examples: ["EE Roth $"]}
  loan_amount: {canonical: "loan_amount", examples: ["loan_amount"]}
  employment_status: {canonical: "recordkeeper_employment_status", examples: ["recordkeeper_employment_status"]}
  termination_date: {canonical: "termination_date", examples: ["termination_date"]}
"""


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset(
    employees: int = DEFAULT_EMPLOYEES,
    year: int = DEFAULT_YEAR,
    seed: int = DEFAULT_SEED,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    periods: int = 26,
    prefix: str | None = None,
) -> dict[str, Path]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or f"realistic_{employees}"
    paths = {
        "payroll": output_dir / f"{prefix}_payroll.csv",
        "recordkeeper": output_dir / f"{prefix}_recordkeeper.csv",
        "ground_truth": output_dir / f"{prefix}_ground_truth.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
        "mapping": output_dir / f"{prefix}_mapping.yaml",
    }
    ids = [f"E{100001 + i}" for i in range(employees)]
    scenarios = scenario_sets(ids)
    employees_data = build_employees(employees, year, rng, scenarios)
    pay_periods = pay_periods_for_year(year, periods)
    payroll_rows: list[dict[str, str]] = []
    rk_rows: list[dict[str, str]] = []
    truth_rows: list[dict[str, str]] = []
    annual_deferrals = {emp.employee_id: 0.0 for emp in employees_data}

    for emp in employees_data:
        for period, pay_date in enumerate(pay_periods, start=1):
            regular, overtime, bonus, fringe, reimbursement = compensation(emp, period, pay_date, rng, scenarios)
            gross = regular + overtime + bonus + fringe + reimbursement
            plan_comp = regular + overtime + bonus
            base = roth = catchup_pretax = catchup_roth = loan = 0.0
            if gross > 0:
                total_deferral = plan_comp * emp.deferral_rate
                if emp.employee_id in scenarios["excess_402g"]:
                    total_deferral = min(plan_comp * .18, 1000)
                base = total_deferral * (1 - emp.roth_share)
                roth = total_deferral * emp.roth_share
                if emp.employee_id in scenarios["secure20_hce_pretax"] and period == min(22, periods):
                    catchup_pretax = 850
                loan = emp.loan_payment if emp.has_loan else 0

            er_match = min(base + roth, plan_comp * .06) * .50
            if emp.employee_id in scenarios["comp_match_under"] and period == min(9, periods):
                er_match = max(0, er_match - 125)
                truth(truth_rows, scenario_id="COMP_MATCH_UNDER", employee_id=emp.employee_id, exception_type="Under-match from compensation definition variance", issue_category="Compensation/Match", engine_area="compensation_match_auditor", source_file="payroll", pay_period_number=period, pay_date=pay_date, expected_detection="compensation_match_issues.csv when plan_match_config is supplied", future_only=False, description="Employer match deliberately below the configured 50% up to 6% formula.", planted_payroll_value=money(er_match), notes="Current engine-supported when compensation match plan rules are supplied.")
            if emp.employee_id in scenarios["comp_match_over"] and period == min(12, periods):
                er_match += 175
                truth(truth_rows, scenario_id="COMP_MATCH_OVER", employee_id=emp.employee_id, exception_type="Over-match from excluded compensation included", issue_category="Compensation/Match", engine_area="compensation_match_auditor", source_file="payroll", pay_period_number=period, pay_date=pay_date, expected_detection="compensation_match_issues.csv when plan_match_config is supplied", future_only=False, description="Employer match deliberately above expected formula amount.", planted_payroll_value=money(er_match), notes="Current engine-supported when compensation match plan rules are supplied.")
            if emp.employee_id in scenarios["excluded_class_match"] and period == min(8, periods):
                er_match = max(er_match, 90)
                truth(truth_rows, scenario_id="EXCLUDED_CLASS_MATCH", employee_id=emp.employee_id, exception_type="Match paid to excluded employee class", issue_category="Compensation/Match", engine_area="compensation_match_auditor", source_file="payroll", pay_period_number=period, pay_date=pay_date, expected_detection="compensation_match_issues.csv when plan_match_config is supplied", future_only=False, description="Excluded employee class deliberately received employer match.", planted_payroll_value=money(er_match), notes="Current engine-supported when compensation match plan rules are supplied.")

            current_status = status_for(emp, pay_date)
            age = year - emp.dob.year - ((pay_date.month, pay_date.day) < (emp.dob.month, emp.dob.day))
            payroll_rows.append({
                "employee_id": emp.employee_id, "first_name": emp.first_name, "last_name": emp.last_name,
                "pay_date": pay_date.isoformat(), "pay_period_number": str(period), "Employee Class": emp.employee_class,
                "department": emp.department, "location": emp.location, "Regular Compensation": money(regular),
                "Overtime": money(overtime), "Bonus": money(bonus), "Fringe": money(fringe),
                "Reimbursement": money(reimbursement), "Gross Compensation": money(gross),
                "Compensation": money(gross), "Plan Compensation": money(plan_comp),
                "EE Deferral $": money(base), "EE Roth $": money(roth), "ER Match $": money(er_match),
                "True-Up Enabled": "true" if emp.true_up_enabled else "false", "loan_amount": money(loan),
                "is_hce": "true" if emp.is_hce else "false", "catchup_pretax": money(catchup_pretax),
                "catchup_roth": money(catchup_roth), "age": str(age), "dob": emp.dob.isoformat(),
                "hire_date": emp.hire_date.isoformat(), "rehire_date": iso(emp.rehire_date),
                "employment_status": current_status, "termination_date": iso(emp.termination_date if current_status == "Terminated" else None),
            })
            annual_deferrals[emp.employee_id] += base + roth + catchup_pretax + catchup_roth
            if emp.employee_id in scenarios["secure20_hce_pretax"] and period == min(22, periods):
                truth(truth_rows, scenario_id="SECURE20_HCE_PRETAX_CATCHUP", employee_id=emp.employee_id, exception_type="HCE_PRETAX_CATCHUP_NOT_ROTH", issue_category="Secure 2.0", engine_area="secure20_catchup_analyzer", source_file="payroll", pay_period_number=period, pay_date=pay_date, expected_detection="secure20_exceptions.csv", future_only=False, description="HCE age 50+ catch-up deliberately coded pretax instead of Roth.", planted_payroll_value=money(catchup_pretax), notes="Current engine-supported Secure 2.0 catch-up scenario.")

            if emp.employee_id in scenarios["only_in_payroll"]:
                if period == 1:
                    truth(truth_rows, scenario_id="ONLY_IN_PAYROLL", employee_id=emp.employee_id, exception_type="ONLY_IN_PAYROLL", issue_category="Core Reconciliation", engine_area="reconcile_stream", source_file="payroll", expected_detection="only_in_payroll_deferrals.csv and only_in_payroll_loans.csv", future_only=False, description="Participant deliberately omitted from the recordkeeper file.", notes="Current ProofLink payroll-only reconciliation behavior.")
                continue

            rk_base, rk_roth, rk_loan = base, roth, loan
            deposit_date = business_day_add(pay_date, 2)
            rk_status = "active"
            rk_term: date | None = None
            if emp.employee_id in scenarios["status_conflict"]:
                rk_status = "terminated"
                if period == 1:
                    truth(truth_rows, scenario_id="EMPLOYMENT_STATUS_CONFLICT", employee_id=emp.employee_id, exception_type="EMPLOYMENT_STATUS_CONFLICT", issue_category="Population Validation", engine_area="population_validation_analyzer", source_file="payroll,recordkeeper", expected_detection="population_validation_issues.csv", future_only=False, description="Payroll active status conflicts with recordkeeper terminated status.", planted_payroll_value="Active", planted_recordkeeper_value="terminated")
            if emp.employee_id in scenarios["post_term_comp"]:
                rk_status = ""
                rk_term = emp.termination_date
                if period == min(14, periods):
                    truth(truth_rows, scenario_id="POST_TERMINATION_COMPENSATION", employee_id=emp.employee_id, exception_type="POST_TERMINATION_COMPENSATION", issue_category="Population Validation", engine_area="population_validation_analyzer", source_file="payroll,recordkeeper", pay_period_number=period, pay_date=pay_date, expected_detection="population_validation_issues.csv", future_only=False, description="Positive compensation deliberately appears more than 30 days after RK termination date.", planted_payroll_value=money(gross), planted_recordkeeper_value=iso(rk_term))
            if emp.employee_id in scenarios["deferral_mismatch"] and period == min(7, periods):
                rk_base += 37.50
                truth(truth_rows, scenario_id="DEFERRAL_MISMATCH", employee_id=emp.employee_id, exception_type="DEFERRAL_MISMATCH", issue_category="Core Reconciliation", engine_area="reconcile_stream", source_file="recordkeeper", pay_period_number=period, pay_date=pay_date, expected_detection="deferral_mismatches.csv", future_only=False, description="Recordkeeper pretax contribution deliberately increased for one period.", planted_payroll_value=money(base + roth), planted_recordkeeper_value=money(rk_base + rk_roth))
            if emp.employee_id in scenarios["loan_mismatch"] and period == min(10, periods):
                rk_loan = max(0, rk_loan - 22)
                truth(truth_rows, scenario_id="LOAN_MISMATCH", employee_id=emp.employee_id, exception_type="LOAN_MISMATCH", issue_category="Core Reconciliation", engine_area="reconcile_stream", source_file="recordkeeper", pay_period_number=period, pay_date=pay_date, expected_detection="loan_mismatches.csv", future_only=False, description="Recordkeeper loan repayment deliberately lower than payroll for one period.", planted_payroll_value=money(loan), planted_recordkeeper_value=money(rk_loan))
            if emp.employee_id in scenarios["late_contribution"] and period == 1:
                deposit_date = business_day_add(pay_date, 9)
                truth(truth_rows, scenario_id="LATE_CONTRIBUTION", employee_id=emp.employee_id, exception_type="LATE_CONTRIBUTION", issue_category="Contribution Timing", engine_area="contribution_timing_analyzer_v2", source_file="recordkeeper", pay_period_number=period, pay_date=pay_date, expected_detection="late_contributions.csv / late_deferrals_contributions.csv", future_only=False, description="Recordkeeper deposit deliberately delayed beyond timing threshold.", planted_payroll_value=pay_date.isoformat(), planted_recordkeeper_value=deposit_date.isoformat())
            rk_rows.append({
                "employee_id": emp.employee_id, "pay_date": pay_date.isoformat(), "deposit_date": deposit_date.isoformat(),
                "EE Deferral $": money(rk_base), "EE Roth $": money(rk_roth), "loan_amount": money(rk_loan),
                "recordkeeper_employment_status": rk_status, "termination_date": iso(rk_term),
                "source_system": "Synthetic RK Trust", "transaction_type": "CONTRIBUTION",
            })

    for emp in employees_data:
        if emp.employee_id in scenarios["excess_402g"]:
            truth(truth_rows, scenario_id="EXCESS_402G", employee_id=emp.employee_id, exception_type="EXCESS_402G_DEFERRAL", issue_category="Comp/402(g)", engine_area="comp_402g_analyzer", source_file="payroll", expected_detection="comp_402g_violations.csv", future_only=False, description="Annual employee deferrals deliberately exceed the 2025 402(g) limit.", planted_payroll_value=money(annual_deferrals[emp.employee_id]), notes="Current engine-supported annual limit scenario.")

    for i in range(min(3, max(1, employees // 150))):
        phantom_id = f"RKONLY{year}{i + 1:02d}"
        pay_date = pay_periods[min(5 + i, len(pay_periods) - 1)]
        rk_rows.append({"employee_id": phantom_id, "pay_date": pay_date.isoformat(), "deposit_date": business_day_add(pay_date, 2).isoformat(), "EE Deferral $": money(250 + 25 * i), "EE Roth $": "0.00", "loan_amount": "0.00", "recordkeeper_employment_status": "active", "termination_date": "", "source_system": "Synthetic RK Trust", "transaction_type": "CONTRIBUTION"})
        truth(truth_rows, scenario_id="ONLY_IN_RECORDKEEPER", employee_id=phantom_id, exception_type="ONLY_IN_RECORDKEEPER", issue_category="Core Reconciliation", engine_area="reconcile_stream", source_file="recordkeeper", pay_period_number=min(6 + i, len(pay_periods)), pay_date=pay_date, expected_detection="only_in_recordkeeper_deferrals.csv and only_in_recordkeeper_loans.csv", future_only=False, description="Recordkeeper-only participant deliberately inserted without payroll rows.", planted_recordkeeper_value=money(250 + 25 * i), notes="Current ProofLink recordkeeper-only reconciliation behavior.")

    payroll_rows.sort(key=lambda row: (row["employee_id"], int(row["pay_period_number"])))
    rk_rows.sort(key=lambda row: (row["employee_id"], row["pay_date"]))
    truth_rows.sort(key=lambda row: (row["scenario_id"], row["employee_id"], row["pay_period_number"], row["pay_date"]))

    write_csv(paths["payroll"], PAYROLL_COLUMNS, payroll_rows)
    write_csv(paths["recordkeeper"], RK_COLUMNS, rk_rows)
    write_csv(paths["ground_truth"], GROUND_TRUTH_COLUMNS, truth_rows)
    paths["mapping"].write_text(mapping_yaml(), encoding="utf-8")
    metadata = {
        "dataset_name": f"ProofLink realistic {employees}-employee synthetic demo",
        "seed": seed,
        "year": year,
        "employee_count": employees,
        "payroll_period_count": len(pay_periods),
        "payroll_row_count": len(payroll_rows),
        "recordkeeper_row_count": len(rk_rows),
        "ground_truth_row_count": len(truth_rows),
        "pay_period_start": pay_periods[0].isoformat(),
        "pay_period_end": pay_periods[-1].isoformat(),
        "generated_at_utc": "2025-06-30T00:00:00Z",
        "scenario_counts": {},
        "schema_notes": {
            "payroll": "Canonical ProofLink columns plus demographics, compensation components, status, hire, termination, and rehire fields.",
            "recordkeeper": "Canonical contribution columns plus recordkeeper status and termination date fields used by population validation.",
            "ground_truth": "Documents deliberately planted exceptions. future_only=false means the current engine has a relevant analyzer or output surface.",
        },
    }
    for row in truth_rows:
        metadata["scenario_counts"][row["scenario_id"]] = metadata["scenario_counts"].get(row["scenario_id"], 0) + 1
    metadata["files"] = {key: {"path": path.name, "sha256": sha256(path)} for key, path in paths.items() if key != "metadata"}
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the deterministic realistic ProofLink demo dataset.")
    parser.add_argument("--employees", type=int, default=DEFAULT_EMPLOYEES)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--periods", type=int, default=26)
    parser.add_argument("--prefix", default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.employees < 1:
        raise SystemExit("--employees must be positive")
    if args.periods < 1:
        raise SystemExit("--periods must be positive")
    paths = generate_dataset(args.employees, args.year, args.seed, args.output_dir, args.periods, args.prefix)
    print("Generated realistic ProofLink demo dataset:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
