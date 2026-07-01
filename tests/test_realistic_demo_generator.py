from pathlib import Path
import json
import os
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import EngineConfig, run_prooflink_engine, run_reconciliation  # noqa: E402
from preflight import run_preflight  # noqa: E402
from scripts.generate_realistic_demo import generate_dataset, sha256  # noqa: E402
from verify_proof import verify_evidence_pack_zip  # noqa: E402


DEMO_DIR = ROOT_DIR / "data" / "demo"
REQUIRED_SCENARIOS = {
    "DEFERRAL_MISMATCH",
    "LOAN_MISMATCH",
    "LATE_CONTRIBUTION",
    "ONLY_IN_PAYROLL",
    "ONLY_IN_RECORDKEEPER",
    "EMPLOYMENT_STATUS_CONFLICT",
    "POST_TERMINATION_COMPENSATION",
    "EXCESS_402G",
    "SECURE20_HCE_PRETAX_CATCHUP",
    "COMP_MATCH_UNDER",
    "COMP_MATCH_OVER",
    "EXCLUDED_CLASS_MATCH",
}

POLISHED_SCENARIO_COUNTS = {
    "DEFERRAL_MISMATCH": 2,
    "LOAN_MISMATCH": 1,
    "ONLY_IN_PAYROLL": 1,
    "ONLY_IN_RECORDKEEPER": 1,
    "LATE_CONTRIBUTION": 2,
    "EMPLOYMENT_STATUS_CONFLICT": 1,
    "POST_TERMINATION_COMPENSATION": 1,
    "SECURE20_HCE_PRETAX_CATCHUP": 1,
    "COMP_MATCH_UNDER": 1,
    "COMP_MATCH_OVER": 1,
}


def polished_plan_rules() -> dict:
    return {
        "eligibility_rule": "age21_and_1year",
        "service_days_required": 365,
        "age_required": 21,
        "align_first_month": False,
        "plan_match_config": {
            "match_formula_name": "50% up to 6% current-engine demo calibration",
            "match_type": "percent_of_comp",
            "match_rate": 0.50,
            "match_cap_pct": 0.06,
            "match_frequency": "per_payroll",
            "true_up_enabled": False,
            "true_up_enabled_column": "True-Up Enabled",
            "eligible_comp_columns": ["Plan Compensation"],
            "excluded_comp_columns": ["Fringe", "Reimbursement"],
            "employee_class_column": "Employee Class",
            "eligible_classes": ["Full-Time", "Part-Time"],
            "excluded_classes": ["Intern", "Union Excluded"],
            "absolute_tolerance": 5.00,
            "relative_tolerance_pct": 0.15,
        },
    }


def test_realistic_500_metadata_mapping_and_ground_truth_are_tracked_source_assets(tmp_path: Path):
    ground_truth_path = DEMO_DIR / "realistic_500_ground_truth.csv"
    metadata_path = DEMO_DIR / "realistic_500_metadata.json"
    mapping_path = DEMO_DIR / "realistic_500_mapping.yaml"

    for path in [ground_truth_path, metadata_path, mapping_path]:
        assert path.exists(), path

    ground_truth = pd.read_csv(ground_truth_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert REQUIRED_SCENARIOS <= set(ground_truth["scenario_id"])
    assert set(ground_truth["future_only"].astype(str).str.lower()) == {"false"}
    assert metadata["seed"] == 20250630
    assert metadata["year"] == 2025
    assert metadata["employee_count"] == 500
    assert metadata["payroll_row_count"] == 500 * 26
    assert metadata["ground_truth_row_count"] == len(ground_truth)
    assert all(not Path(info["path"]).is_absolute() for info in metadata["files"].values())

    generated = generate_dataset(employees=500, year=2025, seed=20250630, output_dir=tmp_path / "generated")
    payroll = pd.read_csv(generated["payroll"])
    rk = pd.read_csv(generated["recordkeeper"])
    assert len(payroll) == metadata["payroll_row_count"]
    assert payroll["employee_id"].nunique() == 500
    assert payroll["pay_period_number"].nunique() == 26
    assert len(rk) == metadata["recordkeeper_row_count"]
    assert sha256(generated["payroll"]) == metadata["files"]["payroll"]["sha256"]
    assert sha256(generated["recordkeeper"]) == metadata["files"]["recordkeeper"]["sha256"]

    safe, report = run_preflight(str(generated["payroll"]), str(generated["recordkeeper"]), str(mapping_path))
    assert safe, report


def test_checked_in_smoke_fixture_is_small_and_covers_required_scenarios():
    smoke_dir = DEMO_DIR / "smoke"
    payroll_path = smoke_dir / "realistic_smoke_25_payroll.csv"
    rk_path = smoke_dir / "realistic_smoke_25_recordkeeper.csv"
    ground_truth_path = smoke_dir / "realistic_smoke_25_ground_truth.csv"
    mapping_path = smoke_dir / "realistic_smoke_25_mapping.yaml"

    payroll = pd.read_csv(payroll_path)
    rk = pd.read_csv(rk_path)
    ground_truth = pd.read_csv(ground_truth_path)

    assert len(payroll) == 25 * 4
    assert payroll["employee_id"].nunique() == 25
    assert payroll["pay_period_number"].nunique() == 4
    assert len(rk) < 120
    assert {
        "DEFERRAL_MISMATCH",
        "LATE_CONTRIBUTION",
        "EMPLOYMENT_STATUS_CONFLICT",
        "POST_TERMINATION_COMPENSATION",
    } <= set(ground_truth["scenario_id"])

    safe, report = run_preflight(str(payroll_path), str(rk_path), str(mapping_path))
    assert safe, report


def test_realistic_generator_is_deterministic_across_output_directories(tmp_path: Path):
    first = generate_dataset(employees=60, year=2025, seed=20250630, output_dir=tmp_path / "first")
    second = generate_dataset(employees=60, year=2025, seed=20250630, output_dir=tmp_path / "second")

    for key in ["payroll", "recordkeeper", "ground_truth", "metadata", "mapping"]:
        assert sha256(first[key]) == sha256(second[key])

    payroll = pd.read_csv(first["payroll"])
    assert len(payroll) == 60 * 26
    assert payroll["employee_id"].nunique() == 60

    smoke_first = generate_dataset(employees=25, year=2025, seed=20250630, output_dir=tmp_path / "smoke_first", periods=4, prefix="realistic_smoke_25")
    smoke_second = generate_dataset(employees=25, year=2025, seed=20250630, output_dir=tmp_path / "smoke_second", periods=4, prefix="realistic_smoke_25")
    for key in ["payroll", "recordkeeper", "ground_truth", "metadata", "mapping"]:
        assert sha256(smoke_first[key]) == sha256(smoke_second[key])


def test_polished_profile_generation_has_exact_shape_and_ground_truth(tmp_path: Path):
    paths = generate_dataset(
        employees=100,
        year=2025,
        seed=20250630,
        output_dir=tmp_path / "polished",
        periods=8,
        prefix="prooflink_polished_100",
        profile="polished",
    )

    payroll = pd.read_csv(paths["payroll"])
    rk = pd.read_csv(paths["recordkeeper"])
    ground_truth = pd.read_csv(paths["ground_truth"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    assert len(payroll) == 100 * 8
    assert payroll["employee_id"].nunique() == 100
    assert payroll["pay_period_number"].nunique() == 8
    assert len(rk) == 793
    assert len(ground_truth) == 12
    assert ground_truth["scenario_id"].value_counts().to_dict() == POLISHED_SCENARIO_COUNTS
    assert metadata["profile"] == "polished"
    assert metadata["scenario_counts"] == POLISHED_SCENARIO_COUNTS

    required_columns = {
        "scenario_id",
        "employee_id",
        "expected_issue_type",
        "issue_category",
        "payroll_date",
        "deposit_date",
        "expected_payroll_amount",
        "expected_recordkeeper_amount",
        "planted_change",
        "expected_detection",
        "demo_explanation",
        "correction_action",
        "notes",
    }
    assert required_columns <= set(ground_truth.columns)

    safe, report = run_preflight(str(paths["payroll"]), str(paths["recordkeeper"]), str(paths["mapping"]))
    assert safe, report


def test_polished_profile_is_deterministic(tmp_path: Path):
    first = generate_dataset(
        employees=100,
        year=2025,
        seed=20250630,
        output_dir=tmp_path / "first",
        periods=8,
        prefix="prooflink_polished_100",
        profile="polished",
    )
    second = generate_dataset(
        employees=100,
        year=2025,
        seed=20250630,
        output_dir=tmp_path / "second",
        periods=8,
        prefix="prooflink_polished_100",
        profile="polished",
    )

    for key in ["payroll", "recordkeeper", "ground_truth", "metadata", "mapping"]:
        assert sha256(first[key]) == sha256(second[key])


def test_polished_profile_engine_run_exact_core_counts_and_verified_pack(tmp_path: Path, monkeypatch):
    paths = generate_dataset(
        employees=100,
        year=2025,
        seed=20250630,
        output_dir=tmp_path / "input",
        periods=8,
        prefix="prooflink_polished_100",
        profile="polished",
    )
    monkeypatch.setenv("MAPPING_YAML_PATH", str(paths["mapping"]))

    result = run_prooflink_engine(
        payroll_path=str(paths["payroll"]),
        rk_path=str(paths["recordkeeper"]),
        config=EngineConfig(
            plan_name="ProofLink Polished Demo Plan",
            payroll_vendor_hint="ADP",
            rk_vendor_hint="VENDOR_RK_1",
            output_dir=str(tmp_path / "output"),
            proofs_dir=str(tmp_path / "proofs"),
        ),
        run_id="polished-demo-test",
        plan_rules=polished_plan_rules(),
    )

    output_dir = tmp_path / "output"
    assert Path(result.evidence_pack_path).exists()
    assert verify_evidence_pack_zip(Path(result.evidence_pack_path))["overall_ok"] is True

    assert len(pd.read_csv(output_dir / "deferral_mismatches.csv")) == 2
    assert len(pd.read_csv(output_dir / "loan_mismatches.csv")) == 1
    assert len(pd.read_csv(output_dir / "only_in_payroll_deferrals.csv")) == 1
    assert len(pd.read_csv(output_dir / "only_in_recordkeeper_deferrals.csv")) == 1

    late = pd.read_csv(output_dir / "late_contributions.csv")
    assert result.summary["timing_metrics"]["total_rows"] == 801
    assert int(late["is_late"].sum()) == 2
    assert int(late["missing_deposit"].sum()) == 8
    assert int(late["unmatched_recordkeeper"].sum()) == 1
    assert set(late["timing_match_rule"]) == {"employee_id+pay_date"}

    population = pd.read_csv(output_dir / "population_validation_issues.csv")
    assert population["exception_type"].value_counts().to_dict() == {
        "EMPLOYMENT_STATUS_CONFLICT": 1,
        "POST_TERMINATION_COMPENSATION": 1,
    }
    assert result.summary["population_validation"]["skipped_rules"] == {}

    assert result.summary["secure20"]["total_violations"] == 1
    assert result.summary["secure20"]["hce_violation_count"] == 1
    assert result.summary["secure20"]["potential_catchup_miscode_count"] == 0
    assert result.summary["comp_402g"]["excess_violation_count"] == 0
    assert result.summary["eligibility_drift"]["eligibility_drift_count"] == 0

    comp_match = result.summary["compensation_match"]
    assert comp_match["issue_count"] == 2
    assert comp_match["under_match_count"] == 1
    assert comp_match["over_match_count"] == 1
    assert comp_match["excluded_class_match_count"] == 0


def test_generated_realistic_dataset_runs_through_reconciliation_engine(tmp_path: Path):
    paths = generate_dataset(employees=160, year=2025, seed=20250630, output_dir=tmp_path / "input")
    results = run_reconciliation(
        payroll_csv=str(paths["payroll"]),
        rk_csv=str(paths["recordkeeper"]),
        payroll_vendor_hint="ADP",
        rk_vendor_hint="VENDOR_RK_1",
        output_dir=str(tmp_path / "output"),
        proofs_dir=str(tmp_path / "proofs"),
        mapping_yaml_path=str(paths["mapping"]),
        run_id="realistic-demo-generator-test",
        plan_name="Realistic Demo Plan",
    )

    assert Path(results["evidence_pack"]).exists()
    assert Path(results["manifest"]).exists()
    assert pd.read_csv(results["deferral_mismatches"]).shape[0] >= 1
    assert pd.read_csv(results["loan_mismatches"]).shape[0] >= 1
    assert pd.read_csv(results["only_in_payroll"]).shape[0] >= 1
    assert pd.read_csv(results["only_in_recordkeeper"]).shape[0] >= 1
    assert results["population_validation_summary"]["issue_count"] >= 1
