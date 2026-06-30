from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import verify_proof


def _write_manifest(path: Path, manifest: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _manifest_for_outputs(outputs: dict) -> dict:
    return {
        "run_timestamp_utc": "2026-06-30T15:00:00+00:00",
        "payroll_file": "inputs/payroll.csv",
        "recordkeeper_file": "inputs/rk.csv",
        "config_name": "test",
        "outputs": outputs,
    }


def _zip_with_manifest(zip_path: Path, manifest_name: str, manifest: dict, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(manifest_name, json.dumps(manifest, indent=2))
        for name, data in members.items():
            zf.writestr(name, data)
    return zip_path


def test_positional_manifest_cli_argument_is_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    default_proofs = tmp_path / "default_proofs"
    default_proofs.mkdir()
    bad_output = tmp_path / "missing.csv"
    bad_manifest = _manifest_for_outputs(
        {"deferral_mismatches": {"path": str(bad_output), "sha256": "not-the-hash"}}
    )
    _write_manifest(default_proofs / "proof_manifest_99999999T999999+0000.json", bad_manifest)
    monkeypatch.setattr(verify_proof, "PROOFS_DIR", default_proofs)

    supplied_manifest = _write_manifest(
        tmp_path / "supplied" / "proof_manifest_20260630T150000+0000.json",
        _manifest_for_outputs({}),
    )

    assert verify_proof.main([str(supplied_manifest)]) == 0


def test_repository_relative_manifest_paths_resolve_from_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(verify_proof, "REPO_ROOT", tmp_path)
    output = tmp_path / "data" / "demo" / "population_output" / "deferral_mismatches.csv"
    output.parent.mkdir(parents=True)
    output.write_text("employee_id,amount\n1001,25\n", encoding="utf-8")

    manifest = _write_manifest(
        tmp_path / "data" / "demo" / "population_proofs" / "proof_manifest_20260630T150001+0000.json",
        _manifest_for_outputs(
            {
                "deferral_mismatches": {
                    "path": "data/demo/population_output/deferral_mismatches.csv",
                    "sha256": verify_proof.sha256_file(output),
                }
            }
        ),
    )

    result = verify_proof.verify_manifest(manifest)
    assert result["output_verification"] == "PASS"
    assert result["overall_verification"] == "PASS"


def test_same_named_files_do_not_resolve_from_unrelated_output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(verify_proof, "REPO_ROOT", tmp_path)
    correct = tmp_path / "data" / "demo" / "population_output" / "reconciliation_report.xlsx"
    wrong = tmp_path / "data" / "demo" / "other_output" / "reconciliation_report.xlsx"
    correct.parent.mkdir(parents=True)
    wrong.parent.mkdir(parents=True)
    correct.write_bytes(b"correct workbook bytes")
    wrong.write_bytes(b"wrong workbook bytes")

    manifest = _write_manifest(
        tmp_path / "data" / "demo" / "population_proofs" / "proof_manifest_20260630T150002+0000.json",
        _manifest_for_outputs(
            {
                "excel_report": {
                    "path": "data/demo/population_output/reconciliation_report.xlsx",
                    "sha256": verify_proof.sha256_file(correct),
                }
            }
        ),
    )

    result = verify_proof.verify_manifest(manifest)
    assert result["output_verification"] == "PASS"


def test_zip_contained_outputs_verify_directly_without_extraction(tmp_path: Path):
    output_bytes = b"employee_id,exception_type\n1001,EMPLOYMENT_STATUS_CONFLICT\n"
    manifest = _manifest_for_outputs(
        {
            "population_validation_issues": {
                "path": "data/demo/population_output/population_validation_issues.csv",
                "sha256": verify_proof.sha256_bytes(output_bytes),
            }
        }
    )
    zip_path = _zip_with_manifest(
        tmp_path / "prooflink_evidence_pack.zip",
        "proof_manifest_20260630T150003+0000.json",
        manifest,
        {"population_validation_issues.csv": output_bytes},
    )

    result = verify_proof.verify_evidence_pack_zip(zip_path)
    assert result["input_verification"] == "NOT AVAILABLE"
    assert result["output_verification"] == "PASS"
    assert result["overall_verification"] == "PASS"


def test_modified_zip_output_fails_verification(tmp_path: Path):
    original_bytes = b"employee_id,amount\n1001,25\n"
    modified_bytes = b"employee_id,amount\n1001,999\n"
    manifest = _manifest_for_outputs(
        {
            "deferral_mismatches": {
                "path": "data/demo/population_output/deferral_mismatches.csv",
                "sha256": verify_proof.sha256_bytes(original_bytes),
            }
        }
    )
    zip_path = _zip_with_manifest(
        tmp_path / "prooflink_evidence_pack.zip",
        "proof_manifest_20260630T150004+0000.json",
        manifest,
        {"deferral_mismatches.csv": modified_bytes},
    )

    result = verify_proof.verify_evidence_pack_zip(zip_path)
    assert result["output_verification"] == "FAIL"
    assert result["overall_verification"] == "FAIL"


def test_missing_raw_inputs_are_not_available_not_output_failure(tmp_path: Path):
    output_bytes = b"employee_id,amount\n1001,25\n"
    manifest = _manifest_for_outputs(
        {
            "deferral_mismatches": {
                "path": "deferral_mismatches.csv",
                "sha256": verify_proof.sha256_bytes(output_bytes),
            }
        }
    )
    zip_path = _zip_with_manifest(
        tmp_path / "prooflink_evidence_pack.zip",
        "proof_manifest_20260630T150005+0000.json",
        manifest,
        {"deferral_mismatches.csv": output_bytes},
    )

    result = verify_proof.verify_evidence_pack_zip(zip_path)
    assert result["input_verification"] == "NOT AVAILABLE"
    assert result["output_verification"] == "PASS"
    assert result["overall_verification"] == "PASS"


def test_supplied_input_files_receive_hash_verification(tmp_path: Path):
    payroll = tmp_path / "payroll.csv"
    rk = tmp_path / "rk.csv"
    payroll.write_text("employee_id,amount\n1001,25\n", encoding="utf-8")
    rk.write_text("employee_id,amount\n1001,25\n", encoding="utf-8")
    output_bytes = b"employee_id,amount\n1001,25\n"
    manifest = _manifest_for_outputs(
        {
            "deferral_mismatches": {
                "path": "deferral_mismatches.csv",
                "sha256": verify_proof.sha256_bytes(output_bytes),
            }
        }
    )
    zip_path = _zip_with_manifest(
        tmp_path / "prooflink_evidence_pack.zip",
        "proof_manifest_20260630T150006+0000.json",
        manifest,
        {"deferral_mismatches.csv": output_bytes},
    )

    result = verify_proof.verify_evidence_pack_zip(zip_path, payroll_path=payroll, recordkeeper_path=rk)
    assert result["input_verification"] == "PASS"
    assert result["output_verification"] == "PASS"
    assert result["overall_verification"] == "PASS"


def test_old_manifest_without_output_path_uses_known_key_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    output = output_dir / "deferral_mismatches.csv"
    output.write_text("employee_id,amount\n1001,25\n", encoding="utf-8")
    monkeypatch.setattr(verify_proof, "DATA_OUT", output_dir)

    manifest = _write_manifest(
        tmp_path / "proofs" / "proof_manifest_20260630T150007+0000.json",
        _manifest_for_outputs(
            {
                "deferral_mismatches": {
                    "sha256": verify_proof.sha256_file(output),
                }
            }
        ),
    )

    result = verify_proof.verify_manifest(manifest)
    assert result["output_verification"] == "PASS"
    assert result["overall_verification"] == "PASS"


def test_invalid_cli_path_returns_clear_nonzero_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    missing = tmp_path / "missing_manifest.json"

    assert verify_proof.main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_extracted_evidence_pack_directory_verifies_outputs(tmp_path: Path):
    evidence_dir = tmp_path / "extracted_pack"
    evidence_dir.mkdir()
    output = evidence_dir / "population_validation_issues.csv"
    output.write_text("employee_id,exception_type\n1001,EMPLOYMENT_STATUS_CONFLICT\n", encoding="utf-8")
    manifest = _manifest_for_outputs(
        {
            "population_validation_issues": {
                "path": "data/demo/population_output/population_validation_issues.csv",
                "sha256": verify_proof.sha256_file(output),
            }
        }
    )
    _write_manifest(evidence_dir / "proof_manifest_20260630T150008+0000.json", manifest)

    result = verify_proof.verify_evidence_pack_dir(evidence_dir)
    assert result["input_verification"] == "NOT AVAILABLE"
    assert result["output_verification"] == "PASS"
    assert result["overall_verification"] == "PASS"
