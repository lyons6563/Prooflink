from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_OUT = REPO_ROOT / "data" / "processed"
PROOFS_DIR = REPO_ROOT / "proofs"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_AVAILABLE = "NOT AVAILABLE"


KNOWN_OUTPUTS = {
    "deferral_mismatches": "deferral_mismatches.csv",
    "loan_mismatches": "loan_mismatches.csv",
    "late_deferrals": "late_deferrals_contributions.csv",
    "late_loans": "late_loans_contributions.csv",
    "excel_report": "reconciliation_report.xlsx",
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of an entire file (binary)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_latest_manifest() -> Path | None:
    """Return the most recent proof_manifest_*.json file, or None if none exist."""
    if not PROOFS_DIR.exists():
        return None
    manifests = sorted(PROOFS_DIR.glob("proof_manifest_*.json"))
    if not manifests:
        return None
    return manifests[-1]


def _manifest_result(
    *,
    input_status: str,
    output_status: str,
    warnings: list[str] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    warnings = warnings or []
    overall_status = STATUS_PASS if output_status == STATUS_PASS and input_status != STATUS_FAIL else STATUS_FAIL
    return {
        "input_verification": input_status,
        "output_verification": output_status,
        "overall_verification": overall_status,
        "overall_ok": overall_status == STATUS_PASS,
        "warnings": warnings,
        "manifest_path": str(manifest_path) if manifest_path else None,
    }


def _candidate_run_artifact_dirs(manifest_path: Path, evidence_root: Path | None = None) -> list[Path]:
    roots: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved not in roots:
            roots.append(resolved)

    if evidence_root is not None:
        add(evidence_root)

    add(manifest_path.parent)
    run_parent = manifest_path.parent.parent
    proofs_name = manifest_path.parent.name
    if proofs_name.endswith("_proofs"):
        add(run_parent / f"{proofs_name[:-len('_proofs')]}_output")
    add(run_parent / "output")
    add(run_parent / "population_output")
    add(DATA_OUT)
    return roots


def _resolve_manifest_output_path(
    key: str,
    info: dict,
    manifest_path: Path,
    evidence_root: Path | None = None,
    prefer_evidence_root: bool = False,
) -> Path | None:
    path_value = info.get("path")
    if not path_value:
        known_name = KNOWN_OUTPUTS.get(key)
        if known_name:
            return _resolve_manifest_output_path(
                key,
                {"path": known_name},
                manifest_path,
                evidence_root=evidence_root,
                prefer_evidence_root=prefer_evidence_root,
            )
        return None

    stored = Path(str(path_value))
    basename = stored.name

    # 1. Exact absolute path, when it still exists.
    if stored.is_absolute() and stored.exists():
        return stored

    if prefer_evidence_root and evidence_root is not None:
        evidence_candidates = [evidence_root / stored, evidence_root / basename]
        for candidate in evidence_candidates:
            if candidate.exists():
                return candidate

    # 2. Exact repository-relative path from the repository root.
    if not stored.is_absolute():
        repo_candidate = REPO_ROOT / stored
        if repo_candidate.exists():
            return repo_candidate

    # 3. File beside the manifest. Evidence packs are intentionally flat today.
    manifest_candidates = [manifest_path.parent / stored, manifest_path.parent / basename]
    for candidate in manifest_candidates:
        if candidate.exists():
            return candidate

    # 4. File beside an explicitly supplied evidence-pack/extraction directory.
    if evidence_root is not None:
        evidence_candidates = [evidence_root / stored, evidence_root / basename]
        for candidate in evidence_candidates:
            if candidate.exists():
                return candidate

    # 5. Deterministic search restricted to the current run's artifact directories.
    for root in _candidate_run_artifact_dirs(manifest_path, evidence_root=evidence_root):
        exact_candidates = []
        if not stored.is_absolute():
            exact_candidates.append(root / stored)
        exact_candidates.append(root / basename)
        for candidate in exact_candidates:
            if candidate.exists():
                return candidate

        if len(stored.parts) == 1 and root.exists():
            matches = sorted(path for path in root.rglob(basename) if path.is_file())
            if matches:
                return matches[0]

    # Return the most likely current-run location for a useful error message.
    for root in _candidate_run_artifact_dirs(manifest_path, evidence_root=evidence_root):
        if root.exists():
            return root / basename
    return manifest_path.parent / basename


def _resolve_input_path(
    path_value: str | None,
    manifest_path: Path,
    *,
    evidence_root: Path | None = None,
    supplied_path: Path | None = None,
) -> tuple[Path | None, bool]:
    if supplied_path is not None:
        return supplied_path, True
    if not path_value:
        return None, False

    stored = Path(str(path_value))
    basename = stored.name

    if stored.is_absolute() and stored.exists():
        return stored, False

    candidates: list[Path] = []
    if not stored.is_absolute():
        candidates.extend([REPO_ROOT / stored, manifest_path.parent / stored])
    candidates.append(manifest_path.parent / basename)
    if evidence_root is not None:
        if not stored.is_absolute():
            candidates.append(evidence_root / stored)
        candidates.append(evidence_root / basename)

    for candidate in candidates:
        if candidate.exists():
            return candidate, False
    return None, False


def _expected_input_hash(manifest: dict[str, Any], input_name: str) -> str | None:
    direct_keys = [
        f"{input_name}_sha256",
        f"{input_name}_file_sha256",
        f"{input_name}_hash",
    ]
    for key in direct_keys:
        value = manifest.get(key)
        if value:
            return str(value)

    input_hashes = manifest.get("input_hashes")
    if isinstance(input_hashes, dict):
        value = input_hashes.get(input_name) or input_hashes.get(f"{input_name}_file")
        if isinstance(value, dict):
            value = value.get("sha256")
        if value:
            return str(value)
    return None


def _verify_one_input(
    label: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    manifest_key: str,
    supplied_path: Path | None = None,
    evidence_root: Path | None = None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    path, supplied = _resolve_input_path(
        manifest.get(manifest_key),
        manifest_path,
        evidence_root=evidence_root,
        supplied_path=supplied_path,
    )

    if path is None or not path.exists():
        if supplied:
            print(f"\n[{label}] MISSING supplied input: {path}")
            return STATUS_FAIL, warnings
        message = f"{label} input file not available; source-input hash was not independently rechecked."
        print(f"\n[{label}] NOT AVAILABLE")
        print(f"  {message}")
        warnings.append(message)
        return STATUS_NOT_AVAILABLE, warnings

    actual_hash = sha256_file(path)
    expected_hash = _expected_input_hash(manifest, label.lower())
    print(f"\n[{label}] {path}")
    print(f"  SHA256: {actual_hash}")

    if expected_hash and expected_hash != actual_hash:
        print("  MISMATCH against manifest input hash.")
        print(f"    expected: {expected_hash}")
        print(f"    actual:   {actual_hash}")
        return STATUS_FAIL, warnings

    if not expected_hash:
        print("  Input hash computed; manifest does not store an input hash for comparison.")
    return STATUS_PASS, warnings


def _merge_input_status(statuses: list[str]) -> str:
    if STATUS_FAIL in statuses:
        return STATUS_FAIL
    if STATUS_NOT_AVAILABLE in statuses:
        return STATUS_NOT_AVAILABLE
    return STATUS_PASS


def _verify_inputs(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    evidence_root: Path | None = None,
    payroll_path: Path | None = None,
    recordkeeper_path: Path | None = None,
) -> tuple[str, list[str]]:
    statuses: list[str] = []
    warnings: list[str] = []
    for label, manifest_key, supplied in (
        ("PAYROLL", "payroll_file", payroll_path),
        ("RECORDKEEPER", "recordkeeper_file", recordkeeper_path),
    ):
        status, input_warnings = _verify_one_input(
            label,
            manifest,
            manifest_path,
            manifest_key=manifest_key,
            supplied_path=supplied,
            evidence_root=evidence_root,
        )
        statuses.append(status)
        warnings.extend(input_warnings)
    return _merge_input_status(statuses), warnings


def _verify_manifest_outputs(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    evidence_root: Path | None = None,
    prefer_evidence_root: bool = False,
) -> str:
    output_ok = True
    outputs = manifest.get("outputs", {})
    print("\n[OUTPUT FILES]")

    for key, info in outputs.items():
        if not isinstance(info, dict):
            print(f"  [{key}] ERROR: Manifest output entry is not an object.")
            output_ok = False
            continue

        path = _resolve_manifest_output_path(
            key,
            info,
            manifest_path,
            evidence_root=evidence_root,
            prefer_evidence_root=prefer_evidence_root,
        )
        if path is None:
            print(f"  [WARN] Unknown output key in manifest: {key}")
            continue

        print(f"  [{key}] {path}")

        if info.get("missing"):
            print("    Manifest says: missing=True (file did not exist at run time).")
            if path.exists():
                print("    Current state: file NOW exists (post-run change).")
            continue

        if not path.exists():
            print("    ERROR: File is missing on disk now.")
            output_ok = False
            continue

        expected_hash = info.get("sha256")
        actual_hash = sha256_file(path)

        if expected_hash != actual_hash:
            print("    MISMATCH!")
            print(f"      expected: {expected_hash}")
            print(f"      actual:   {actual_hash}")
            output_ok = False
        else:
            print("    OK (hash matches manifest).")

    return STATUS_PASS if output_ok else STATUS_FAIL


def _print_summary(result: dict[str, Any]) -> None:
    print("\n============================================")
    print(f"Input verification:  {result['input_verification']}")
    print(f"Output verification: {result['output_verification']}")
    print(f"Overall verification: {result['overall_verification']}")
    if result["overall_ok"]:
        print("OUTPUT CHECKS PASSED - files match the manifest.")
    else:
        print("ONE OR MORE CHECKS FAILED - files were changed or are missing.")
    print("============================================")


def verify_manifest(
    manifest_path: Path,
    *,
    evidence_root: Path | None = None,
    payroll_path: Path | None = None,
    recordkeeper_path: Path | None = None,
    prefer_evidence_root: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    print(f"Verifying manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    input_status, warnings = _verify_inputs(
        manifest,
        manifest_path,
        evidence_root=evidence_root,
        payroll_path=payroll_path,
        recordkeeper_path=recordkeeper_path,
    )
    output_status = _verify_manifest_outputs(
        manifest,
        manifest_path,
        evidence_root=evidence_root,
        prefer_evidence_root=prefer_evidence_root,
    )
    result = _manifest_result(
        input_status=input_status,
        output_status=output_status,
        warnings=warnings,
        manifest_path=manifest_path,
    )
    _print_summary(result)
    return result


def _zip_manifest_name(zf: zipfile.ZipFile) -> str:
    manifests = sorted(
        name for name in zf.namelist()
        if PurePosixPath(name).name.startswith("proof_manifest_") and name.endswith(".json")
    )
    if not manifests:
        raise FileNotFoundError("No proof_manifest_*.json found in evidence pack ZIP")
    return manifests[-1]


def _resolve_zip_member(zf: zipfile.ZipFile, info: dict[str, Any]) -> str | None:
    path_value = info.get("path")
    if not path_value:
        return None

    names = set(zf.namelist())
    normalized = str(path_value).replace("\\", "/")
    if normalized in names:
        return normalized

    basename = PurePosixPath(normalized).name
    basename_matches = sorted(name for name in names if PurePosixPath(name).name == basename)
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def _verify_zip_outputs(manifest: dict[str, Any], zf: zipfile.ZipFile) -> str:
    output_ok = True
    print("\n[OUTPUT FILES IN ZIP]")
    for key, info in manifest.get("outputs", {}).items():
        if not isinstance(info, dict):
            print(f"  [{key}] ERROR: Manifest output entry is not an object.")
            output_ok = False
            continue
        if info.get("missing"):
            print(f"  [{key}] Manifest says missing=True; skipped.")
            continue

        member = _resolve_zip_member(zf, info)
        print(f"  [{key}] {member or info.get('path')}")
        if member is None:
            print("    ERROR: Output is missing from evidence pack ZIP.")
            output_ok = False
            continue

        actual_hash = sha256_bytes(zf.read(member))
        expected_hash = info.get("sha256")
        if expected_hash != actual_hash:
            print("    MISMATCH!")
            print(f"      expected: {expected_hash}")
            print(f"      actual:   {actual_hash}")
            output_ok = False
        else:
            print("    OK (hash matches manifest).")
    return STATUS_PASS if output_ok else STATUS_FAIL


def verify_evidence_pack_zip(
    zip_path: Path,
    *,
    payroll_path: Path | None = None,
    recordkeeper_path: Path | None = None,
) -> dict[str, Any]:
    zip_path = Path(zip_path)
    print(f"Verifying evidence pack ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        manifest_name = _zip_manifest_name(zf)
        manifest = json.loads(zf.read(manifest_name).decode("utf-8"))
        manifest_context_path = zip_path.parent / PurePosixPath(manifest_name).name

        input_status, warnings = _verify_inputs(
            manifest,
            manifest_context_path,
            evidence_root=zip_path.parent,
            payroll_path=payroll_path,
            recordkeeper_path=recordkeeper_path,
        )
        output_status = _verify_zip_outputs(manifest, zf)

    result = _manifest_result(
        input_status=input_status,
        output_status=output_status,
        warnings=warnings,
        manifest_path=manifest_context_path,
    )
    _print_summary(result)
    return result


def _manifest_from_evidence_dir(evidence_dir: Path) -> Path:
    manifests = sorted(evidence_dir.glob("proof_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No proof_manifest_*.json found in {evidence_dir}")
    return manifests[-1]


def verify_evidence_pack_dir(
    evidence_dir: Path,
    *,
    payroll_path: Path | None = None,
    recordkeeper_path: Path | None = None,
) -> dict[str, Any]:
    evidence_dir = Path(evidence_dir)
    manifest_path = _manifest_from_evidence_dir(evidence_dir)
    return verify_manifest(
        manifest_path,
        evidence_root=evidence_dir,
        payroll_path=payroll_path,
        recordkeeper_path=recordkeeper_path,
        prefer_evidence_root=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ProofLink proof manifests and evidence packs.")
    parser.add_argument(
        "artifact",
        nargs="?",
        help="Manifest JSON, evidence-pack ZIP, or extracted evidence-pack directory. Omit to verify the latest default manifest.",
    )
    parser.add_argument("--payroll", help="Optional original payroll input file for source-input hash verification.")
    parser.add_argument("--recordkeeper", help="Optional original recordkeeper input file for source-input hash verification.")
    args = parser.parse_args(argv)

    payroll_path = Path(args.payroll) if args.payroll else None
    recordkeeper_path = Path(args.recordkeeper) if args.recordkeeper else None

    try:
        if args.artifact:
            artifact = Path(args.artifact)
            if not artifact.exists():
                print(f"ERROR: Supplied verification target does not exist: {artifact}", file=sys.stderr)
                return 2
            if artifact.is_dir():
                result = verify_evidence_pack_dir(
                    artifact,
                    payroll_path=payroll_path,
                    recordkeeper_path=recordkeeper_path,
                )
            elif artifact.suffix.lower() == ".zip":
                result = verify_evidence_pack_zip(
                    artifact,
                    payroll_path=payroll_path,
                    recordkeeper_path=recordkeeper_path,
                )
            else:
                result = verify_manifest(
                    artifact,
                    payroll_path=payroll_path,
                    recordkeeper_path=recordkeeper_path,
                )
        else:
            latest = find_latest_manifest()
            if latest is None:
                print(f"No proof_manifest_*.json files found in {PROOFS_DIR}")
                return 1
            result = verify_manifest(
                latest,
                payroll_path=payroll_path,
                recordkeeper_path=recordkeeper_path,
            )
    except Exception as exc:
        print(f"ERROR: Verification failed: {exc}", file=sys.stderr)
        return 1

    return 0 if result.get("overall_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

