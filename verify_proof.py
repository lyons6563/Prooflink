from pathlib import Path
import json
import hashlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_OUT = PROJECT_ROOT / "data" / "processed"
PROOFS_DIR = PROJECT_ROOT / "proofs"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of an entire file (binary)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_manifest_output_path(key: str, info: dict, manifest_path: Path) -> Path | None:
    path_value = info.get("path")
    if path_value:
        path = Path(path_value)
        if path.is_absolute():
            if path.exists():
                return path
            return manifest_path.parent / path.name

        candidates = [
            manifest_path.parent / path,
            manifest_path.parent.parent / "output" / path,
            DATA_OUT / path,
            PROJECT_ROOT / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        if len(path.parts) == 1:
            try:
                for candidate in manifest_path.parent.parent.rglob(path.name):
                    if candidate.is_file():
                        return candidate
            except Exception:
                pass

        return candidates[0]

    known_outputs = {
        "deferral_mismatches": DATA_OUT / "deferral_mismatches.csv",
        "loan_mismatches": DATA_OUT / "loan_mismatches.csv",
        "late_deferrals": DATA_OUT / "late_deferrals_contributions.csv",
        "late_loans": DATA_OUT / "late_loans_contributions.csv",
        "excel_report": DATA_OUT / "reconciliation_report.xlsx",
    }
    return known_outputs.get(key)


def _resolve_input_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        if path.exists():
            return path
        return manifest_path.parent / path.name
    candidates = [manifest_path.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_latest_manifest() -> Path | None:
    """Return the most recent proof_manifest_*.json file, or None if none exist."""
    if not PROOFS_DIR.exists():
        return None
    manifests = sorted(PROOFS_DIR.glob("proof_manifest_*.json"))
    if not manifests:
        return None
    return manifests[-1]


def verify_manifest(manifest_path: Path) -> bool:
    print(f"Verifying manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    overall_ok = True

    # 1) Check payroll file
    payroll_path = _resolve_input_path(manifest["payroll_file"], manifest_path)
    if payroll_path.exists():
        expected = sha256_file(payroll_path)
        print(f"\n[PAYROLL] {payroll_path}")
        print(f"  SHA256: {expected}")
    else:
        print(f"\n[PAYROLL] MISSING: {payroll_path}")
        overall_ok = False

    # 2) Check recordkeeper file
    rk_path = _resolve_input_path(manifest["recordkeeper_file"], manifest_path)
    if rk_path.exists():
        expected = sha256_file(rk_path)
        print(f"\n[RECORDKEEPER] {rk_path}")
        print(f"  SHA256: {expected}")
    else:
        print(f"\n[RECORDKEEPER] MISSING: {rk_path}")
        overall_ok = False

    # 3) Check each output listed in the manifest
    outputs = manifest.get("outputs", {})
    print("\n[OUTPUT FILES]")

    for key, info in outputs.items():
        path = _resolve_manifest_output_path(key, info, manifest_path)
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
            overall_ok = False
            continue

        expected_hash = info.get("sha256")
        actual_hash = sha256_file(path)

        if expected_hash != actual_hash:
            print("    MISMATCH!")
            print(f"      expected: {expected_hash}")
            print(f"      actual:   {actual_hash}")
            overall_ok = False
        else:
            print("    OK (hash matches manifest).")

    print("\n============================================")
    if overall_ok:
        print("ALL CHECKS PASSED – files match the manifest.")
    else:
        print("ONE OR MORE CHECKS FAILED – files were changed or are missing.")
    print("============================================")
    return overall_ok


if __name__ == "__main__":
    latest = find_latest_manifest()
    if latest is None:
        print(f"No proof_manifest_*.json files found in {PROOFS_DIR}")
    else:
        verify_manifest(latest)
