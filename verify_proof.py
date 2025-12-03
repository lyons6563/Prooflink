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


def find_latest_manifest() -> Path | None:
    """Return the most recent proof_manifest_*.json file, or None if none exist."""
    if not PROOFS_DIR.exists():
        return None
    manifests = sorted(PROOFS_DIR.glob("proof_manifest_*.json"))
    if not manifests:
        return None
    return manifests[-1]


def verify_manifest(manifest_path: Path) -> None:
    print(f"Verifying manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    overall_ok = True

    # 1) Check payroll file
    payroll_path = PROJECT_ROOT / manifest["payroll_file"]
    if payroll_path.exists():
        expected = sha256_file(payroll_path)
        print(f"\n[PAYROLL] {payroll_path}")
        print(f"  SHA256: {expected}")
    else:
        print(f"\n[PAYROLL] MISSING: {payroll_path}")
        overall_ok = False

    # 2) Check recordkeeper file
    rk_path = PROJECT_ROOT / manifest["recordkeeper_file"]
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
        # We know what the filenames are from main.py
        # This must match the mapping used when you built the manifest.
        if key == "deferral_mismatches":
            path = DATA_OUT / "deferral_mismatches.csv"
        elif key == "loan_mismatches":
            path = DATA_OUT / "loan_mismatches.csv"
        elif key == "late_deferrals":
            path = DATA_OUT / "late_deferrals_contributions.csv"
        elif key == "late_loans":
            path = DATA_OUT / "late_loans_contributions.csv"
        elif key == "excel_report":
            path = DATA_OUT / "reconciliation_report.xlsx"
        else:
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


if __name__ == "__main__":
    latest = find_latest_manifest()
    if latest is None:
        print(f"No proof_manifest_*.json files found in {PROOFS_DIR}")
    else:
        verify_manifest(latest)
