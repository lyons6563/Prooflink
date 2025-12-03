from pathlib import Path
import csv
import subprocess
import sys


# Base paths
SRC_DIR = Path(__file__).resolve().parents[1]          # ...\dev\src
RAW_DIR = SRC_DIR / "data" / "raw"                     # ...\dev\src\data\raw
PROCESSED_ROOT = SRC_DIR.parent / "data" / "processed" # ...\dev\data\processed

BATCH_MANIFEST = RAW_DIR / "batch_manifest.csv"
ANALYZER_SCRIPT = SRC_DIR / "contribution_timing_analyzer_v2.py"


def main() -> None:
    print(f"Using manifest: {BATCH_MANIFEST}")

    if not BATCH_MANIFEST.exists():
        raise FileNotFoundError(f"Batch manifest not found: {BATCH_MANIFEST}")

    with BATCH_MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            batch_id = row["batch_id"]
            payroll_file = row["payroll_file"]
            rk_file = row["rk_file"]

            payroll_path = RAW_DIR / payroll_file
            rk_path = RAW_DIR / rk_file

            # 👇 give each batch its *own* output directory
            batch_output_dir = PROCESSED_ROOT / batch_id

            print("\n====================================")
            print(f"Running batch {batch_id}")
            print("====================================")
            print(f"  Payroll: {payroll_path}")
            print(f"  RK:      {rk_path}")

            cmd = [
                sys.executable,
                str(ANALYZER_SCRIPT),
                "--payroll",
                str(payroll_path),
                "--rk",
                str(rk_path),
                "--output-dir",
                str(batch_output_dir),
            ]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Batch {batch_id} failed with exit code {result.returncode}"
                )

    print("\nAll batches completed successfully.")


if __name__ == "__main__":
    main()
