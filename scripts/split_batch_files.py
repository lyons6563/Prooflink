import os
import re
import csv

# Paths relative to your src root
INPUT_FILE = os.path.join("data", "raw", "batch_input_raw.txt")
OUTPUT_DIR = os.path.join("data", "batch")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "batch_manifest.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Matches blocks like:
# payroll_01.csv
# ```csv
# ...csv content...
# ```
pattern = r"(payroll_\d{2}\.csv|rk_\d{2}\.csv)\s*```csv\n(.*?)```"
matches = re.findall(pattern, content, flags=re.DOTALL)

if not matches:
    raise RuntimeError("No CSV blocks found in batch_input_raw.txt. Check formatting and ```csv fences.")

print(f"Found {len(matches)} CSV blocks in batch_input_raw.txt")

# For building the manifest
# batches["01"] = {"payroll": "payroll_01.csv", "rk": "rk_01.csv"}
batches = {}

for filename, csv_body in matches:
    csv_body = csv_body.strip()
    out_path = os.path.join(OUTPUT_DIR, filename)

    # Write the individual CSV file
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_body)

    print(f"Created: {out_path}")

    # Update batch mapping for manifest
    m = re.match(r"(payroll|rk)_(\d{2})\.csv", filename)
    if not m:
        continue

    side, num = m.group(1), m.group(2)  # side = "payroll" or "rk"
    if num not in batches:
        batches[num] = {}
    batches[num][side] = filename

# Now generate the manifest CSV
with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch_id", "payroll_file", "recordkeeper_file"])

    for num in sorted(batches.keys()):
        batch_id = f"batch_{num}"
        payroll_file = batches[num].get("payroll", "")
        rk_file = batches[num].get("rk", "")
        writer.writerow([batch_id, payroll_file, rk_file])

print(f"\nWrote manifest: {MANIFEST_PATH}")
print("Done.")
