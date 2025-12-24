# ADP Workforce Now (WFN) Quick Start Guide

## Why This Exists

The SECURE 2.0 Preflight Checker provides **auditability** and **pre-payroll gating** for compliance teams. Before processing payroll, you can:

- **Catch violations early**: Identify HCE catch-up contribution issues before payroll is finalized
- **Maintain audit trail**: Generate exception reports that document compliance checks
- **Prevent costly corrections**: Avoid post-payroll adjustments and potential penalties
- **Enable proactive compliance**: Run checks on draft payroll data before submission

This tool is designed for **export-based workflows**—no integrations required. Simply export your payroll data from ADP WFN and run the preflight check.

---

## Inputs We Need

To run the SECURE 2.0 preflight check, you'll need to export the following data from ADP Workforce Now:

### Required Fields

- **Employee ID**: Unique identifier for each employee
- **Employee Name**: Full name (for reporting purposes)
- **Gross Pay**: Pay period gross pay amount
- **YTD Gross Pay**: Year-to-date gross pay (for annualization)
- **Pay Period Start**: Start date of the pay period
- **Pay Period End**: End date of the pay period

### Optional Fields (for catch-up analysis)

- **Catch-up Contribution**: Amount of catch-up contribution (if any)
- **Catch-up Type**: Either "Roth" or "Traditional"

### Context for ADP WFN Reports

When exporting from ADP WFN, look for reports that include:

- **Roth contribution data**: To identify employees making Roth catch-up contributions
- **HCE status indicators**: To flag Highly Compensated Employees (or compensation data to calculate HCE status)
- **Social Security wages**: For accurate annualization calculations

**Recommended ADP WFN Report**: Use the "401(k) Contribution Report" or "Payroll Register" export that includes:
- Employee demographics (ID, name)
- Pay period dates and amounts
- 401(k) deferral amounts (pre-tax and Roth)
- Catch-up contribution details (if available)

---

## Step-by-Step Workflow

### Step 1: Export from ADP WFN

1. Log into **ADP Workforce Now**
2. Navigate to **Reports** → **Payroll Reports** (or **Benefits Reports**)
3. Select a report that includes:
   - Employee ID and name
   - Pay period dates (start and end)
   - Gross pay and YTD gross pay
   - 401(k) contribution details (if available)
4. **Export as CSV** format
5. Save the file with a descriptive name (e.g., `adp_payroll_2024_01.csv`)

### Step 2: Prepare Your CSV

Ensure your exported CSV has the following column headers (case-sensitive):

```
employee_id,employee_name,gross_pay,ytd_gross_pay,pay_period_start,pay_period_end,catch_up_contribution,catch_up_type
```

**Note**: If your ADP export uses different column names, you may need to rename them to match the required headers.

### Step 3: Configure the Preflight Checker

1. Copy the example config template:
   ```bash
   cp configs/secure20_preflight_config.adp_wfn.example.yaml configs/my_adp_config.yaml
   ```

2. Edit `configs/my_adp_config.yaml` and set:
   - **current_year**: The calendar year you're analyzing (e.g., `2024`)
   - **compensation_limit**: HCE threshold for the year (e.g., `150000` for 2024)
   - **age_threshold**: Minimum age for catch-up contributions (typically `50`)
   - **roth_only_risk_year**: Year when Roth-only catch-up restriction begins (typically `2024` or later)
   - **method**: Annualization method (`"gross_or_ytd"` recommended)

### Step 4: Run the Preflight Check

Execute the preflight checker from your terminal:

```bash
python secure20_preflight.py --payroll adp_payroll_2024_01.csv --config configs/my_adp_config.yaml
```

**Alternative with custom output path**:
```bash
python secure20_preflight.py -p adp_payroll_2024_01.csv -c configs/my_adp_config.yaml -o preflight_exceptions.csv
```

### Step 5: Review Results

The tool will output:

- **Console summary**: 
  - `SAFE` or `NOT SAFE` status
  - Number of violations found
  - Top employee IDs with issues

- **Exception CSV file**: Detailed list of violations with:
  - Employee ID and name
  - Violation type and description
  - Relevant amounts and dates

**Exit Codes**:
- `0` = SAFE (no violations detected)
- `2` = NOT SAFE (violations found)

---

## Common Failure Modes

### 1. "ADP says it's set up, but violations still appear"

**Problem**: ADP WFN may show that Roth catch-up is configured, but the actual payroll data doesn't reflect the change.

**Solution**: 
- Verify the export includes the most recent pay period data
- Check that `catch_up_type` column shows "Roth" for HCEs age 50+
- Confirm the payroll has been processed with the new settings

### 2. "Vanguard says it'll switch, but it doesn't"

**Problem**: Recordkeeper (e.g., Vanguard) may indicate they'll automatically switch catch-up contributions to Roth, but the change hasn't taken effect.

**Solution**:
- Run the preflight check on the **actual payroll export** (not just the recordkeeper's promise)
- Verify the `catch_up_type` field in your export matches expectations
- Contact both ADP and Vanguard to confirm the change has been implemented in the payroll system

### 3. "Missing required CSV columns"

**Problem**: The exported CSV doesn't match the expected column names.

**Solution**:
- Check that column headers exactly match: `employee_id`, `employee_name`, `gross_pay`, `ytd_gross_pay`, `pay_period_start`, `pay_period_end`
- Rename columns in Excel or a text editor before running the preflight check
- Ensure dates are in a standard format (YYYY-MM-DD recommended)

### 4. "No violations found, but I know there should be"

**Problem**: The preflight check reports SAFE, but you suspect there are issues.

**Solution**:
- Verify the config file has the correct `roth_only_risk_year` (should match the year you're analyzing)
- Check that `age_threshold` is set correctly (typically 50)
- Ensure the export includes all employees (not just a subset)
- Review the exception CSV even if status is SAFE—it may contain warnings

### 5. "HCE status not detected correctly"

**Problem**: Employees who should be flagged as HCEs are not being identified.

**Solution**:
- Verify `compensation_limit` in config matches the current year's HCE threshold
- Check that `ytd_gross_pay` in the export is accurate and up-to-date
- Ensure `annualization.method` is set appropriately (`"gross_or_ytd"` recommended)

---

## Next Steps

After running the preflight check:

1. **Review exceptions**: Open the exception CSV and identify employees with violations
2. **Contact affected employees**: Notify HCEs age 50+ who need to switch to Roth catch-up
3. **Update payroll settings**: Work with ADP to ensure Roth catch-up is properly configured
4. **Re-run preflight**: After making changes, export fresh data and run the check again
5. **Document results**: Save exception reports for audit purposes

---

## Support

For questions or issues:
- Review the exception CSV for detailed error messages
- Check that your config file matches the example template
- Ensure your CSV export includes all required columns
- Verify date formats and numeric values are valid

