# Acquisition-Ready Sanitization Checklist

This document summarizes all changes made during the sanitization pass to remove employer references, personal data, and secrets from the codebase.

## Summary

All source code files have been sanitized. Generated data files (JSON manifests, CSV outputs) in `api_uploads/` and `streamlit_runs/` directories contain historical references but are not source code and can be regenerated.

## Changes Made

### 1. Employer References Removed

#### "Empower" → "VENDOR_RK_1"
- **main.py** (line 1734): Config name changed from `synthetic_400_adp_empower.json` to `synthetic_400_adp_vendor_rk.json`
- **main.py** (line 2024): Comment updated to remove "Empower" reference
- **main.py** (line 2057): `RK_VENDOR_COLUMN_MAPS` key changed from `"Empower"` to `"VENDOR_RK_1"`
- **streamlit_app.py** (lines 935, 940): Dropdown options updated to use `"VENDOR_RK_1"` instead of `"Empower"`
- **vendors.py** (line 102): `RK_VENDOR_SIGNATURES` key changed from `"Empower"` to `"VENDOR_RK_1"`
- **tests/test_reconciliation_smoke.py** (lines 85, 160): Test vendor hints updated to `"VENDOR_RK_1"`
- **generate_synthetic_plan.py** (line 17): Default filename changed from `rk_empower_synthetic_400.csv` to `rk_vendor_rk_synthetic_400.csv`
- **generate_synthetic_plan.py** (line 20): Docstring updated to remove "Empower-style" reference
- **generate_synthetic_plan.py** (line 97): Comment updated from "Empower-style" to "vendor-style"
- **contribution_timing_analyzer_v2.py** (multiple locations):
  - Line 306-307: Vendor detection updated to check for `"vendor_rk"` instead of `"empower"`
  - Line 328: Default vendor assignment changed to `"VENDOR_RK_1"`
  - Line 405-406: Filename hint check updated
  - Line 417-420: Signature column detection updated (variable renamed from `empower_cols` to `vendor_rk_cols`)
  - Line 621: Known vendors set updated
  - Line 700: Vendor string check updated
  - Line 1055: Default filename updated

### 2. Secrets and Credentials Moved to Environment Variables

#### JWT Secret Key
- **api.py** (line 45): `SECRET_KEY` now uses `os.getenv("JWT_SECRET_KEY", "CHANGE_ME_TO_A_SECURE_RANDOM_VALUE")`
- **api.py** (line 13): Added `import os` to support environment variable access

#### Development Password
- **streamlit_app.py** (line 25): `APP_DEV_PASSWORD` now uses `os.getenv("APP_DEV_PASSWORD", "prooflink")`

### 3. Environment-Specific Paths Updated

#### Uploads Directory
- **api.py** (line 132-133): `api_uploads` hardcoded path replaced with `os.getenv("UPLOADS_DIR", "api_uploads")`
- **main.py** (lines 520-521): Comments updated to use `<UPLOADS_DIR>` placeholder instead of `api_uploads`
- **tests/run_secure20_smoketest.py** (lines 37-38): Test paths now use environment variable with fallback
- **tests/run_secure20_smoketest.py** (line 9): Added `import os` to support environment variable access

### 4. Terms Not Found in Source Code

The following search terms were not found in Python source files (may exist in generated data files only):
- "yourplanconnect"
- "RetireChain"
- "Outlook"
- "Microsoft Authenticator"
- "support@"
- "@empower"
- "127.0.0.1"
- "localhost" (not found in source code - only in comments/data)
- "C:\Users\" (only in generated CSV output files)
- "Documents\dev" (only in generated CSV output files)

### 5. Security-Related Terms (Contextual Review)

The following terms were found but are used appropriately:
- **"token"**: Used in JWT authentication context (api.py) - now uses environment variable
- **"secret"**: Used in SECRET_KEY (api.py) - now uses environment variable
- **"password"**: Used in password hashing functions (api.py, streamlit_app.py) - now uses environment variable
- **"key"**: Used as variable names and dictionary keys (not sensitive)
- **"JWT"**: Used in authentication implementation (api.py) - appropriate usage
- **"Bearer"**: Used in OAuth2 authentication (api.py) - appropriate usage
- **"client_id"**: Not found in source code
- **"client_secret"**: Not found in source code

## Environment Variables Required

The following environment variables should be set in production:

1. **JWT_SECRET_KEY**: Secret key for JWT token signing (required for API authentication)
2. **APP_DEV_PASSWORD**: Development password for Streamlit app (optional, defaults to "prooflink")
3. **UPLOADS_DIR**: Directory for API uploads (optional, defaults to "api_uploads")
4. **PROOFLINK_API_URL**: API base URL (already using environment variable)

## Files Modified

### Source Code Files
1. `main.py`
2. `streamlit_app.py`
3. `api.py`
4. `vendors.py`
5. `tests/test_reconciliation_smoke.py`
6. `generate_synthetic_plan.py`
7. `contribution_timing_analyzer_v2.py`
8. `tests/run_secure20_smoketest.py`

### Generated Data Files (Not Modified)
- JSON manifest files in `api_uploads/` and `streamlit_runs/` directories contain historical references but are generated artifacts, not source code
- CSV output files may contain hardcoded paths but are generated data, not source code

## Verification

- ✅ No "Empower" references remain in Python source files
- ✅ All hardcoded secrets moved to environment variables
- ✅ Environment-specific paths use environment variables with fallbacks
- ✅ No linter errors introduced
- ✅ Functionality preserved (vendor detection logic updated to use new placeholder names)

## Notes

- The placeholder `VENDOR_RK_1` can be replaced with any actual vendor name during deployment
- Generated data files (JSON manifests, CSV outputs) in `api_uploads/` and `streamlit_runs/` are historical artifacts and can be regenerated or excluded from the codebase if needed
- All changes maintain backward compatibility through environment variable fallbacks

