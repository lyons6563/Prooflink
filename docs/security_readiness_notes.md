# Security Readiness Notes

These are not required before a controlled demo, but they are required before real sponsor payroll or recordkeeper data is uploaded by external users.

## Current demo posture

ProofLink is acceptable as a local or controlled demo tool if synthetic or scrubbed data is used.

It is not yet ready for production sponsor data without additional controls.

## Required before production use

1. Remove all default secrets.
   - Do not allow default Streamlit password values in deployed environments.
   - Do not allow default JWT secret values.

2. Protect all API endpoints that create, view, list, or download runs.
   - `/api/v1/runs`
   - `/api/v1/runs/{run_id}`
   - `/api/v1/runs/{run_id}/evidence-pack`

3. Add user/org scoping.
   - Every run should belong to a user or organization.
   - Users should not be able to list or download other users' runs.

4. Add data retention controls.
   - configurable deletion window
   - delete raw uploads after processing if not required
   - optionally keep only evidence outputs

5. Replace SQLite for production multi-user use.
   - SQLite is fine for local demo/run history.
   - Use a production database before multi-client deployment.

6. Add upload validation and size limits.
   - file type checks
   - max file size
   - row count limits
   - clear error reporting

7. Add logging discipline.
   - never log participant-level PII unnecessarily
   - never print full payroll rows to logs
   - separate operational logs from evidence outputs

8. Add deployment documentation.
   - required environment variables
   - secret management
   - storage path
   - backup/retention behavior

## Demo rule

Until these controls are implemented, use synthetic, scrubbed, or internally controlled sample files only.

Do not ask an external sponsor to upload live payroll or recordkeeper files into a hosted ProofLink instance.
