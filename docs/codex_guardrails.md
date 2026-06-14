# Codex Guardrails

These guardrails exist to keep the next build from becoming impressive but commercially useless.

## Build only the wedge

The wedge is:

> Compensation definition errors that create employer match impact.

Do not build around vague compliance intelligence.

## Test first

Codex should build a standalone analyzer and tests before touching the engine or UI.

## No broad refactor

Do not reorganize the repo, rename core files, or redesign the app unless absolutely required.

## Keep outputs audit-friendly

Every flagged issue should have:

- participant identifier
- period/pay date
- source values
- calculated expected value
- actual value
- variance
- likely root cause
- evidence output

## No legal conclusion

The product detects exceptions and organizes evidence. It does not make final legal determinations.
