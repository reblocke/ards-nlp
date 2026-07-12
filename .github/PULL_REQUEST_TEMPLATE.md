## Summary

What does this change do, and why?

## Verification

- [ ] `make fmt`
- [ ] `make lint`
- [ ] `make test`
- [ ] Pipeline run (if relevant): `make run`
- [ ] BigQuery/modeling target run (if relevant): `make qa`, `make modeling-smoke`,
      `make modeling`, or `make modeling-qa`

## Notes for reviewers

- Any design/assumption changes documented in `docs/DECISIONS.md`?
- Any new external artifacts include `<file>.source.json` provenance?
- Any modeling or annotation workflow changes reflected in the relevant runbook?
- Any raw MIMIC data, report text, local config, generated predictions/models, or reviewer-level
  labels intentionally excluded from git?
- Does the change preserve `silver_*` naming for automated labels and reserve `gold_*` for reviewed
  labels?
