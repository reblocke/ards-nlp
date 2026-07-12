# Contributing

## Ground rules

- Prefer **small, reviewable** pull requests.
- Keep the **computational core** pure; keep I/O in scripts/CLI layers.
- Update or add tests when behavior changes.
- If you make a design/assumption change, write it down in `docs/DECISIONS.md`.

## Optional local hooks

If you want git hooks in your local clone:

```bash
uv sync --group hooks
uv run --group hooks pre-commit install
```

The default verification flow does not require `pre-commit`.

## Definition of done

A PR is typically ready to merge when:

- `make fmt` is clean
- `make lint` is clean
- `make test` passes
- Any report outputs impacted by the change are updated (if applicable)
- Modeling changes update `docs/MODELING_RUNBOOK.md` or `docs/MODELING_SNAPSHOT_V1.md` when
  commands, outputs, or interpretation change
- Annotation/review changes update `docs/ANNOTATION_RUNBOOK.md` when review input or validation
  expectations change
- Pilot-planning changes update `docs/ANNOTATION_PLANNING.md` and keep scenario outputs aggregate
- Comparator changes refresh `docs/COMPARATOR_SNAPSHOT_V1.md` after the full local bakeoff passes
- New external artifacts include provenance sidecars (see `docs/DATA_MANAGEMENT.md`)
- No raw MIMIC report text, local configs, generated model artifacts, predictions, or reviewer-level
  labels are staged
- `make release-audit` and `make clamp-ards-resources-public-audit` pass before a release is tagged

## Pull request checklist

- [ ] What is the scientific/analysis goal of the change?
- [ ] What are the success criteria and how were they verified?
- [ ] Are new dependencies necessary? If yes, were they added via `uv add` and locked?
- [ ] Does the change keep paths relative (no hard-coded absolute paths)?
- [ ] Does the change avoid hidden state and non-determinism?
- [ ] Does the change preserve the `silver_*` vs future `gold_*` naming distinction?
