# Design principles (scientific programming)

This repository is organized around a small set of design principles that keep scientific code:

- correct (as far as can be operationalized)
- reproducible
- reviewable
- extensible

## 1) Make the scientific claim explicit

Treat code as a sequence of falsifiable claims.

- Write down the question, assumptions, and success criteria in the issue/PR.
- Encode success criteria as tests where feasible.
- Separate exploratory work from confirmatory pipelines.

## 2) Prefer simple, boring structure

- Use a predictable directory layout (data/code/reports separated).
- Keep scripts thin; keep reusable logic in `src/`.
- Choose a small set of tooling standards and enforce them automatically.

## 3) Functional core, imperative shell

- Pure functions for transformations.
- Side effects (I/O, randomness, network) concentrated at the edges.
- Configuration passed in explicitly (YAML or CLI), not through hidden globals.

## 4) Reproducibility is a feature

- Pin environments (`pyproject.toml` + `uv.lock`).
- Record seeds and random state.
- Keep raw inputs immutable; write derived artifacts to separate directories.
- For any fetched external artifact, store provenance (URL, version/hash, date, license).

## 5) Test what matters

- Unit tests for invariants (schema, types, boundary conditions).
- Regression tests for known failure modes and previous bugs.
- CI runs tests and linting on every PR.

## 6) Make change safe

- Version control is mandatory.
- Prefer small PRs.
- Use code review to detect hidden assumptions, missing tests, and scientific misuse.
- Track key decisions in `docs/DECISIONS.md`.

## 7) Avoid hidden state

Common failure modes:

- relying on current working directory
- relying on implicit global variables
- scripts that only run interactively
- notebooks with hidden cell ordering

Countermeasures:

- find repo root programmatically
- pass state explicitly
- make notebooks restartable
- make pipelines idempotent

## 8) Security and privacy by default

- never commit credentials
- keep secrets in `.env` (ignored)
- minimize data exposure in logs and artifacts

## 9) Optimize late, guided by measurement

- profile first
- optimize only bottlenecks
- keep correctness tests in place during optimization

## 10) Treat AI assistance as a power tool

AI can accelerate implementation, but it increases the risk of subtle errors.

- require agent-produced code to be small, testable, and reviewable
- insist on explicit assumptions and acceptance tests
- verify behavior empirically (unit tests + minimal end-to-end runs)
