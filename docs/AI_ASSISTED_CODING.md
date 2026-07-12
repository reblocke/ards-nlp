# AI-assisted coding guardrails

Coding assistants can raise throughput, but they also raise the probability of subtle errors.

## Operational rules

- Treat AI output as **untrusted code** until verified.
- Require **tests or executable checks** for any behavior change.
- Prefer **small diffs**; reject broad refactors without a clear payoff.
- Keep domain assumptions explicit (in PRs and `docs/DECISIONS.md`).

## Ten implementation principles (for scientific code)

These principles are baked into the repo structure and workflows:

1. Adopt sensible standards.
2. Track changes (version control).
3. Accelerate the research workflow (automation, pipelines).
4. Write “good” code (readability, modularity, composability).
5. Test your code.
6. Think about others (future you, collaborators).
7. Seek awareness and consensus (shared norms).
8. Conduct code reviews.
9. Build a shared knowledge base (docs, runbooks).
10. Consider infrastructure (CI, compute, data storage).

## What to demand from an agent

- A clear plan and acceptance tests.
- Minimal, reviewable patches.
- Updated unit tests (or a rationale for why tests are not feasible).
- A brief post-change verification report.
