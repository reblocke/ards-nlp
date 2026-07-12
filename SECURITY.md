# Security Policy

## Supported Version

Security updates are applied to the latest release and the `main` branch.

| Version | Supported |
|---|---|
| 0.3.x | Yes |
| Earlier versions | No |

## Reporting A Vulnerability

Use [GitHub private vulnerability reporting](https://github.com/reblocke/ards-nlp/security/advisories/new).
Do not open a public issue for a suspected vulnerability, credential, patient identifier, or
restricted-data exposure.

Include:

- the affected path, version, or commit;
- a minimal reproduction using synthetic data only;
- the impact and any known exploitation conditions;
- suggested remediation, if available.

Do not attach MIMIC-CXR, REDCap, CLAMP output, report text, entity mentions, row-level predictions,
credentials, or third-party licensed resources. A maintainer will acknowledge a complete report,
assess severity, and coordinate remediation and disclosure through the private advisory.

## Research Data Incidents

This repository is not an approved channel for clinical data. If restricted or identifiable data
is found in Git history, an archive, an issue, or an Actions log:

1. stop further sharing;
2. report it privately through the advisory channel;
3. preserve only the minimum non-sensitive evidence needed for response;
4. follow the applicable institutional privacy and incident-response process.

Repository cleanup does not replace institutional reporting obligations.
