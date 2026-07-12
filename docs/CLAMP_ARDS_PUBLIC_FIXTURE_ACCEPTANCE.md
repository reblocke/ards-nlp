# CLAMP Public Fixture Acceptance

## Completed

- [x] Restricted compatibility run compared all 227,835 documents and 80,908 entities exactly
  under the fingerprinted resource set.
- [x] The 23-file exported project is excluded from public Git and release archives.
- [x] The authorized 23-term phenotype and rule behavior are packaged as MIT JSON attributed to
  Dan Knox.
- [x] Only `defaultAbbrs.txt`, `defaultNegexDict.txt`, and `defaultTokenRule.txt` remain external
  runtime requirements.
- [x] Public CI uses independently authored synthetic resources on Linux and Windows.
- [x] Resource and history audits are mandatory.

## Pending

- [ ] Run the 463 exact-byte synthetic cases twice through the genuine licensed CLAMP environment.
- [ ] Record CLAMP build, Java, Windows, export, chronology, and per-file checksum provenance.
- [ ] Demonstrate exact run-to-run sentence, token, final-entity, multiplicity, and order stability.
- [ ] Complete named manual PHI review of every normalized expected-output file.
- [ ] Record authority to redistribute the normalized synthetic expected outputs.
- [ ] Commit only the reviewed normalized fixture; never commit raw XMI or licensed resources.
- [ ] Change the CI maturity statement from pending only after strict fixture validation passes.

See `docs/CLAMP_ARDS_GOLDEN_CORPUS_RUNBOOK.md` for the execution procedure.
