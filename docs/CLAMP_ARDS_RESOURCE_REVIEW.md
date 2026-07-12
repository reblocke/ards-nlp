# CLAMP Resource Boundary

## Release Decision

The complete 23-file exported CLAMP project is excluded from this repository and from all release
archives. Fingerprints are retained only to detect accidental redistribution.

The public Python compatibility mirror has two distinct inputs:

1. The packaged MIT phenotype specification contains the authorized custom dictionary, pipeline
   choices, and rule semantics attributed to Dan Knox.
2. Three default resources remain external and separately licensed:
   `defaultAbbrs.txt`, `defaultNegexDict.txt`, and `defaultTokenRule.txt`.

No CLAMP executable, POS model, descriptor, exported config, dictionary file, Ruta file, pipeline
file, or generated wrapper is distributed.

## Runtime Layout

The documented ignored default is:

```text
data/external/clamp_ards_project/
  Components/
    Assertion classifier/DF_NegEx_assertion/defaultNegexDict.txt
    Sentence detector/DF_Clamp_sentence_detector/defaultAbbrs.txt
    Tokenizer/DF_Clamp_tokenizer/defaultTokenRule.txt
```

Users may instead set `ARDS_CLAMP_PROJECT_DIR` or pass `--project-dir`. Production runs use the
packaged fingerprint manifest. Tests pass an independently authored synthetic manifest through
`ARDS_CLAMP_RESOURCE_MANIFEST` or `--resource-manifest`.

## Manifest v3

`config/clamp_ards_resource_manifest.json`:

- preserves path, SHA-256, and byte-size provenance for all 23 excluded export files;
- declares only the three external default files as `runtime_required_files`;
- records the authorized phenotype specification version, attribution, license, and SHA-256;
- records the prior v2 manifest hash so restricted v2 fixture provenance remains readable.

The batch summary fingerprints only the three files actually read and separately records
`phenotype_spec_version` and `phenotype_spec_sha256`.

## Audit Contract

`make clamp-ards-resources-public-audit` must pass on every pull request. It fails when:

- an excluded export path is tracked or remains in reachable history;
- a tracked or historical blob matches one of the 23 excluded file hashes;
- a manifest or ledger row is missing, inconsistent, or unsafe;
- a required resource is neither cleared for redistribution nor kept behind the documented
  external-resource boundary.

`make release-audit` separately rejects Office documents, restricted binary/data extensions,
machine-local paths, and generated output in tracked or reachable state.

## Remaining Reproducibility Work

The external-resource boundary is complete. The genuine public synthetic oracle is not. Public
fixture completion still requires:

- two genuine runs through the licensed CLAMP environment;
- complete runtime and output checksums;
- exact run-to-run determinism evidence;
- named manual PHI review;
- approval to redistribute the normalized synthetic expected outputs.

Until then, CI reports the fixture as explicitly pending while keeping the resource audit hard.
