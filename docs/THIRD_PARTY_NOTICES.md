# Third-Party Notices

This repository contains adapters and provenance metadata for external research software. External
source, data, models, and licensed resources are not relicensed by the repository's MIT License.

## Knighton et al. JAMIA Open Article

- File: `docs/2022 Knighton JAMIA Open ARDS alert paper published.pdf`
- Citation: Knighton AJ, et al. An alert tool to promote lung protective ventilation for possible
  acute respiratory distress syndrome. *JAMIA Open*. 2022;5(2):ooac050.
- DOI: `https://doi.org/10.1093/jamiaopen/ooac050`
- License: Creative Commons Attribution 4.0 International, as stated in the article.
- SHA-256: `3ee913cfdc6f9fa5845f9988eefa36b1ef6b8918a06d20973bb7251727e19437`
- This article remains under its own license; the repository's MIT License does not apply to it.

## Legacy ARDS CLAMP Compatibility

The repository contains no CLAMP executable and no exported CLAMP project.

- The custom 23-term ARDS dictionary, pipeline choices, and rule semantics were designed by Dan
  Knox and are re-expressed in an MIT-licensed JSON specification with authorization.
- The original 23-file export is excluded. Its SHA-256 and size fingerprints remain in
  `config/clamp_ards_resource_manifest.json` so release audits can detect accidental inclusion.
- The Python mirror requires only `defaultAbbrs.txt`, `defaultNegexDict.txt`, and
  `defaultTokenRule.txt`. Users must obtain those resources separately under applicable licenses.
- The optional resources belong under ignored `data/external/clamp_ards_project/` or another path
  supplied through `ARDS_CLAMP_PROJECT_DIR` or `--project-dir`.
- The POS model, descriptors, generated wrappers, CLAMP configuration files, original dictionary,
  original Ruta file, pipeline export, and documentation export are not required or distributed.
- Real report inputs, CLAMP outputs, entity tables, and row-level predictions remain restricted and
  ignored.

The file-level evidence and disposition are recorded in
`docs/CLAMP_ARDS_RESOURCE_LEDGER.csv` and `docs/CLAMP_ARDS_RESOURCE_REVIEW.md`.

## Amaral/Morales ARDS Diagnosis

- Source: `https://github.com/amarallab/ARDS_diagnosis`
- Pinned commit: `6154ac32e16dd9497a466351582603e1c1095a05`
- Upstream license: GPL-2.0
- Boundary: source and pickle artifacts remain under ignored `data/external/` and execute in a
  separate environment. No upstream implementation is copied into this package.

## UW BioNLP HANSO

- Source: `https://github.com/uw-bionlp/ards`
- Pinned commit: `e9fc27f7034cc6b54f0ccdba4a58377948cf0258`
- Upstream license: BSD-3-Clause
- Boundary: source, weights, caches, and legacy environment remain external. Model execution is
  gated until trained weights and terms are documented.

## Afshar Joyce ARDS Classifier

- Source: `https://github.com/AfsharJoyceInfoLab/ARDS_Classifier`
- Pinned commit: `0ede2cfab5349f6dcf3a05e04060ddd045db0095`
- Upstream license: none found in the pinned repository
- Boundary: only static pickle inspection is enabled by default. MIMIC text cannot be supplied
  without explicit permission and synthetic-anchor review.
