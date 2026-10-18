# Literature Notes

This directory stores tracked literature pointers that are useful for interpreting or extending
the benchmark. These files should contain citation metadata, project relevance, and local
retrieval pointers only.

Do not commit PDFs, full-text wiki exports, long article excerpts, or copyrighted article text.
The single CC BY 4.0 article already included at
`docs/2022 Knighton JAMIA Open ARDS alert paper published.pdf` is a documented exception;
see `docs/THIRD_PARTY_NOTICES.md`. Other PDFs remain excluded.
Local retrieval packets may be generated under `artifacts/literature/`, which is ignored by git.
If a local PDF copy is ever kept under `data/external/literature/`, it must remain untracked and
have a `.source.json` provenance sidecar.

## Index

- [ARDS continuous subphenotyping](ards-continuous-subphenotyping.md): interpretation of binary
  labels and continuous or probabilistic disease representations.
- [Caldonazzo 2026: CXR-CT agreement](caldonazzo-2026-cxr-ct-agreement.md): imaging-reference
  uncertainty, report-image alignment, and implications for ARDS NLP validation and writing.
