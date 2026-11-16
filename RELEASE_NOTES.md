# v1.0.0 — EACL 2027 camera-ready artifact

- Original nine-direction analysis, two held-out directions, MuSiQue scope
  controls, composition accounting, and four figures.
- Same-family Qwen-7B/Qwen-72B control, separately generated from three runs.
- Twelve-direction token accounting with the nine-positive-gap headline.
- Thirty frozen result CSVs and named camera-ready claims.
- 209 hash-verified trace artifacts distributed separately from Git history.
- Verbatim prompt templates and construction-source provenance.

Assets:

- `bsir-v1.0.0-traces.tar.gz`: model-output traces and run metadata.
- `bsir-v1.0.0-reproduction.zip`: complete code, references, and traces.
- `SHA256SUMS`: archive checksums.

This release reproduces analyses from recorded responses and makes no model calls.

Packaging correction: normalized mixed LF/CRLF line endings in the original
`prose_quantities.csv` reference. All cells and numerical values are unchanged.

Public-data preparation: normalize internal serving aliases in the three
same-family Qwen runs and their metadata to canonical public model names.
No model response, answer, usage count, or analysis result changed.
