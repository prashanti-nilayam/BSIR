# Release validation

Validated on 15 November 2026 using CPython 3.13.8 on macOS arm64, with the
11 exact package versions in `requirements-lock.txt`.

## Isolated reproduction

1. Cloned the prepared Git repository into a temporary directory, without
   references to the working project's source tree or its installed modules.
2. Confirmed all 30 frozen CSVs preserved exact bytes after Git checkout.
3. Submitted an archive with an incorrect checksum: rejected before installation.
4. Installed `bsir-v1.0.0-traces.tar.gz` with `scripts/fetch_traces.py`: all
   209 members and individual checksums verified.
5. Ran `python scripts/reproduce.py --output verification` using the fresh
   pinned Python environment.

Result:

```text
PASS: 106/106 original numerical claims agree with the manuscript
PASS: 12 camera-ready quantities match named claims
PASS: 30 result CSVs match exactly; plot data and design quantities are consistent; 4 figures were generated
REPRODUCTION PASS
```

The original fitted populations remain unchanged. The same-family control and
12-direction token accounting are generated and verified separately. The
external trace-directory override also passes manifest verification.

## Release checks

- Checked all 771,925 trace rows and metadata for credential-shaped strings,
  credential field names, and private host paths: no findings in the checks run.
- Raw traces, environments, logs, and Python caches are excluded from Git history.
- Analysis scripts pass Python syntax compilation.
- Complete reproduction ZIP passes member uniqueness and CRC checks.
- Archive hashes are distributed in `SHA256SUMS`; the trace-archive checksum
  is also tracked in `data/trace_archive.sha256`.

The checks reproduce analysis from recorded responses; they do not rerun model
inference or independently validate the original model endpoints. Figure plot
data and PDF validity are checked; cross-platform PDF byte identity is not required.

The original `prose_quantities.csv` contained mixed line endings on seven rows.
The public reference uses consistent CSV line endings with identical cells;
`.gitattributes` preserves CSV bytes on checkout for the strict verifier.

## Public model-identifier normalization

Normalized the model identifiers in three same-family Qwen JSONL runs and
three corresponding metadata files to the canonical public model names.
Compared all 11,838 affected records with the preserved experiment originals:
only the model field differs. Metadata changes are limited to the model and
panel-model fields. Response text, answers, token counts, ordering, and resumed
run duplicates are unchanged. No internal serving aliases remain in public
trace or metadata bytes.

Regenerated the 209-file manifest, archive checksum, and both release bundles.
Reran the same-family and token analyses and camera-ready claim checks: their
CSV outputs match the frozen references exactly. A fresh installation of the
normalized archive also verifies every archive member and checksum.
