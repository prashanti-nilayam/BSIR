# Borrowed Strength, Imported Risk

Reproduction code and experimental traces for **Borrowed Strength, Imported
Risk: Capability Gaps and Reception Govern Heterogeneous LLM Debate**, accepted
at EACL 2027.

Repository: https://github.com/prashanti-nilayam/BSIR

[Download v1.0.0 data and bundles](https://github.com/prashanti-nilayam/BSIR/releases/tag/v1.0.0).

Prashanti Nilayam, Kiran Kumar Ramanna, Prashil Tumbade, Nagaraj N Bhat, and
Sankalp Nayak — ServiceNow.

## What this release reproduces

The artifact reconstructs the paper's tables, full-precision plot data, four
figures, design counts, and named numerical claims from recorded responses.
It includes the same-family Qwen-7B ← Qwen-72B control and the direct-peer token
comparison in the camera-ready paper. These additions are evaluated separately
and do not alter the original nine-direction fits or two held-out directions.

**No API keys, GPU, benchmark downloads, or model inference are needed.**
This release reruns analysis of recorded calls. It does not provide a supported
end-to-end runner for regenerating those calls against historical model endpoints.
Prompt templates, construction source, and recorded run configurations are
provided for inspection.

## Quick start

Use Python 3.13 for the fully validated environment. The analysis code uses
Python 3.11+ syntax; older Python versions also require compatible dependency
wheels. Pinned top-level dependencies are in `requirements.txt`; all packages
in the validated environment are frozen in `requirements-lock.txt`.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Download `bsir-v1.0.0-traces.tar.gz` from the repository's **v1.0.0 Release**,
then install and verify it:

```bash
.venv/bin/python scripts/fetch_traces.py --archive /path/to/bsir-v1.0.0-traces.tar.gz
.venv/bin/python scripts/reproduce.py --output reproduced
```

Alternatively, download and verify the fixed release asset directly:

```bash
.venv/bin/python scripts/fetch_traces.py --url https://github.com/prashanti-nilayam/BSIR/releases/download/v1.0.0/bsir-v1.0.0-traces.tar.gz
```

The complete reproduction ZIP includes the same code and an already populated `traces/`
directory, so ZIP users skip `fetch_traces.py`.

The run first verifies every trace against `MANIFEST.sha256`, regenerates 30
CSV result files and four figures, then compares every CSV byte-for-byte with
`results/` and checks accounting identities and paper quantities. Success ends
with `REPRODUCTION PASS`. Generated outputs are isolated under `reproduced/`;
the recorded traces and reference results are not modified.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Parsers, statistical analyses, figure generation, and validators |
| `results/` | Frozen CSV references for the camera-ready paper |
| `claims/` | Independently specified original and camera-ready numerical claims |
| `prompts/` | Verbatim prompt templates and construction-source snapshots |
| `data/` | Trace download instructions and archive checksum |
| `MANIFEST.sha256` | SHA-256 for all 209 trace artifacts |
| `traces/` | Downloaded data, excluded from Git history |

See [DATASET.md](DATASET.md) for the record schema, provenance, support rules,
and known coverage gaps; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for analysis
populations and determinism; [prompts/README.md](prompts/README.md) for prompts.

## Data and licensing

This is a model-output trace release, not a new benchmark. It includes 109 JSONL
runs and 100 run-metadata files. MATH-hard, GSM8K, and MuSiQue are the source
benchmarks. Benchmark question/context text and model weights are not included;
model responses may quote parts of their input questions.

Code and documentation: **MIT**. Original model outputs, claims, and derived
results: **CC BY 4.0**. Benchmark-derived identifiers and reference answers
retain their source terms. The source benchmark licenses are MIT for
MATH/MATH-500 and GSM8K, and CC BY 4.0 for MuSiQue. See [LICENSE.md](LICENSE.md)
and the upstream links in [DATASET.md](DATASET.md).

## Citation

Please cite the paper when using its analyses or traces. `CITATION.cff` records
the software authors and version. The archival paper citation will be added
when the ACL Anthology record is available.

```bibtex
@inproceedings{nilayam2027borrowedstrength,
  title = {Borrowed Strength, Imported Risk: Capability Gaps and Reception Govern Heterogeneous LLM Debate},
  author = {Nilayam, Prashanti and Ramanna, Kiran Kumar and Tumbade, Prashil and Bhat, Nagaraj N and Nayak, Sankalp},
  booktitle = {Proceedings of the 20th Conference of the European Chapter of the Association for Computational Linguistics},
  year = {2027},
  note = {Accepted; archival publication details forthcoming}
}
```

Questions and corrections can be raised in repository issues. For correspondence:
prashanti.nilayam@servicenow.com.
