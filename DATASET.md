# Experimental trace data

## Scope and provenance

The archive contains 771,925 recorded JSONL rows (including resumed-run
duplicates), across 109 runs and 100 JSON metadata records: the original
106 runs/97 metadata files, plus three Qwen-7B/Qwen-72B same-family runs and
three matching metadata files. Files are partitioned into `experiment/` and
`control/`. The main panel definitions live in `scripts/panels.py`; the
additional control is in `scripts/same_family.py`.

The benchmarks are public English-language reasoning and question-answering data:

- MATH/MATH-500: 262 level-4/5 items for MATH-hard; [MATH](https://github.com/hendrycks/math),
  [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).
- GSM8K: 1,319 test items, plus a separately labeled 100-item small-model diagnostic;
  [upstream](https://github.com/openai/grade-school-math).
- MuSiQue-Ans: 300 matched questions;
  [upstream](https://github.com/StonyBrookNLP/musique).

Identifiers, reference answers, model responses, and usage counts are recorded;
source question/context fields are omitted. Responses can repeat source text.
No model weights or provider software are distributed.

## JSONL schema

Each nonblank line is one recorded model call. Core fields are:

| Field | Meaning |
| --- | --- |
| `item_id` | Benchmark item identifier, interpreted within the run's benchmark |
| `round` | Zero-based round: independent answer at R0, revision at R1–R4 |
| `agent_id` | Zero-based panel slot; most panels use 0, 1, 2 |
| `model` | Public model identifier; internal serving aliases are normalized as described below |
| `raw_output` | Recorded model response |
| `gold_answer` | Reference answer; MuSiQue may include aliases separated by `|||` |
| `parsed_answer` | Parser output stored during inference, retained for provenance |
| `correct` | Original correctness flag, retained but not trusted by reproduction |
| `prompt_tokens`, `completion_tokens` | Model/provider-reported token usage |
| `adversary_retries` | Retry count where recorded |

Fields may vary across historical runs; absent optional fields are not inferred.
The token comparison requires reported prompt and completion usage and fails
rather than filling missing values. The reproduction reparses `raw_output` and
re-scores against `gold_answer` with `scripts/scoring.py`.

## Public model names

Internal serving aliases in the three same-family Qwen runs have been replaced
with `Qwen-2.5-7B-Instruct` and `Qwen-2.5-72B-Instruct`, in both JSONL model
fields and metadata panel compositions. Only model identifiers changed: outputs,
reference answers, usage counts, order, and duplicate rows are preserved. Missing
historical model fields remain missing. The manifest and archive checksum cover
the normalized public files; the analysis results are unchanged.

## Indexing and support

The key is `(item_id, round, agent_id)` within a run. Resumed runs can contain
duplicates; the last record written wins. R0/R1 comparisons and full R0–R4
trajectories use different completeness requirements, explicitly implemented in
the analysis scripts. Do not pool run rows as if every row were independent.
Bootstrap clustering shares benchmark-item weights across replicates, slots,
and conditions. All analyses use the recorded model outputs, not new calls.

The additional Qwen-72B control has three replicates and 786 matched
replicate-items. It has no adversarial arm. The token accounting contains 12
directions, of which nine have positive reference capability gaps. These nine
are distinct from the original nine primary directions.

Metadata describe recorded settings, timestamps, model names, prompt-template
paths, and configurations. Nine original runs lack metadata; the release does
not invent missing files. The primary panel definitions and paper specify the
analysis coverage, which can be smaller than metadata's requested item count.

## Limits

The artifact samples specific models, tasks, and one debate protocol; directions
share models and traces. It is not a population sample of independent model
pairs. The outputs are recorded under historical endpoints and cannot guarantee
that present-day inference with a model of the same name reproduces them.
Reported token counts are an inference-volume proxy, not a price or latency
measurement. Benchmark licenses and model/provider terms are separate from the
license on the authors' original code and outputs.
