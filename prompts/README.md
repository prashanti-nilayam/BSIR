# Prompt templates and provenance

The text files contain the honest initial/revision templates for MATH, GSM8K,
and MuSiQue, plus the committed-wrong adversarial system templates and the older Alex
template referenced by one diagnostic run. Curly-brace
placeholders stand for benchmark question text, reference answers, wrong targets,
or other agents' complete prior-round responses; no benchmark examples are
inserted. The templates are for inspection, not format-string evaluation with
unescaped TeX braces.

`source/` preserves the prompt-construction source used in the working project:
benchmark adapters, debate message/retry construction, prompt-diversity variants,
and adversary target/initial-message construction. File hashes are in
`source/SHA256.json`. These snapshots document dynamic prompts and are not a
standalone inference package: their original runner imports are intentionally
not vendored into the analysis environment. The analysis scripts do not import
them or require API credentials.

The ordinary debate framing follows Du et al.'s multi-agent debate implementation
(https://github.com/composable-models/llm_multiagent_debate). The project source
attributes the original prompt text. Benchmark-specific answer contracts and
adversarial/diversity constructions are visible in these snapshots. Metadata
identify which recorded runs use the diversity variants, adversarial template,
per-agent temperatures, and retry settings.

Revision prompts show the two peer responses in full. The source also preserves
the retained own-history/retry behavior. These templates and recorded metadata
support auditing the protocol; no complete per-call message-history transcript
was recorded in the released JSONL schema.
