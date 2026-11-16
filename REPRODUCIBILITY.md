# Reproduction and analysis populations

`python scripts/reproduce.py --output reproduced` verifies the data manifest,
regenerates all 30 result CSVs, renders four figures, and runs the strict verifier.
It is sequential and requires no inference. Use a fresh output directory.

## Populations

- Original primary analysis: nine directions, unchanged by the camera-ready additions.
- Original confirmatory analysis: two fixed-Qwen directions, excluded from the original fits.
- Additional same-family control: Qwen-7B receiver/Qwen-72B peer, three replicates;
  separately generated in `same_family.py`. Its first-exchange support omits an
  adversarial arm because none was run. This condition never enters the original fits.
- MuSiQue scope probe: six panels on 300 questions, kept separate from primary fits.
- Token analysis: twelve directions, including the added control; the headline
  selects nine positive-gap directions using homogeneous-reference gaps.
  Each complete mixed-panel replicate-item charges all 15 calls. Peer-reference
  R0 agents estimate mean accuracy and cost of a single standalone peer call.
  Debate accuracy selects the best observed majority round retrospectively.

`same_family_ladder.csv`, `same_family_discordant_flows.csv`, `token_cost.csv`,
and `release_quantities.csv` extend the original 26 CSV references. The
`release_claims.csv` manifest freezes the added numbers independently of their
computation, following the original quantity-addressed claim checker.

## Determinism and verification

- Answers are reparsed; stored correctness is not used as ground truth.
- Resumed duplicates follow last-record-wins indexing.
- Item clustering and sorted cluster order are retained from the submitted code.
- Bootstrap seed is 260719826, with 4,000 Poisson draws.
- Exact direction permutations and original regression definitions are unchanged.
- Result CSVs must match the frozen references byte-for-byte.
- Figure PDFs must be valid and their plot data must agree with the numerical
  references. PDF bytes may differ across font/rendering environments.
- Every distributed JSONL and metadata file has a SHA-256 manifest entry.

`BORROWED_STRENGTH_TRACES` can point to an externally extracted trace directory.
The default is `traces/`. `--output` isolates generated outputs and rejects the
repository root and protected source/data/reference directories. The manifest
uses canonical `traces/...` names even when the data directory is overridden.

The reproduction environment is frozen in `requirements-lock.txt`. The
repository does not call model APIs or attempt to reproduce historical inference.
