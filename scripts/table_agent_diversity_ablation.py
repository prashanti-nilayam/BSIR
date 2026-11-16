#!/usr/bin/env python3
"""Same-model diversity control on MATH-hard.

Compares a Gemma-3-12B panel whose three agents use careful/critical/concise
prompt emphases at temperatures 0/.3/.7 against a uniform panel sharing one
prompt at temperature 0. Both use seeds 0, 1, 2 on the same items.

Three per-round quantities:
  pooled_agent   -- accuracy over every agent response
  at_least_two   -- share of items where >=2 of 3 agents are correct
  any_agent      -- share of items where >=1 agent is correct (oracle)

Output: results/agent_diversity_ablation.csv
"""
from artifact_paths import RESULTS
import traces

ROUNDS = range(5)
N_AGENTS = 3
PANELS = {
    "diverse": [f"hetcfg_gemma3_homo_math_5round_rep{r}" for r in (1, 2, 3)],
    "uniform": [f"honest_gemma3_homo_math_5round_rep{r}" for r in (1, 2, 3)],
}
OUT = RESULTS / "agent_diversity_ablation.csv"


def panel_rates(runs: list[str]) -> tuple[dict, int]:
    """Per-round rates pooled across replicates, on items complete in all rounds."""
    agent_hits = {rd: [0, 0] for rd in ROUNDS}       # [correct, total]
    two_hits = {rd: [0, 0] for rd in ROUNDS}
    any_hits = {rd: [0, 0] for rd in ROUNDS}
    n_items = 0

    for run in runs:
        records = traces.load(run, "math")
        grouped = traces.by_item_round(records)
        items = sorted(traces.complete_items(records, N_AGENTS, ROUNDS))
        n_items += len(items)

        for item in items:
            for rd in ROUNDS:
                agents = grouped[(item, rd)]
                correct = sum(1 for a in agents.values() if a["ok"])
                agent_hits[rd][0] += correct
                agent_hits[rd][1] += N_AGENTS
                two_hits[rd][0] += int(correct >= 2)
                two_hits[rd][1] += 1
                any_hits[rd][0] += int(correct >= 1)
                any_hits[rd][1] += 1

    pct = lambda p: 100.0 * p[0] / p[1] if p[1] else float("nan")
    return {
        "pooled_agent": {rd: pct(agent_hits[rd]) for rd in ROUNDS},
        "at_least_two": {rd: pct(two_hits[rd]) for rd in ROUNDS},
        "any_agent": {rd: pct(any_hits[rd]) for rd in ROUNDS},
    }, n_items


def main() -> None:
    rows = []
    for panel, runs in PANELS.items():
        rates, n_items = panel_rates(runs)
        for metric in ("pooled_agent", "at_least_two", "any_agent"):
            row = {"panel": panel, "metric": metric, "replicate_items": n_items}
            for rd in ROUNDS:
                row[f"R{rd}"] = round(rates[metric][rd], 2)
            rows.append(row)

    fields = ["panel", "metric", "replicate_items"] + [f"R{rd}" for rd in ROUNDS]
    traces.write_csv(OUT, fields, rows)
    for r in rows:
        print(f"  {r['panel']:8s} {r['metric']:13s} "
              + "  ".join(f"{r[f'R{rd}']:6.2f}" for rd in ROUNDS))


if __name__ == "__main__":
    main()
