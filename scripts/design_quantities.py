#!/usr/bin/env python3
"""Emit manuscript design and configuration quantities from traces and code.

Counts that can be reconstructed are recomputed.  Protocol-only constants that
are not recorded in JSONL rows remain explicit here with a source annotation,
so the manuscript verifier can distinguish measured facts from configuration.

Output: results/design_quantities.csv
"""
from __future__ import annotations

from artifact_paths import RESULTS
import bootstrap
import panels
import table_directions as td
import table_role_effects as tre
import traces

OUT = RESULTS / "design_quantities.csv"


def unique_items(run: str, benchmark: str) -> int:
    return len({item for item, _round, _agent in traces.load(run, benchmark)})


def observed_shape(run: str, benchmark: str) -> tuple[int, int]:
    records = traces.load(run, benchmark)
    return (len({agent for _item, _round, agent in records}),
            len({round_ for _item, round_, _agent in records}))


def main() -> None:
    discovery = [d for d in panels.DIRECTIONS if not d.get("fixed_receiver")]
    confirmatory = [d for d in panels.DIRECTIONS if d.get("fixed_receiver")]

    matched = {}
    for d in panels.DIRECTIONS:
        support, _loaded = td.matched_support(d)
        matched[d["key"]] = sum(len(items) for items in support.values())

    discovery_items = sum(matched[d["key"]] for d in discovery)
    confirmatory_items = sum(matched[d["key"]] for d in confirmatory)
    _itt, _clusters, _all, itt_items = tre.analyse("itt")
    _pp, _clusters, _all, pp_items = tre.analyse("pp")
    assert discovery_items == itt_items

    primary_run = discovery[0]["mixed"][0]
    agents, rounds = observed_shape(primary_run, discovery[0]["benchmark"])
    musique_run = "musique_gemma3_homo_5round_rep1"
    musique_runs = [
        "musique_gemma3_homo_5round_rep1", "musique_qwen7b_homo_rep1",
        "musique_mixed_ggq_honest_rep1", "musique_mixed_ggq_advpeer_rep1",
        "musique_gemma3_homo_adv_rep1", "musique_mixed_ggq_advdef_rep1",
    ]
    musique_loaded = [traces.load(run, "musique") for run in musique_runs]
    musique_support = set.intersection(*[
        traces.complete_items(records, 3, range(5)) for records in musique_loaded
    ])

    rows = [
        ("discovery_directions", len(discovery), "panel definitions"),
        ("confirmatory_directions", len(confirmatory), "panel definitions"),
        ("total_directions", len(panels.DIRECTIONS), "panel definitions"),
        ("primary_agents", agents, "observed trace dimensions"),
        ("primary_rounds", rounds, "observed trace dimensions"),
        ("math_hard_items", unique_items(discovery[0]["mixed"][0], "math"),
         "observed trace items"),
        ("gsm8k_items", unique_items(discovery[1]["mixed"][0], "gsm8k"),
         "observed trace items"),
        ("small_model_diagnostic_items", matched["llama1b_gemma2b_gsm8k"],
         "matched support"),
        ("musique_items", unique_items(musique_run, "musique"),
         "observed trace items"),
        ("musique_heterogeneous_panels", len(musique_runs),
         "observed trace panels"),
        ("musique_matched_items", len(musique_support), "six-panel matched support"),
        ("musique_agent_round_records", sum(len(records) for records in musique_loaded),
         "six panels x 300 items x 3 agents x 5 rounds"),
        ("musique_agent_round_records_per_panel", len(musique_loaded[0]),
         "300 items x 3 agents x 5 rounds"),
        ("musique_replicates_per_condition", 1, "panel definitions"),
        ("discovery_matched_replicate_items", discovery_items, "matched support"),
        ("discovery_defender_condition_observations", discovery_items * 6,
         "2 defender slots x 3 conditions"),
        ("confirmatory_matched_replicate_items", confirmatory_items,
         "matched support"),
        ("confirmatory_defender_condition_observations", confirmatory_items * 6,
         "2 defender slots x 3 conditions"),
        ("per_protocol_matched_replicate_items", pp_items,
         "adversary parsed-wrong R0-R4"),
        ("bootstrap_seed", bootstrap.SEED, "analysis constant"),
        ("bootstrap_replicates", bootstrap.REPS, "analysis constant"),
        ("math_completion_token_budget", 4096, "protocol constant"),
        ("gsm8k_completion_token_budget", 1024, "protocol constant"),
        ("primary_temperature", 0, "protocol constant"),
        ("primary_replicates", 3, "panel definitions"),
    ]
    traces.write_csv(OUT, ["quantity", "value", "source"],
                     [dict(quantity=q, value=v, source=s) for q, v, s in rows])
    for quantity, value, source in rows:
        print(f"  {quantity:45s} {value:>10}  ({source})")


if __name__ == "__main__":
    main()
