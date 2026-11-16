#!/usr/bin/env python3
"""Token accounting for honest heterogeneous debate versus direct peer inference.

The comparison is deliberately provider-price agnostic.  For every matched
replicate-item, it contrasts:

* debate: all 15 calls in the honest [R,R,P] panel (3 agents x 5 rounds);
* direct: one round-0 call to P.

The peer-alone traces usually contain three exchangeable agents.  Their R0
records are pooled to estimate both single-call accuracy and mean tokens per
call; the direct baseline is charged for one call, not all three.  Token counts
are the usage values reported in each trace.  Because providers/models may use
different tokenizers, sums across models are an inference-volume proxy rather
than a dollar-cost estimate.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


from artifact_paths import RESULTS

import panels  # noqa: E402
import traces  # noqa: E402


ROUNDS = range(5)
N_AGENTS = 3
PEER_SLOT = 2
OUT = RESULTS / "token_cost.csv"
SUMMARY = RESULTS / "token_cost_summary.md"


def load_path(path: Path, benchmark: str) -> dict:
    """Load and re-score a trace at an explicit path."""
    parse_out, parse_gold, match = traces.ADAPTERS[benchmark]
    records = {}
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["item_id"], int(record["round"]),
                   int(record["agent_id"]))
            records[key] = record
    for record in records.values():
        pred = parse_out(record.get("raw_output") or "")
        gold = parse_gold(record.get("gold_answer"))
        record["ok"] = bool(pred is not None and gold is not None
                            and match(pred, gold))
    return records


def released_run(name: str, benchmark: str) -> dict:
    return traces.load(name, benchmark)


def complete_mixed_items(records: dict) -> set[str]:
    return traces.complete_items(records, N_AGENTS, ROUNDS)


def complete_peer_r0_items(records: dict) -> set[str]:
    agents = sorted({agent for _item, _round, agent in records})
    items = {item for item, _round, _agent in records}
    return {
        item for item in items
        if all((item, 0, agent) in records for agent in agents)
    }


def token_pair(record: dict, context: str) -> tuple[int, int]:
    prompt = record.get("prompt_tokens")
    completion = record.get("completion_tokens")
    if not isinstance(prompt, (int, float)) or not isinstance(
            completion, (int, float)):
        raise ValueError(f"missing token usage for {context}")
    return int(prompt), int(completion)


def analyse(direction: dict) -> dict:
    benchmark = direction["benchmark"]
    mixed_runs = direction["mixed"]
    peer_runs = direction["p_homo"]
    explicit_mixed = direction.get("explicit_mixed")

    mixed_loaded = []
    for index, run in enumerate(mixed_runs):
        if explicit_mixed:
            mixed_loaded.append(load_path(explicit_mixed[index], benchmark))
        else:
            mixed_loaded.append(released_run(run, benchmark))
    peer_loaded = [released_run(run, benchmark) for run in peer_runs]

    panel_prompt = panel_completion = panel_calls = 0
    direct_prompt = direct_completion = direct_calls_observed = 0
    direct_correct = 0
    matched_items = 0
    panel_majority = {round_: 0 for round_ in ROUNDS}
    receiver_correct = {round_: 0 for round_ in ROUNDS}
    receiver_total = {round_: 0 for round_ in ROUNDS}

    for rep_index, mixed in enumerate(mixed_loaded):
        peer = peer_loaded[rep_index] if rep_index < len(peer_loaded) else peer_loaded[0]
        support = complete_mixed_items(mixed) & complete_peer_r0_items(peer)
        matched_items += len(support)
        peer_agents = sorted({agent for _item, _round, agent in peer})

        for item in support:
            for round_ in ROUNDS:
                round_correct = 0
                for agent in range(N_AGENTS):
                    record = mixed[(item, round_, agent)]
                    prompt, completion = token_pair(
                        record, f"{direction['label']} debate {item}/R{round_}/A{agent}")
                    panel_prompt += prompt
                    panel_completion += completion
                    panel_calls += 1
                    round_correct += int(record["ok"])
                    if agent != PEER_SLOT:
                        receiver_correct[round_] += int(record["ok"])
                        receiver_total[round_] += 1
                panel_majority[round_] += int(round_correct >= 2)

            for agent in peer_agents:
                record = peer[(item, 0, agent)]
                prompt, completion = token_pair(
                    record, f"{direction['label']} direct {item}/A{agent}")
                direct_prompt += prompt
                direct_completion += completion
                direct_calls_observed += 1
                direct_correct += int(record["ok"])

    if not matched_items:
        raise ValueError(f"no matched support for {direction['label']}")
    if panel_calls != matched_items * 15:
        raise AssertionError((direction["label"], panel_calls, matched_items))

    panel_prompt_item = panel_prompt / matched_items
    panel_completion_item = panel_completion / matched_items
    direct_prompt_call = direct_prompt / direct_calls_observed
    direct_completion_call = direct_completion / direct_calls_observed
    panel_total_item = panel_prompt_item + panel_completion_item
    direct_total_call = direct_prompt_call + direct_completion_call
    best_panel_round = max(ROUNDS, key=lambda r: panel_majority[r])
    best_receiver_round = max(
        ROUNDS, key=lambda r: receiver_correct[r] / receiver_total[r])

    return {
        "direction": direction["label"],
        "task": direction["task"],
        "fixed_receiver": int(bool(direction.get("fixed_receiver"))),
        "new_rebuttal_control": int(bool(direction.get("new_rebuttal_control"))),
        "matched_replicate_items": matched_items,
        "debate_calls_per_item": 15,
        "direct_calls_per_item": 1,
        "debate_prompt_tokens_per_item": round(panel_prompt_item, 1),
        "debate_completion_tokens_per_item": round(panel_completion_item, 1),
        "debate_total_tokens_per_item": round(panel_total_item, 1),
        "direct_prompt_tokens_per_call": round(direct_prompt_call, 1),
        "direct_completion_tokens_per_call": round(direct_completion_call, 1),
        "direct_total_tokens_per_call": round(direct_total_call, 1),
        "prompt_token_ratio": round(panel_prompt_item / direct_prompt_call, 2),
        "completion_token_ratio": round(
            panel_completion_item / direct_completion_call, 2),
        "total_token_ratio": round(panel_total_item / direct_total_call, 2),
        "direct_peer_R0_accuracy": round(
            100 * direct_correct / direct_calls_observed, 1),
        "debate_best_majority_accuracy": round(
            100 * panel_majority[best_panel_round] / matched_items, 1),
        "debate_best_majority_round": f"R{best_panel_round}",
        "debate_best_receiver_accuracy": round(
            100 * receiver_correct[best_receiver_round]
            / receiver_total[best_receiver_round], 1),
        "debate_best_receiver_round": f"R{best_receiver_round}",
    }


def add_qwen72_control(directions: list[dict]) -> None:
    directions.append({
        "key": "qwen7b_qwen72b_math_rebuttal",
        "label": "Qwen-7B <- Qwen-72B",
        "task": "MATH-hard-fx",
        "benchmark": "math",
        "fixed_receiver": True,
        "new_rebuttal_control": True,
        "mixed": [
            f"het_2qw7b1qw72_honest_hard_5round_rep{rep}"
            for rep in (1, 2, 3)
        ],
        "p_homo": [
            f"qwen72b_homo_e1_math_hard_5round_rep{rep}"
            for rep in (1, 2, 3)
        ],
    })


def write_summary(rows: list[dict]) -> None:
    gamma_by_label = {
        row["direction"]: float(row["Gamma"])
        for row in csv.DictReader(
            (RESULTS / "directions.csv").open())
    }
    gamma_by_label["Qwen-7B <- Qwen-72B"] = 7.082
    positive = [row for row in rows if gamma_by_label[row["direction"]] > 0]
    ratios = [float(row["total_token_ratio"]) for row in positive]
    prompt_ratios = [float(row["prompt_token_ratio"]) for row in positive]
    completion_ratios = [float(row["completion_token_ratio"]) for row in positive]
    panel_wins = sum(
        float(row["debate_best_majority_accuracy"])
        > float(row["direct_peer_R0_accuracy"])
        for row in positive
    )

    lines = [
        "# Token-cost comparison",
        "",
        "Each debate value charges all 15 calls in the honest three-agent, "
        "five-round panel. The direct baseline charges one round-0 call to "
        "the peer on the same matched items. Peer-only panels with three "
        "exchangeable agents are pooled to estimate the mean cost and "
        "accuracy of one call.",
        "",
        "Token counts are model/provider-reported usage. They are an "
        "inference-volume proxy, not a dollar-cost estimate, because the "
        "panels mix model-specific tokenizers and serving stacks.",
        "",
        "## Headline",
        "",
        f"Across the {len(positive)} positive-gap directions (including the "
        "additional same-family control), debate uses "
        f"{min(ratios):.1f}–{max(ratios):.1f}x as many total reported tokens "
        f"as one peer call (median {statistics.median(ratios):.1f}x). "
        f"Completion-token multiples are {min(completion_ratios):.1f}–"
        f"{max(completion_ratios):.1f}x (median "
        f"{statistics.median(completion_ratios):.1f}x).",
        "",
        f"The raw call-count multiple is 15x. Full-transcript replay makes "
        f"the prompt-token multiple much larger: {min(prompt_ratios):.1f}–"
        f"{max(prompt_ratios):.1f}x (median "
        f"{statistics.median(prompt_ratios):.1f}x). The protocol also has "
        "five sequential generation stages rather than one, although the "
        "traces do not support an empirical wall-clock latency comparison.",
        "",
        f"The best observed debate majority exceeds the direct peer's "
        f"single-pass R0 accuracy in {panel_wins}/{len(positive)} positive-gap "
        "directions. This is an optimistic debate comparison because it "
        "selects the best of five rounds after observing outcomes; a deployed "
        "fixed-round protocol cannot make that selection without validation.",
        "",
        "## Per-direction values",
        "",
        "| Direction | n | Debate tokens/item | Direct tokens/call | Multiple | "
        "Direct acc. | Best majority acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['direction']} | {row['matched_replicate_items']} | "
            f"{float(row['debate_total_tokens_per_item']):,.0f} | "
            f"{float(row['direct_total_tokens_per_call']):,.0f} | "
            f"{float(row['total_token_ratio']):.1f}x | "
            f"{row['direct_peer_R0_accuracy']} | "
            f"{row['debate_best_majority_accuracy']} "
            f"({row['debate_best_majority_round']}) |"
        )
    lines.extend([
        "",
        "The accuracy columns are descriptive context, not a controlled "
        "price-performance frontier: direct accuracy is a single R0 call, "
        "whereas the debate column is the ex-post best majority round.",
        "",
    ])
    SUMMARY.write_text("\n".join(lines))


def main() -> None:
    directions = [dict(direction) for direction in panels.DIRECTIONS]
    add_qwen72_control(directions)
    rows = [analyse(direction) for direction in directions]
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
