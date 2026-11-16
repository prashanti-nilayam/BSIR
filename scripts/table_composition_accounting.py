#!/usr/bin/env python3
"""Fixed-round item accounting for the five full-reference honest panels.

At R4, count panel-only wins (at least two members correct while the peer is
wrong), peer-only wins, newly correct panels (no member correct at R0), and
discarded initial availability (some member correct at R0 but the panel wrong
at R4).  The fixed final round avoids selecting a round after observing these
item-level contrasts.

Output: results/composition_accounting.csv
"""
from __future__ import annotations

from artifact_paths import RESULTS
import panels
import table_rounds
import traces

OUT = RESULTS / "composition_accounting.csv"
ROUND = 4
N_AGENTS = 3
ROUNDS = range(5)


def _supports(direction: dict, mixed: list[dict], p_homo: list[dict]) -> list[set]:
    support = []
    for index, records in enumerate(mixed):
        current = traces.complete_items(records, N_AGENTS, ROUNDS)
        peer_records = p_homo[index] if index < len(p_homo) else p_homo[0]
        peer_agents = {agent for (_, _, agent) in peer_records}
        peer_rounds = {rd for (_, rd, _) in peer_records}
        if len(peer_agents) == N_AGENTS and set(ROUNDS) <= peer_rounds:
            current &= traces.complete_items(peer_records, N_AGENTS, ROUNDS)
        else:
            current &= {item for (item, _, _) in peer_records}
        support.append(current)
    return support


def _pct(value: int, total: int) -> float:
    return 100.0 * value / total if total else float("nan")


def _row(label: str, direction: dict) -> dict:
    mixed = [traces.load(run, direction["benchmark"]) for run in direction["mixed"]]
    p_homo = [traces.load(run, direction["benchmark"]) for run in direction["p_homo"]]
    support = _supports(direction, mixed, p_homo)
    counts = {
        "panel_correct": 0,
        "peer_correct": 0,
        "panel_only": 0,
        "peer_only": 0,
        "new_correct": 0,
        "discarded_initial": 0,
    }
    total = 0
    for records, items in zip(mixed, support):
        for item in items:
            r0 = [bool(records[(item, 0, slot)]["ok"]) for slot in range(N_AGENTS)]
            r4 = [bool(records[(item, ROUND, slot)]["ok"]) for slot in range(N_AGENTS)]
            panel = sum(r4) >= 2
            peer = r4[panels.PEER_SLOT]
            initial_oracle = any(r0)
            counts["panel_correct"] += int(panel)
            counts["peer_correct"] += int(peer)
            counts["panel_only"] += int(panel and not peer)
            counts["peer_only"] += int(peer and not panel)
            counts["new_correct"] += int(panel and not initial_oracle)
            counts["discarded_initial"] += int((not panel) and initial_oracle)
            total += 1

    net = counts["panel_only"] - counts["peer_only"]
    if net != counts["panel_correct"] - counts["peer_correct"]:
        raise ValueError(f"panel/peer accounting failed for {direction['key']}")
    return {
        "setting": label,
        "round": f"R{ROUND}",
        "n": total,
        "panel_accuracy": round(_pct(counts["panel_correct"], total), 2),
        "peer_accuracy": round(_pct(counts["peer_correct"], total), 2),
        "panel_only": round(_pct(counts["panel_only"], total), 2),
        "peer_only": round(_pct(counts["peer_only"], total), 2),
        "net_vs_peer": round(_pct(net, total), 2),
        "new_correct": round(_pct(counts["new_correct"], total), 2),
        "discarded_initial": round(_pct(counts["discarded_initial"], total), 2),
    }


def main() -> None:
    rows = []
    for label, key in table_rounds.SETTINGS:
        direction = next(d for d in panels.DIRECTIONS if d["key"] == key)
        rows.append(_row(label, direction))
    for row in rows:
        print(
            f"  {row['setting']:30s} n={row['n']:5d} "
            f"panel={row['panel_accuracy']:5.2f} peer={row['peer_accuracy']:5.2f} "
            f"panel-only={row['panel_only']:5.2f} peer-only={row['peer_only']:5.2f} "
            f"net={row['net_vs_peer']:+6.2f} new={row['new_correct']:5.2f} "
            f"discarded={row['discarded_initial']:5.2f}"
        )
    traces.write_csv(OUT, list(rows[0]), rows)


if __name__ == "__main__":
    main()
