#!/usr/bin/env python3
"""Evaluated directions on matched R0/R1 support.

Per direction: matched replicate-item count, homogeneous-panel R0 accuracies
for defender and peer, the capability gap, and defender headroom.

  A0_D = R0 accuracy of the defender slots in [D,D,D]
  A0_P = R0 accuracy of the peer slots in [P,P,P]
  Gamma = A0_P - A0_D
  H     = 100 - A0_D

Output: results/directions.csv
"""
from artifact_paths import RESULTS
import panels
import traces

ROUNDS = range(2)          # matched support is defined on R0 and R1
N_AGENTS = 3
OUT = RESULTS / "directions.csv"


def load_panel(runs, benchmark):
    """[(replicate_index, records)] for each replicate of one panel."""
    return [(i, traces.load(run, benchmark)) for i, run in enumerate(runs, 1)]


def matched_support(direction):
    """{replicate: set(item)} present with all agents in R0+R1 of all panels."""
    bench = direction["benchmark"]
    loaded = {p: load_panel(direction[p], bench)
              for p in ("d_homo", "p_homo", "mixed", "adv")}
    support = {}
    n_reps = max(len(v) for v in loaded.values())
    for rep in range(1, n_reps + 1):
        sets = []
        # The three contrast panels must be complete in R0 and R1.
        for panel in ("d_homo", "mixed", "adv"):
            reps = loaded[panel]
            recs = dict(reps).get(rep) or reps[0][1]
            sets.append(traces.complete_items(recs, N_AGENTS, ROUNDS))
        # The peer reference panel only bounds which items have a peer-alone
        # measurement at all; it is not itself a three-agent debate panel (the
        # small-model diagnostic's reference is a 100-item, single-agent R0
        # capture), so it constrains by item presence rather than completeness.
        p_reps = dict(loaded["p_homo"])
        p_recs = p_reps.get(rep) or loaded["p_homo"][0][1]
        sets.append({i for (i, _, _) in p_recs})
        support[rep] = set.intersection(*sets)
    return support, loaded


def _is_single_agent(records):
    return {a for (_, _, a) in records} == {0}


def r0_accuracy(loaded_panel, support):
    """Pooled R0 accuracy over every agent of a homogeneous panel."""
    hit = tot = 0
    reps = dict(loaded_panel)
    for rep, items in support.items():
        recs = reps.get(rep) or loaded_panel[0][1]
        slots = sorted({a for (_, _, a) in recs})
        for item in items:
            for s in slots:
                r = recs.get((item, 0, s))
                if r is not None:
                    hit += int(r["ok"])
                    tot += 1
    return 100.0 * hit / tot if tot else float("nan")


def main() -> None:
    rows = []
    for d in panels.DIRECTIONS:
        support, loaded = matched_support(d)
        n_items = sum(len(v) for v in support.values())
        a0d = r0_accuracy(loaded["d_homo"], support)
        a0p = r0_accuracy(loaded["p_homo"], support)
        rows.append({
            "direction": d["label"], "task": d["task"],
            "replicate_items": n_items,
            "A0_D": round(a0d, 1), "A0_P": round(a0p, 1),
            "Gamma": round(a0p - a0d, 1), "H": round(100.0 - a0d, 1),
            "fixed_receiver": int(bool(d.get("fixed_receiver"))),
        })
        print(f"  {d['label']:28s} n={n_items:5d}  A0_D={a0d:5.1f}  "
              f"A0_P={a0p:5.1f}  G={a0p-a0d:+6.1f}  H={100-a0d:5.1f}")

    traces.write_csv(OUT, ["direction", "task", "replicate_items", "A0_D",
                           "A0_P", "Gamma", "H", "fixed_receiver"], rows)


if __name__ == "__main__":
    main()
