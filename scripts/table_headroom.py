#!/usr/bin/env python3
"""Honest uplift against available headroom, on full-trajectory support.

Support here is stricter than Table directions: every one of R0-R4 must be
present in the defender-homogeneous panel, the honest heterogeneous panel, and
the peer's own panel. Counts therefore differ slightly from the R0/R1 support.

  mixed_best      best round of defender accuracy in [D,D,P], with its round
  controlled_peak max over rounds of (mixed accuracy - homogeneous accuracy)
  gap_closed      (mixed_best - defender A0) / (peer A0 - defender A0),
                  reported only where the peer holds an advantage

Output: results/headroom.csv
"""
from artifact_paths import RESULTS
import panels
import table_gate as tg
import traces

ROUNDS = range(5)
N_AGENTS = 3
OUT = RESULTS / "headroom.csv"


def full_support(direction):
    bench = direction["benchmark"]
    d_homo = [traces.load(r, bench) for r in direction["d_homo"]]
    mixed = [traces.load(r, bench) for r in direction["mixed"]]
    p_homo = [traces.load(r, bench) for r in direction["p_homo"]]
    support = {}
    for i in range(len(mixed)):
        dh = d_homo[i] if i < len(d_homo) else d_homo[0]
        ph = p_homo[i] if i < len(p_homo) else p_homo[0]
        sup = (traces.complete_items(dh, N_AGENTS, ROUNDS)
               & traces.complete_items(mixed[i], N_AGENTS, ROUNDS))
        p_agents = {a for (_, _, a) in ph}
        p_rounds = {rd for (_, rd, _) in ph}
        if len(p_agents) == N_AGENTS and set(ROUNDS) <= p_rounds:
            sup &= traces.complete_items(ph, N_AGENTS, ROUNDS)
        else:
            sup &= {it for (it, _, _) in ph}
        support[i + 1] = sup
    return support, {"d_homo": list(enumerate(d_homo, 1)),
                     "mixed": list(enumerate(mixed, 1)),
                     "p_homo": list(enumerate(p_homo, 1))}


def all_agent_accuracy(loaded_panel, support, rnd):
    hit = tot = 0
    reps = dict(loaded_panel)
    for rep, items in support.items():
        recs = reps.get(rep) or loaded_panel[0][1]
        slots = sorted({a for (_, _, a) in recs})
        for item in items:
            for s in slots:
                r = recs.get((item, rnd, s))
                if r is not None:
                    hit += int(r["ok"])
                    tot += 1
    return 100.0 * hit / tot if tot else float("nan")


def main() -> None:
    rows = []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = full_support(d)
        n_items = sum(len(v) for v in support.values())

        a0_def = all_agent_accuracy(loaded["d_homo"], support, 0)
        a0_peer = all_agent_accuracy(loaded["p_homo"], support, 0)

        mixed_by_round = {rd: tg.accuracy(loaded["mixed"], support, rd)
                          for rd in ROUNDS}
        # Every agent of [D,D,D] is a defender, so its baseline pools all
        # three; the heterogeneous panel contributes only its defender slots.
        homo_by_round = {rd: all_agent_accuracy(loaded["d_homo"], support, rd)
                         for rd in ROUNDS}
        best_rd = max(ROUNDS, key=lambda rd: mixed_by_round[rd])
        controlled_peak = max(mixed_by_round[rd] - homo_by_round[rd]
                              for rd in ROUNDS)

        gamma = a0_peer - a0_def
        # GPT-4.1 <- OSS-120B lands on exactly 127.5000 here. The manuscript
        # prints 127; half-up and half-to-even both give 128. Left as computed.
        gap_closed = ((mixed_by_round[best_rd] - a0_def) / gamma * 100.0
                      if gamma > 0 else None)

        rows.append({
            "direction": d["label"],
            "replicate_items": n_items,
            "def_A0": round(a0_def, 1),
            "peer_A0": round(a0_peer, 1),
            "mixed_best": round(mixed_by_round[best_rd], 1),
            "best_round": f"R{best_rd}",
            "controlled_peak": round(controlled_peak, 1),
            "gap_closed_pct": "" if gap_closed is None else round(gap_closed, 2),
        })
        r = rows[-1]
        print(f"  {d['label']:28s} n={n_items:5d} defA0={r['def_A0']:5.1f} "
              f"peerA0={r['peer_A0']:5.1f} best={r['mixed_best']:5.1f} "
              f"({r['best_round']}) peak={r['controlled_peak']:+5.1f} "
              f"closed={r['gap_closed_pct']}")

    traces.write_csv(OUT, list(rows[0].keys()), rows)


if __name__ == "__main__":
    main()
