#!/usr/bin/env python3
"""Per-direction R0->R1 transition rates for defender slots.

  recovery   = P(R1 correct | R0 wrong)
  corruption = P(R1 wrong   | R0 correct)

reported in each of the three matched panels: the defender's homogeneous
panel, the honest heterogeneous panel, and the adversarial one.

Output: results/rates.csv
"""
from artifact_paths import RESULTS
import panels
import table_directions as td
import traces

OUT = RESULTS / "rates.csv"


def transitions(loaded_panel, support):
    """(recovery %, corruption %) over defender slots on the matched support."""
    rec_hit = rec_tot = cor_hit = cor_tot = 0
    reps = dict(loaded_panel)
    for rep, items in support.items():
        recs = reps.get(rep) or loaded_panel[0][1]
        for item in items:
            for s in panels.DEFENDER_SLOTS:
                r0 = recs.get((item, 0, s))
                r1 = recs.get((item, 1, s))
                if r0 is None or r1 is None:
                    continue
                if r0["ok"]:
                    cor_tot += 1
                    cor_hit += int(not r1["ok"])
                else:
                    rec_tot += 1
                    rec_hit += int(r1["ok"])
    pct = lambda h, t: 100.0 * h / t if t else float("nan")
    return pct(rec_hit, rec_tot), pct(cor_hit, cor_tot)


def main() -> None:
    rows = []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue                      # discovery set only
        support, loaded = td.matched_support(d)
        vals = {p: transitions(loaded[p], support)
                for p in ("d_homo", "mixed", "adv")}
        rows.append({
            "direction": d["label"],
            "recovery_homog": round(vals["d_homo"][0], 1),
            "recovery_honest": round(vals["mixed"][0], 1),
            "recovery_adv": round(vals["adv"][0], 1),
            "corruption_homog": round(vals["d_homo"][1], 1),
            "corruption_honest": round(vals["mixed"][1], 1),
            "corruption_adv": round(vals["adv"][1], 1),
        })
        r = rows[-1]
        print(f"  {d['label']:28s} rec {r['recovery_homog']:5.1f} "
              f"{r['recovery_honest']:5.1f} {r['recovery_adv']:5.1f}   "
              f"cor {r['corruption_homog']:5.1f} {r['corruption_honest']:5.1f} "
              f"{r['corruption_adv']:5.1f}")

    traces.write_csv(OUT, list(rows[0].keys()), rows)


if __name__ == "__main__":
    main()
