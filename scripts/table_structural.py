#!/usr/bin/env python3
"""Structural accounting: capability gap vs. the mass it can act on.

Eligible mass is the share of defender exposures where the defender is wrong
at R0 and the honest peer is correct -- the states in which transfer is even
possible. Where that mass approaches Gamma, the product fit is close to an
accounting identity, so both are reported side by side.

Output: results/structural.csv
"""
from artifact_paths import RESULTS
import panels
import table_directions as td
import table_reception as trc
import traces

OUT = RESULTS / "structural.csv"


def main() -> None:
    rows = []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(d)
        gamma = (td.r0_accuracy(loaded["p_homo"], support)
                 - td.r0_accuracy(loaded["d_homo"], support))

        n_elig, _ = trc.exposures(loaded, support, "mixed", True, False)
        total = sum(len(items) for items in support.values()) \
            * len(panels.DEFENDER_SLOTS)
        mass = 100.0 * n_elig / total

        rows.append({
            "direction": d["label"],
            "Gamma": round(gamma, 1),
            "eligible_mass": round(mass, 1),
            "difference": round(mass - gamma, 1),
        })
        r = rows[-1]
        print(f"  {d['label']:28s} G={r['Gamma']:+6.1f}  mass={r['eligible_mass']:5.1f}"
              f"  diff={r['difference']:+6.1f}")

    traces.write_csv(OUT, ["direction", "Gamma", "eligible_mass", "difference"], rows)


if __name__ == "__main__":
    main()
