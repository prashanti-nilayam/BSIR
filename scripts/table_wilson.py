#!/usr/bin/env python3
"""Reception power with Wilson score intervals.

The Wilson interval is closed-form and reproduces exactly. The manuscript also
reports an item-clustered Poisson bootstrap interval; that resampling depends
on a random seed which the manuscript does not state, so it cannot be
reproduced from the traces alone and is left blank here. See NOTES in the
printed output.

Output: results/wilson.csv
"""
import math

from artifact_paths import RESULTS
import bootstrap
import panels
import table_directions as td
import table_reception as trc
import traces

Z = 1.959963984540054          # two-sided 95% normal quantile
OUT = RESULTS / "wilson.csv"


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100.0 * (centre - half), 100.0 * (centre + half)


def main() -> None:
    rows = []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(d)
        n_rho, k_rho = trc.exposures(loaded, support, "mixed", True, False)
        lo, hi = wilson(k_rho, n_rho)
        by_cluster = trc.exposures_by_cluster(loaded, support, "mixed",
                                              d["benchmark"], True, False)
        blo, bhi = bootstrap.ratio_interval(by_cluster, list(by_cluster))
        rows.append({
            "direction": d["label"],
            "rho": round(100.0 * k_rho / n_rho, 1),
            "n": n_rho,
            "wilson_lo": round(lo, 1),
            "wilson_hi": round(hi, 1),
            "bootstrap_lo": round(blo, 1),
            "bootstrap_hi": round(bhi, 1),
        })
        r = rows[-1]
        print(f"  {d['label']:28s} rho={r['rho']:5.1f}  n={r['n']:4d}  "
              f"Wilson [{r['wilson_lo']:5.1f}, {r['wilson_hi']:5.1f}]  "
              f"boot [{r['bootstrap_lo']:5.1f}, {r['bootstrap_hi']:5.1f}]")

    traces.write_csv(OUT, list(rows[0].keys()), rows)
    print("\n  Bootstrap: item-clustered Poisson, seed "
          f"{bootstrap.SEED}, {bootstrap.REPS} replicates, weights assigned in"
          "\n  sorted cluster order so the interval is a function of the seed"
          "\n  and the data alone.")


if __name__ == "__main__":
    main()
