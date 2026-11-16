#!/usr/bin/env python3
"""Cross-direction relations refitted without the small-model diagnostic.

Every association and specification is recomputed on the eight directions that
remain when the 100-item diagnostic is dropped, with exact p over all
8! = 40,320 label permutations.

Output: results/sensitivity.csv
"""
from artifact_paths import RESULTS
import panels
import table_directions as td
import table_gate as tg
import table_reception as trc
import table_rates as tr
import table_specifications as ts
import traces

OUT = RESULTS / "sensitivity.csv"
DROP = "llama1b_gemma2b_gsm8k"


def main() -> None:
    gamma, headroom, rho, upsilon = [], [], [], []
    honest_gain, adv_cor, adv_cor_vs_homo = [], [], []

    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver") or d["key"] == DROP:
            continue
        support, loaded = td.matched_support(d)
        a0d = td.r0_accuracy(loaded["d_homo"], support)
        a0p = td.r0_accuracy(loaded["p_homo"], support)
        n_rho, k_rho = trc.exposures(loaded, support, "mixed", True, False)
        n_up, k_up = trc.exposures(loaded, support, "adv", False, True)

        gamma.append(a0p - a0d)
        headroom.append(100.0 - a0d)
        rho.append(k_rho / n_rho)
        upsilon.append(k_up / n_up)
        honest_gain.append(tg.net_gain(loaded, support, "mixed"))
        cor_adv = tr.transitions(loaded["adv"], support)[1]
        cor_homo = tr.transitions(loaded["d_homo"], support)[1]
        adv_cor.append(cor_adv)
        adv_cor_vs_homo.append(cor_adv - cor_homo)

    associations = [
        ("r(Gamma, honest net gain)", gamma, honest_gain),
        ("r(Gamma, adversarial corruption rate)", gamma, adv_cor),
        ("r(Gamma, corruption vs homogeneous)", gamma, adv_cor_vs_homo),
        ("r(Gamma, rho)", gamma, rho),
        ("r(H, rho)", headroom, rho),
        ("r(rho, upsilon)", rho, upsilon),
    ]

    rows = []
    for name, xs, ys in associations:
        rows.append({
            "quantity": name,
            "statistic": "r",
            "value": round(ts.pearson(xs, ys), 3),
            "exact_p": round(ts.exact_permutation_p(xs, ys), 4),
            "r_squared": "",
        })
        print(f"  {name:40s} r={rows[-1]['value']:+.3f}  p={rows[-1]['exact_p']:.4f}")

    gplus = [max(0.0, g) for g in gamma]
    specs = [
        ("Gamma+", gplus),
        ("rho", rho),
        ("Gamma+ x rho", [g * r for g, r in zip(gplus, rho)]),
    ]
    for name, xs in specs:
        slope = ts.ols_slope(xs, honest_gain)
        rows.append({
            "quantity": name,
            "statistic": "slope",
            "value": round(slope, 3),
            "exact_p": "",
            "r_squared": round(ts.r_squared(xs, honest_gain), 4),
        })
        print(f"  {name:40s} slope={slope:8.3f}  R2={rows[-1]['r_squared']:.4f}")

    traces.write_csv(OUT, ["quantity", "statistic", "value", "exact_p",
                           "r_squared"], rows)


if __name__ == "__main__":
    main()
