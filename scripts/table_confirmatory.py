#!/usr/bin/env python3
"""Fixed-Qwen-7B MATH-hard ladder.

Qwen-7B is held as defender while the peer moves from none ([Q,Q,Q]) to
Gemma-3-12B to GPT-OSS-120B. All quantities are measured on the support
matched across the three panel families.

  Gamma        peer capability advantage over the Qwen defender (pp)
  rho          reception power, honest peer
  R1/R4 gain   controlled accuracy change vs. [Q,Q,Q] at that round
  R1 recovery  P(R1 correct | R0 wrong), defender slots
  R1 honest corruption  P(R1 wrong | R0 correct), honest panel
  upsilon      adversarial uptake

Output: results/confirmatory.csv
"""
from artifact_paths import RESULTS
import bootstrap
import panels
import table_directions as td
import table_gate as tg
import table_rates as tr
import table_reception as trc
import traces

OUT = RESULTS / "confirmatory.csv"
LADDER = [("Qwen", None), ("Gemma", "qwen7b_gemma12b_math"),
          ("OSS-120B", "qwen7b_gptoss_math")]


def full_round_support(direction, rounds=range(5)):
    """Items complete in all four panels across every round."""
    bench = direction["benchmark"]
    support = {}
    loaded = {p: [traces.load(r, bench) for r in direction[p]]
              for p in ("d_homo", "mixed", "adv", "p_homo")}
    n_reps = len(loaded["mixed"])
    for i in range(n_reps):
        sets = [traces.complete_items(loaded[p][i if i < len(loaded[p]) else 0],
                                      3, rounds)
                for p in ("d_homo", "mixed", "adv", "p_homo")]
        support[i + 1] = set.intersection(*sets)
    return support


def compute() -> dict:
    """Return full-precision ladder quantities keyed by displayed peer."""
    cols = {}
    for label, key in LADDER:
        if key is None:
            continue
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        support, loaded = td.matched_support(d)

        a0d = td.r0_accuracy(loaded["d_homo"], support)
        a0p = td.r0_accuracy(loaded["p_homo"], support)
        n_rho, k_rho = trc.exposures(loaded, support, "mixed", True, False)
        n_up, k_up = trc.exposures(loaded, support, "adv", False, True)
        br = trc.exposures_by_cluster(loaded, support, "mixed",
                                      d["benchmark"], True, False)
        bu = trc.exposures_by_cluster(loaded, support, "adv",
                                      d["benchmark"], False, True)
        rho_lo, rho_hi = bootstrap.ratio_interval(br, list(br))
        ups_lo, ups_hi = bootstrap.ratio_interval(bu, list(bu))

        r1_gain = tg.net_gain(loaded, support, "mixed")
        # R4 needs every panel complete through R4, a stricter support than
        # the R0/R1 one the rest of the ladder uses.
        support4 = full_round_support(d)
        r4_gain = ((tg.accuracy(loaded["mixed"], support4, 4)
                    - tg.accuracy(loaded["mixed"], support4, 0))
                   - (tg.accuracy(loaded["d_homo"], support4, 4)
                      - tg.accuracy(loaded["d_homo"], support4, 0)))

        rec_mixed, cor_mixed = tr.transitions(loaded["mixed"], support)
        rec_homo, cor_homo = tr.transitions(loaded["d_homo"], support)

        cols[label] = {
            "Gamma": a0p - a0d,
            "rho": 100.0 * k_rho / n_rho,
            "r1_gain": r1_gain,
            "r4_gain": r4_gain,
            "r1_recovery": rec_mixed,
            "r1_honest_corruption": cor_mixed,
            "upsilon": 100.0 * k_up / n_up,
            "rho_lo": rho_lo, "rho_hi": rho_hi,
            "upsilon_lo": ups_lo, "upsilon_hi": ups_hi,
            "n_rho": n_rho, "n_upsilon": n_up,
            "_homo_recovery": rec_homo,
            "_homo_corruption": cor_homo,
            "n": sum(len(v) for v in support.values()),
        }

    base = cols["Gemma"]
    cols["Qwen"] = {
        "Gamma": 0.0, "rho": None, "r1_gain": 0.0, "r4_gain": 0.0,
        "rho_lo": None, "rho_hi": None, "upsilon_lo": None,
        "upsilon_hi": None, "n_rho": None, "n_upsilon": None,
        "r1_recovery": base["_homo_recovery"],
        "r1_honest_corruption": base["_homo_corruption"],
        "upsilon": None, "n": base["n"],
    }
    return cols


def main() -> None:
    cols = compute()

    fields = ["quantity", "Qwen", "Gemma", "OSS-120B"]
    order = [("Gamma", "Gamma (pp)"), ("rho", "rho (%)"),
             ("r1_gain", "R1 gain (pp)"), ("r4_gain", "R4 gain (pp)"),
             ("r1_recovery", "R1 recovery (%)"),
             ("r1_honest_corruption", "R1 honest corruption (%)"),
             ("upsilon", "upsilon (%)"), ("rho_lo", "rho CI lo"),
             ("rho_hi", "rho CI hi"), ("upsilon_lo", "upsilon CI lo"),
             ("upsilon_hi", "upsilon CI hi"), ("n_rho", "rho exposures"),
             ("n_upsilon", "upsilon exposures"),
             ("n", "matched replicate-items")]

    rows = []
    for key, label in order:
        row = {"quantity": label}
        for name, _ in LADDER:
            v = cols[name].get(key)
            row[name] = ("" if v is None
                         else (v if key.startswith("n") else round(v, 2)))
        rows.append(row)
        print(f"  {label:26s} " + "  ".join(
            f"{str(row[n]):>8}" for n, _ in LADDER))

    traces.write_csv(OUT, fields, rows)


if __name__ == "__main__":
    main()
