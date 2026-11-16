#!/usr/bin/env python3
"""The capability gate: direction-level contrasts split at Gamma = 0.

Directions with a negative measured peer gap are "ungated", positive "gated".
Each quantity is an equal-weight mean over the directions in its group, with
an exact sign-flip test: enumerate all 2^n sign assignments of the
direction-level values and report the share whose mean is at least as extreme
as the observed one.

Output: results/gate.csv
"""
import itertools

from artifact_paths import RESULTS
import panels
import table_directions as td
import table_rates as tr
import traces

OUT = RESULTS / "gate.csv"


def accuracy(loaded_panel, support, rnd):
    """Defender-slot accuracy at one round, on the matched support."""
    hit = tot = 0
    reps = dict(loaded_panel)
    for rep, items in support.items():
        recs = reps.get(rep) or loaded_panel[0][1]
        for item in items:
            for s in panels.DEFENDER_SLOTS:
                r = recs.get((item, rnd, s))
                if r is not None:
                    hit += int(r["ok"])
                    tot += 1
    return 100.0 * hit / tot if tot else float("nan")


def net_gain(loaded, support, panel):
    """Controlled R0->R1 accuracy change: the panel's own gain minus the
    defender-homogeneous panel's gain over the same items."""
    own = accuracy(loaded[panel], support, 1) - accuracy(loaded[panel], support, 0)
    base = (accuracy(loaded["d_homo"], support, 1)
            - accuracy(loaded["d_homo"], support, 0))
    return own - base


def sign_flip_p(values):
    """Exact two-sided sign-flip test on the mean."""
    n = len(values)
    obs = abs(sum(values) / n)
    extreme = sum(
        1 for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * v for s, v in zip(signs, values)) / n) >= obs - 1e-12
    )
    return extreme / (2 ** n)


def main() -> None:
    per_direction = []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(d)
        rec = {p: tr.transitions(loaded[p], support)[0]
               for p in ("d_homo", "mixed", "adv")}
        cor = {p: tr.transitions(loaded[p], support)[1]
               for p in ("d_homo", "mixed", "adv")}
        a0d = td.r0_accuracy(loaded["d_homo"], support)
        a0p = td.r0_accuracy(loaded["p_homo"], support)
        per_direction.append({
            "gamma": a0p - a0d,
            "honest_recovery": rec["mixed"] - rec["d_homo"],
            "honest_net": net_gain(loaded, support, "mixed"),
            "adv_corruption": cor["adv"] - cor["d_homo"],
            "adv_net": net_gain(loaded, support, "adv"),
        })

    groups = {
        "ungated": [x for x in per_direction if x["gamma"] < 0],
        "gated": [x for x in per_direction if x["gamma"] > 0],
    }
    quantities = ["honest_recovery", "honest_net", "adv_corruption", "adv_net"]

    rows = []
    for q in quantities:
        row = {"quantity": q}
        for g, members in groups.items():
            vals = [m[q] for m in members]
            row[f"{g}_mean"] = round(sum(vals) / len(vals), 2)
            row[f"{g}_p"] = round(sign_flip_p(vals), 3)
            row[f"{g}_n"] = len(vals)
        rows.append(row)
        print(f"  {q:16s} ungated {row['ungated_mean']:+7.2f} ({row['ungated_p']:.3f})"
              f"   gated {row['gated_mean']:+7.2f} ({row['gated_p']:.3f})")

    traces.write_csv(OUT, ["quantity", "ungated_n", "ungated_mean", "ungated_p",
                           "gated_n", "gated_mean", "gated_p"], rows)


if __name__ == "__main__":
    main()
