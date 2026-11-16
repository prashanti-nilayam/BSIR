#!/usr/bin/env python3
"""Direction-level role contrasts.

Equal-weight means over the nine discovery directions of the per-direction
rate differences, with an exact two-sided sign-flip test over all 2^9 sign
assignments.

Reported twice: intent-to-treat over all matched items, and per protocol over
items where the adversary parsed a wrong answer in every round. Intervals are
the item-clustered Poisson bootstrap; the per-protocol population draws its own
weight stream. Odds ratios live in results/logistic.csv.

Output: results/role_effects.csv
"""
from artifact_paths import RESULTS
import bootstrap
import panels
import table_directions as td
import table_gate as tg
import table_rates as tr
import traces

OUT = RESULTS / "role_effects.csv"

CONTRASTS = [
    ("Recovery: honest vs homogeneous", "recovery", "mixed", "d_homo"),
    ("Corruption: honest vs homogeneous", "corruption", "mixed", "d_homo"),
    ("Recovery: adversarial vs honest", "recovery", "adv", "mixed"),
    ("Corruption: adversarial vs honest", "corruption", "adv", "mixed"),
]


def compliant_support(support, loaded, benchmark):
    """Restrict to items where the adversary parsed a wrong answer in every
    round R0-R4 -- the per-protocol population."""
    parse_out = traces.ADAPTERS[benchmark][0]
    reps = dict(loaded["adv"])
    out = {}
    for rep, items in support.items():
        recs = reps.get(rep) or loaded["adv"][0][1]
        keep = set()
        for item in items:
            good = True
            for rd in range(5):
                r = recs.get((item, rd, panels.PEER_SLOT))
                if r is None or r["ok"] or parse_out(r.get("raw_output") or "") is None:
                    good = False
                    break
            if good:
                keep.add(item)
        out[rep] = keep
    return out


def cluster_counts(loaded, support, panel, benchmark, outcome):
    """{cluster: (successes, denominator)} for one panel and outcome."""
    reps = dict(loaded[panel])
    out = {}
    for rep, items in support.items():
        recs = reps.get(rep) or loaded[panel][0][1]
        for item in items:
            key = bootstrap.cluster_key(benchmark, item)
            k, n = out.get(key, (0.0, 0.0))
            for s in panels.DEFENDER_SLOTS:
                r0, r1 = recs.get((item, 0, s)), recs.get((item, 1, s))
                if r0 is None or r1 is None:
                    continue
                if outcome == "recovery":
                    if r0["ok"]:
                        continue
                    k += int(r1["ok"])
                else:
                    if not r0["ok"]:
                        continue
                    k += int(not r1["ok"])
                n += 1
            out[key] = (k, n)
    return out


def analyse(kind):
    """kind: 'itt' (all matched items) or 'pp' (adversary-compliant only)."""
    per_direction = []
    cluster_data = []
    all_clusters = set()
    n_total = 0
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(d)
        if kind == "pp":
            support = compliant_support(support, loaded, d["benchmark"])
        n_total += sum(len(v) for v in support.values())
        rates = {p: tr.transitions(loaded[p], support)
                 for p in ("d_homo", "mixed", "adv")}
        cd = {}
        for outcome in ("recovery", "corruption"):
            for p in ("d_homo", "mixed", "adv"):
                cc = cluster_counts(loaded, support, p, d["benchmark"], outcome)
                cd[(outcome, p)] = cc
                all_clusters |= set(cc)
        cluster_data.append(cd)
        per_direction.append({
            "recovery": {p: v[0] for p, v in rates.items()},
            "corruption": {p: v[1] for p, v in rates.items()},
        })

    return per_direction, cluster_data, all_clusters, n_total


def main() -> None:
  rows = []
  for kind, kind_label in (("itt", "intent to treat"),
                           ("pp", "per protocol")):
    per_direction, cluster_data, all_clusters, n_total = analyse(kind)
    for label, outcome, treat, base in CONTRASTS:
        vals = [x[outcome][treat] - x[outcome][base] for x in per_direction]
        mean = sum(vals) / len(vals)

        merged = []
        for cd in cluster_data:
            t, b = cd[(outcome, treat)], cd[(outcome, base)]
            keys = set(t) | set(b)
            merged.append({c: (t.get(c, (0.0, 0.0))[0], t.get(c, (0.0, 0.0))[1],
                               b.get(c, (0.0, 0.0))[0], b.get(c, (0.0, 0.0))[1])
                           for c in keys})

        def contrast(arrays):
            import numpy as np
            tk, tn, bk, bn = arrays
            with np.errstate(invalid="ignore", divide="ignore"):
                return 100.0 * (np.where(tn > 0, tk / tn, np.nan)
                                - np.where(bn > 0, bk / bn, np.nan))

        # The per-protocol population draws its own weight stream.
        seed = bootstrap.SEED + (1 if kind == "pp" else 0)
        blo, bhi = bootstrap.mean_of_direction_stats(
            merged, sorted(all_clusters), contrast, seed=seed)
        rows.append({
            "population": kind_label,
            "contrast": label,
            "matched_replicate_items": n_total,
            "mean_pp": round(mean, 2),
            "exact_p": round(tg.sign_flip_p(vals), 4),
            "bootstrap_lo": round(blo, 2),
            "bootstrap_hi": round(bhi, 2),
        })
        print(f"  {kind_label:15s} {label:38s} mean={mean:+7.2f} "
              f"p={rows[-1]['exact_p']:.4f} boot=[{blo:+7.2f},{bhi:+7.2f}]")

  traces.write_csv(OUT, list(rows[0].keys()), rows)
  print("\n  Bootstrap: item-clustered Poisson, seed "
        f"{bootstrap.SEED}, {bootstrap.REPS} replicates, weights assigned"
        "\n  in sorted cluster order -- reproducible from the seed alone.")


if __name__ == "__main__":
    main()
