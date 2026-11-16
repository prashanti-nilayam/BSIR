#!/usr/bin/env python3
"""Compute the quantities the manuscript states only in running text.

The tables are emitted by table_*.py; this covers everything written by hand
into Sections 4 and the appendix -- correlations, paired contrasts across the
fixed-Qwen ladder, compliance rates, and the confirmatory ceiling.

Output: results/prose_quantities.csv  (name, value, paper)
"""
import math
import csv

from artifact_paths import REPO, RESULTS as RESULTS_DIR
import bootstrap
import panels
import table_directions as td
import table_gate as tg
import table_logistic as tl
import table_rates as tr
import table_reception as trc
import table_specifications as ts
import traces

OUT = RESULTS_DIR / "prose_quantities.csv"
FIXED = ("qwen7b_gemma12b_math", "qwen7b_gptoss_math")

CLAIMS = REPO / "claims" / "prose_claims.csv"


def claims():
    return {r["quantity"]: r for r in csv.DictReader(CLAIMS.open())}


def pooled_r2(sets):
    """Product and hinge R^2 on a pooled set of directions."""
    g = [max(0.0, d["gamma"]) for d in sets]
    y = [d["gain"] for d in sets]
    prod = [a * d["rho"] / 100.0 for a, d in zip(g, sets)]
    return ts.r_squared(prod, y), ts.r_squared(g, y)


def per_direction(keys=None):
    """Direction-level quantities used by the cross-direction statistics."""
    out = []
    for d in panels.DIRECTIONS:
        if keys is None and d.get("fixed_receiver"):
            continue
        if keys is not None and d["key"] not in keys:
            continue
        support, loaded = td.matched_support(d)
        a0d = td.r0_accuracy(loaded["d_homo"], support)
        a0p = td.r0_accuracy(loaded["p_homo"], support)
        n_r, k_r = trc.exposures(loaded, support, "mixed", True, False)
        n_u, k_u = trc.exposures(loaded, support, "adv", False, True)
        out.append({
            "key": d["key"], "gamma": a0p - a0d, "headroom": 100.0 - a0d,
            "rho": 100.0 * k_r / n_r, "uptake": 100.0 * k_u / n_u,
            "gain": tg.net_gain(loaded, support, "mixed"),
            "adv_gain": tg.net_gain(loaded, support, "adv"),
        })
    return out


def corr_pair(xs, ys):
    return ts.pearson(xs, ys), ts.exact_permutation_p(xs, ys)


def through_origin(xs, ys):
    slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    ss_res = sum((y - slope * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum(y * y for y in ys)
    return slope, 1.0 - ss_res / ss_tot


def compliance(direction):
    """Share of complete replicate-items where the adversary parsed a wrong
    answer in all five rounds."""
    support, loaded = td.matched_support(direction)
    parse_out = traces.ADAPTERS[direction["benchmark"]][0]
    reps = dict(loaded["adv"])
    good = total = 0
    for rep, items in support.items():
        recs = reps.get(rep) or loaded["adv"][0][1]
        for item in items:
            rs = [recs.get((item, rd, panels.PEER_SLOT)) for rd in range(5)]
            if any(r is None for r in rs):
                continue
            total += 1
            good += all(not r["ok"]
                        and parse_out(r.get("raw_output") or "") is not None
                        for r in rs)
    return 100.0 * good / total if total else float("nan")


def _fixed_cells(key, what):
    """{cluster: counts} for one rung of the fixed-Qwen ladder."""
    d = next(x for x in panels.DIRECTIONS if x["key"] == key)
    support, loaded = td.matched_support(d)
    if what == "rho":
        return trc.exposures_by_cluster(loaded, support, "mixed",
                                        d["benchmark"], True, False)
    if what == "uptake":
        return trc.exposures_by_cluster(loaded, support, "adv",
                                        d["benchmark"], False, True)
    # controlled gain: per cluster, (mixed R1, mixed R0, homo R1, homo R0, n)
    out = {}
    for panel, off in (("mixed", 0), ("d_homo", 2)):
        reps = dict(loaded[panel])
        for rep, items in support.items():
            recs = reps.get(rep) or loaded[panel][0][1]
            for item in items:
                c = bootstrap.cluster_key(d["benchmark"], item)
                v = list(out.get(c, [0.0] * 5))
                for sl in panels.DEFENDER_SLOTS:
                    r0, r1 = recs.get((item, 0, sl)), recs.get((item, 1, sl))
                    if r0 is None or r1 is None:
                        continue
                    v[off] += r1["ok"]
                    v[off + 1] += r0["ok"]
                    if panel == "mixed":
                        v[4] += 1
                out[c] = tuple(v)
    return out


def paired(q):
    """Differences between the OSS and Gemma rungs, paired on shared items."""
    import numpy as np
    specs = [
        ("gain", lambda a: 100.0 * ((a[0] - a[1]) / a[4] - (a[2] - a[3]) / a[4])),
        ("rho", lambda a: 100.0 * a[0] / a[1]),
        ("uptake", lambda a: 100.0 * a[0] / a[1]),
    ]
    for what, stat in specs:
        gem = _fixed_cells(FIXED[0], what)
        oss = _fixed_cells(FIXED[1], what)
        clusters = sorted(set(gem) | set(oss))
        order, w = bootstrap.weight_matrix(clusters)
        pos = {c: i for i, c in enumerate(order)}

        def draws(cells):
            width = len(next(iter(cells.values())))
            arrs = []
            for f in range(width):
                v = np.zeros(len(order))
                for c, t in cells.items():
                    v[pos[c]] = t[f]
                arrs.append(w @ v)
            with np.errstate(invalid="ignore", divide="ignore"):
                return stat(arrs)

        def point(cells):
            width = len(next(iter(cells.values())))
            tot = [sum(t[f] for t in cells.values()) for f in range(width)]
            return stat(tot)

        diff = point(oss) - point(gem)
        d = draws(oss) - draws(gem)
        d = d[np.isfinite(d)]
        q[f"paired_{what}_diff"] = diff
        q[f"paired_{what}_lo"] = float(np.percentile(d, 2.5))
        q[f"paired_{what}_hi"] = float(np.percentile(d, 97.5))
        # Two-sided bootstrap tail area from the same paired draws used for
        # the interval.  A zero tail count is reported in prose as less than
        # 2 / REPS, the resolution of this two-sided empirical calculation.
        q[f"p_paired_{what}_diff"] = min(
            1.0, 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))


def item_level_odds(q):
    """Recovery / corruption odds per ten points of peer gap, pooling the
    Qwen defender's exposures in [Q,Q,Q], [Q,Q,G] and [Q,Q,O]."""
    rungs = [(None, 0.0)] + [(k, None) for k in FIXED]
    for outcome, tag in (("recovery", "recovery"), ("corruption", "corruption")):
        cells = []
        for key in FIXED:
            d = next(x for x in panels.DIRECTIONS if x["key"] == key)
            support, loaded = td.matched_support(d)
            gap = (td.r0_accuracy(loaded["p_homo"], support)
                   - td.r0_accuracy(loaded["d_homo"], support))
            for panel, g in (("d_homo", 0.0), ("mixed", gap)):
                if panel == "d_homo" and key == FIXED[1]:
                    continue          # shared baseline, counted once
                reps = dict(loaded[panel])
                for rep, items in support.items():
                    recs = reps.get(rep) or loaded[panel][0][1]
                    for item in items:
                        for sl in panels.DEFENDER_SLOTS:
                            r0 = recs.get((item, 0, sl))
                            r1 = recs.get((item, 1, sl))
                            if r0 is None or r1 is None:
                                continue
                            if outcome == "recovery":
                                if r0["ok"]:
                                    continue
                                y = int(r1["ok"])
                            else:
                                if not r0["ok"]:
                                    continue
                                y = int(not r1["ok"])
                            x = [1.0, g / 10.0,
                                 1.0 if rep == 2 else 0.0,
                                 1.0 if rep == 3 else 0.0,
                                 float(sl == panels.DEFENDER_SLOTS[1])]
                            cells.append((x, item, y, 1 - y))
        n_clusters = len({c for _x, c, _a, _b in cells})
        beta, se, _n = tl.fit(cells, 5, n_clusters)
        z = 1.959963984540054
        q[f"or_{tag}_gap"] = math.exp(beta[1])
        q[f"or_{tag}_lo"] = math.exp(beta[1] - z * se[1])
        q[f"or_{tag}_hi"] = math.exp(beta[1] + z * se[1])


def main() -> None:
    q = {}
    disc = per_direction()
    gam = [d["gamma"] for d in disc]
    gain = [d["gain"] for d in disc]
    rho = [d["rho"] for d in disc]
    upt = [d["uptake"] for d in disc]
    head = [d["headroom"] for d in disc]

    q["mean_honest_gain"] = sum(gain) / len(gain)
    q["p_mean_honest_gain"] = tg.sign_flip_p(gain)
    for name, xs, ys in (("gap_gain", gam, gain), ("gap_uptake", gam, upt),
                         ("gap_rho", gam, rho), ("headroom_rho", head, rho),
                         ("rho_uptake", rho, upt)):
        r, p = corr_pair(xs, ys)
        q[f"r_{name}"], q[f"p_{name}"] = r, p

    gplus = [max(0.0, g) for g in gam]
    for tag, xs in (("hinge", gplus),
                    ("product", [g * r / 100.0 for g, r in zip(gplus, rho)])):
        slope = ts.ols_slope(xs, gain)
        n = len(xs)
        icept = sum(gain) / n - slope * sum(xs) / n
        q[f"max_residual_{tag}"] = max(abs(y - (icept + slope * x))
                                       for x, y in zip(xs, gain))
    q["adv_mean_below_homogeneous"] = (sum(d["adv_gain"] for d in disc)
                                       / len(disc))

    # "With GPT-OSS added": the discovery nine plus the OSS rung of the
    # fixed-Qwen ladder. Adding the Gemma rung as well does not reproduce the
    # reported correlations, so the pooled set is ten directions, not eleven.
    allx = disc + per_direction((FIXED[1],))
    pg = [d["gamma"] for d in allx]
    for name, ys in (("gap_gain", [d["gain"] for d in allx]),
                     ("gap_rho", [d["rho"] for d in allx]),
                     ("gap_uptake", [d["uptake"] for d in allx])):
        r, p = corr_pair(pg, ys)
        q[f"pooled_r_{name}"], q[f"pooled_p_{name}"] = r, p

    # Fixed-Qwen ladder: through-origin fits over [Qwen, Gemma, OSS].
    fx = {d["key"]: d for d in per_direction(FIXED)}
    gaps = [0.0, fx[FIXED[0]]["gamma"], fx[FIXED[1]]["gamma"]]
    r1 = [0.0, fx[FIXED[0]]["gain"], fx[FIXED[1]]["gain"]]
    q["ladder_slope_r1"], q["ladder_r2_r1"] = through_origin(gaps, r1)

    r4 = [0.0]
    for key in FIXED:
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        import table_confirmatory as tc
        s4 = tc.full_round_support(d)
        _, loaded = td.matched_support(d)
        r4.append((tg.accuracy(loaded["mixed"], s4, 4)
                   - tg.accuracy(loaded["mixed"], s4, 0))
                  - (tg.accuracy(loaded["d_homo"], s4, 4)
                     - tg.accuracy(loaded["d_homo"], s4, 0)))
    q["ladder_slope_r4"], q["ladder_r2_r4"] = through_origin(gaps, r4)

    # Cross-peer prediction using the discovery-set product coefficients.
    slope = ts.ols_slope([g * r / 100.0 for g, r in zip(gplus, rho)], gain)
    icept = (sum(gain) / len(gain)
             - slope * sum(g * r / 100.0 for g, r in zip(gplus, rho)) / len(gain))
    g_gem, g_oss = fx[FIXED[0]], fx[FIXED[1]]
    q["predict_oss_from_gemma"] = icept + slope * (g_oss["gamma"]
                                                   * g_gem["rho"] / 100.0)
    q["predict_gemma_from_oss"] = icept + slope * (g_gem["gamma"]
                                                   * g_oss["rho"] / 100.0)

    for key, tag in zip(FIXED, ("gemma", "oss")):
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        q[f"compliance_{tag}"] = compliance(d)

    # R4 levels and honest-to-adversarial swings on both fixed-receiver rungs.
    import table_confirmatory as tc
    import table_role_effects as tre
    for key, tag in zip(FIXED, ("gemma", "oss")):
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        s4 = tc.full_round_support(d)
        _, loaded = td.matched_support(d)
        q[f"r4_qwen_with_{tag}"] = tg.accuracy(loaded["mixed"], s4, 4)

        def swing(support):
            return ((tg.accuracy(loaded["adv"], support, 4)
                     - tg.accuracy(loaded["adv"], support, 0))
                    - (tg.accuracy(loaded["mixed"], support, 4)
                       - tg.accuracy(loaded["mixed"], support, 0)))

        q[f"r4_adv_swing_{tag}_itt"] = swing(s4)
        pp4 = tre.compliant_support(s4, loaded, d["benchmark"])
        q[f"r4_adv_swing_{tag}_pp"] = swing(pp4)
        if tag == "oss":
            adv = ((tg.accuracy(loaded["adv"], s4, 4)
                    - tg.accuracy(loaded["adv"], s4, 0))
                   - (tg.accuracy(loaded["d_homo"], s4, 4)
                      - tg.accuracy(loaded["d_homo"], s4, 0)))
            q["r4_adv_below_homogeneous"] = adv
            q["r4_adv_swing"] = q["r4_adv_swing_oss_itt"]

    # Confirmatory ceiling on the OSS rung.
    d = next(x for x in panels.DIRECTIONS if x["key"] == FIXED[1])
    s4 = tc.full_round_support(d)
    _, loaded = td.matched_support(d)
    best = {"peer_in_panel": 0.0, "panel_two": 0.0, "oracle": 0.0}
    reps = dict(loaded["mixed"])
    for rd in range(5):
        pk = pn = tk = ok = tot = 0
        for rep, items in s4.items():
            recs = reps.get(rep) or loaded["mixed"][0][1]
            for item in items:
                peer = recs.get((item, rd, panels.PEER_SLOT))
                if peer is not None:
                    pk += peer["ok"]
                    pn += 1
                n_ok = peer["ok"] if peer is not None else 0
                for s in panels.DEFENDER_SLOTS:
                    r = recs.get((item, rd, s))
                    if r is not None:
                        n_ok += r["ok"]
                tk += int(n_ok >= 2)
                ok += int(n_ok >= 1)
                tot += 1
        best["peer_in_panel"] = max(best["peer_in_panel"], 100.0 * pk / pn)
        best["panel_two"] = max(best["panel_two"], 100.0 * tk / tot)
        best["oracle"] = max(best["oracle"], 100.0 * ok / tot)
    q["oss_peer_in_panel"] = best["peer_in_panel"]
    q["oss_panel_two"] = best["panel_two"]
    q["oss_oracle"] = best["oracle"]

    q["pooled10_product_r2"], q["pooled10_hinge_r2"] = pooled_r2(allx)
    both = disc + per_direction(FIXED)
    q["pooled11_product_r2"], q["pooled11_hinge_r2"] = pooled_r2(both)

    # Paired contrasts across the two rungs, bootstrapped on shared items.
    paired(q)
    item_level_odds(q)

    ab = {r["panel"] + "/" + r["metric"]: r
          for r in __import__("csv").DictReader(
              (RESULTS_DIR / "agent_diversity_ablation.csv").open())}
    di = ab["diverse/any_agent"]
    q["diverse_oracle_contraction"] = float(di["R0"]) - float(di["R4"])

    mh = {r["metric"]: r for r in csv.DictReader(
        (RESULTS_DIR / "musique_heterogeneous.csv").open())}
    musique_claims = {
        "musique_gamma": ("capability_gap_qwen_minus_gemma", "estimate"),
        "musique_gamma_lo": ("capability_gap_qwen_minus_gemma", "ci_low"),
        "musique_gamma_hi": ("capability_gap_qwen_minus_gemma", "ci_high"),
        "musique_honest_gain_r1": ("r1_controlled_gain_honest", "estimate"),
        "musique_honest_gain_r1_lo": ("r1_controlled_gain_honest", "ci_low"),
        "musique_honest_gain_r1_hi": ("r1_controlled_gain_honest", "ci_high"),
        "musique_recovery_diff":
            ("r1_recovery_difference_honest_minus_homogeneous", "estimate"),
        "musique_recovery_diff_lo":
            ("r1_recovery_difference_honest_minus_homogeneous", "ci_low"),
        "musique_recovery_diff_hi":
            ("r1_recovery_difference_honest_minus_homogeneous", "ci_high"),
        "musique_corruption_diff":
            ("r1_corruption_difference_honest_minus_homogeneous", "estimate"),
        "musique_corruption_diff_lo":
            ("r1_corruption_difference_honest_minus_homogeneous", "ci_low"),
        "musique_corruption_diff_hi":
            ("r1_corruption_difference_honest_minus_homogeneous", "ci_high"),
        "musique_rho": ("honest_reception", "estimate"),
        "musique_rho_lo": ("honest_reception", "ci_low"),
        "musique_rho_hi": ("honest_reception", "ci_high"),
        "musique_rho0": ("honest_unaided_revision", "estimate"),
        "musique_rho0_lo": ("honest_unaided_revision", "ci_low"),
        "musique_rho0_hi": ("honest_unaided_revision", "ci_high"),
        "musique_adv_excess_corruption":
            ("r1_corruption_difference_adversarial_peer_minus_homogeneous", "estimate"),
        "musique_adv_excess_corruption_lo":
            ("r1_corruption_difference_adversarial_peer_minus_homogeneous", "ci_low"),
        "musique_adv_excess_corruption_hi":
            ("r1_corruption_difference_adversarial_peer_minus_homogeneous", "ci_high"),
        "musique_uptake": ("adversarial_uptake", "estimate"),
        "musique_adv_gain_r4": ("r4_controlled_gain_adversarial_peer", "estimate"),
        "musique_adv_gain_r4_lo": ("r4_controlled_gain_adversarial_peer", "ci_low"),
        "musique_adv_gain_r4_hi": ("r4_controlled_gain_adversarial_peer", "ci_high"),
        "musique_same_adv_excess_corruption":
            ("r1_corruption_difference_same_family_adversary_minus_homogeneous", "estimate"),
        "musique_same_adv_excess_corruption_lo":
            ("r1_corruption_difference_same_family_adversary_minus_homogeneous", "ci_low"),
        "musique_same_adv_excess_corruption_hi":
            ("r1_corruption_difference_same_family_adversary_minus_homogeneous", "ci_high"),
        "musique_same_adv_gain_r4":
            ("r4_controlled_gain_same_family_adversary", "estimate"),
        "musique_same_adv_gain_r4_lo":
            ("r4_controlled_gain_same_family_adversary", "ci_low"),
        "musique_same_adv_gain_r4_hi":
            ("r4_controlled_gain_same_family_adversary", "ci_high"),
        "musique_contaminated_effect_r1":
            ("r1_contaminated_honest_qwen_effect", "estimate"),
        "musique_contaminated_effect_r1_lo":
            ("r1_contaminated_honest_qwen_effect", "ci_low"),
        "musique_contaminated_effect_r1_hi":
            ("r1_contaminated_honest_qwen_effect", "ci_high"),
        "musique_contaminated_effect_r4":
            ("r4_contaminated_honest_qwen_effect", "estimate"),
        "musique_contaminated_effect_r4_lo":
            ("r4_contaminated_honest_qwen_effect", "ci_low"),
        "musique_contaminated_effect_r4_hi":
            ("r4_contaminated_honest_qwen_effect", "ci_high"),
    }
    for name, (metric, field) in musique_claims.items():
        q[name] = float(mh[metric][field])

    discordant = list(csv.DictReader(
        (RESULTS_DIR / "discordant_flows.csv").open()))
    musique_flow = next(r for r in discordant if r["cohort"] == "MuSiQue scope")
    q["musique_discordant_correction"] = float(musique_flow["S_rho"])
    q["musique_discordant_corruption"] = float(musique_flow["T_eta"])

    composition = list(csv.DictReader(
        (RESULTS_DIR / "composition_accounting.csv").open()))
    new_correct = [float(r["new_correct"]) for r in composition]
    discarded = [float(r["discarded_initial"]) for r in composition]
    q["composition_new_min"] = min(new_correct)
    q["composition_new_max"] = max(new_correct)
    q["composition_discarded_min"] = min(discarded)
    q["composition_discarded_max"] = max(discarded)

    expected = claims()
    rows = []
    for name in sorted(q):
        claim = expected.get(name)
        paper = float(claim["expected"]) if claim else None
        val = q[name]
        dec = int(claim["precision"]) if claim else 5
        if claim and claim["comparison"] == "zero_tail_bound":
            # No observed draw in one tail only establishes p < 2/REPS.
            effective = 2.0 / bootstrap.REPS if val == 0 else val
            agree = effective <= paper
        else:
            agree = paper is None or round(val, dec) == round(paper, dec)
        rows.append({"quantity": name, "value": round(val, 5),
                     "paper": "" if paper is None else paper,
                     "agrees": "yes" if agree else "NO"})
        flag = "" if agree else "   <-- differs"
        print(f"  {name:30s} {val:12.5f}   paper={paper}{flag}")

    traces.write_csv(OUT, ["quantity", "value", "paper", "agrees"], rows)
    bad = [r for r in rows if r["agrees"] == "NO"]
    print(f"\n  {len(rows) - len(bad)}/{len(rows)} agree with the manuscript")


if __name__ == "__main__":
    main()
