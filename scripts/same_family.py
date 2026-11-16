#!/usr/bin/env python3
"""Fixed-receiver ladder with the additional same-family peer (Qwen-7B <- Qwen-72B).

Reuses the released analysis machinery (scripts/{traces,panels,table_directions,
table_gate,bootstrap}.py) so the same-family direction is computed under exactly the
definitions used for the manuscript:

  A0_D, A0_P  pooled R0 accuracy on the homogeneous defender / peer panels
  Gamma       A0_P - A0_D
  rho         P(defender correct at R1 | defender wrong at R0, peer correct at R0)
  unaided     same recovery rate when the peer is also wrong at R0
  gain        controlled R0->R1 defender-slot accuracy change (mixed minus d_homo)

The same-family direction has no adversarial arm, so matched support is taken over
d_homo and mixed (complete in R0+R1) intersected with peer-panel item presence;
uptake / adv_gain are not defined for it.

Duplicate records from the resumed rep3 are collapsed by traces.load, which keys
on (item_id, round, agent_id) and keeps the last record written.
"""
from __future__ import annotations

from artifact_paths import RESULTS

import bootstrap  # noqa: E402
import panels  # noqa: E402
import table_directions as td  # noqa: E402
import table_gate as tg  # noqa: E402
import table_discordant_flows as tdf  # noqa: E402
import table_reception as tr  # noqa: E402
import traces  # noqa: E402

NEW = dict(
    key="qwen7b_qwen72b_math",
    label="Qwen-7B <- Qwen-72B",
    task="MATH-hard-fx",
    benchmark="math",
    fixed_receiver=True,
    d_homo=[f"honest_qw7b_homo_hard_5round_rep{r}" for r in (1, 2, 3)],
    p_homo=[f"qwen72b_homo_e1_math_hard_5round_rep{r}" for r in (1, 2, 3)],
    mixed=[f"het_2qw7b1qw72_honest_hard_5round_rep{r}" for r in (1, 2, 3)],
)


def matched_support_no_adv(direction):
    """td.matched_support minus the adversarial arm."""
    bench = direction["benchmark"]
    loaded = {p: td.load_panel(direction[p], bench) for p in ("d_homo", "p_homo", "mixed")}
    support = {}
    n_reps = max(len(v) for v in loaded.values())
    for rep in range(1, n_reps + 1):
        sets = []
        for panel in ("d_homo", "mixed"):
            reps = loaded[panel]
            recs = dict(reps).get(rep) or reps[0][1]
            sets.append(traces.complete_items(recs, td.N_AGENTS, td.ROUNDS))
        p_reps = dict(loaded["p_homo"])
        p_recs = p_reps.get(rep) or loaded["p_homo"][0][1]
        sets.append({i for (i, _, _) in p_recs})
        support[rep] = set.intersection(*sets)
    return support, loaded


def row(direction, support, loaded, with_adv: bool):
    a0d = td.r0_accuracy(loaded["d_homo"], support)
    a0p = td.r0_accuracy(loaded["p_homo"], support)
    n_rho, k_rho = tr.exposures(loaded, support, "mixed", True, False)
    by_cluster = tr.exposures_by_cluster(loaded, support, "mixed", direction["benchmark"], True, False)
    lo, hi = bootstrap.ratio_interval(by_cluster, list(by_cluster))
    n_un, k_un = tr.exposures(loaded, support, "mixed", False, False)
    rho = 100.0 * k_rho / n_rho
    gamma = a0p - a0d
    out = {
        "direction": direction["label"],
        "n_items": sum(len(v) for v in support.values()),
        "A0_D": a0d,
        "A0_P": a0p,
        "Gamma": gamma,
        "H": 100.0 - a0d,
        "rho": rho,
        "rho_n": n_rho,
        "rho_lo": lo,
        "rho_hi": hi,
        "unaided": 100.0 * k_un / n_un,
        "unaided_n": n_un,
        "gain": tg.net_gain(loaded, support, "mixed"),
        "Gamma_rho": gamma * rho / 100.0,
    }
    if with_adv:
        n_up, k_up = tr.exposures(loaded, support, "adv", False, True)
        out["uptake"] = 100.0 * k_up / n_up
        out["adv_gain"] = tg.net_gain(loaded, support, "adv")
    return out


def mass_row(direction, support, loaded):
    """tdf._panel_row, but on a caller-supplied support (the same-family rung has no adv arm).

    Keeps both of the released script's integrity checks: the four discordant
    cells must partition the exposures, and the cell-wise reconstruction of the
    R0->R1 change must equal the directly measured change.
    """
    total = sum(len(items) for items in support.values()) * len(panels.DEFENDER_SLOTS)
    gamma = (td.r0_accuracy(loaded["p_homo"], support)
             - td.r0_accuracy(loaded["d_homo"], support))

    n_s, k_s = tr.exposures(loaded, support, "mixed", True, False)
    n_t, k_t = tr.exposures(loaded, support, "mixed", False, True)
    n_bw, k_bw = tr.exposures(loaded, support, "mixed", False, False)
    n_bc, k_bc = tr.exposures(loaded, support, "mixed", True, True)
    if n_s + n_t + n_bw + n_bc != total:
        raise ValueError(f"discordant cells do not partition {direction['key']}")

    mixed_change = 100.0 * (k_s + k_bw - k_t - k_bc) / total
    direct_change = (tg.accuracy(loaded["mixed"], support, 1)
                     - tg.accuracy(loaded["mixed"], support, 0))
    if abs(mixed_change - direct_change) > 1e-9:
        raise ValueError(f"transition accounting failed for {direction['key']}")

    r = tdf._rate
    return {
        "cohort": "fixed receiver",
        "direction": direction["label"],
        "Gamma": round(gamma, 2),
        "n_exposures": total,
        "S_mass": round(r(n_s, total), 2),
        "rho": round(r(k_s, n_s), 2),
        "S_rho": round(r(k_s, total), 2),
        "T_mass": round(r(n_t, total), 2),
        "mixed_gap": round(r(n_s - n_t, total), 2),
        "eta": round(r(k_t, n_t), 2),
        "T_eta": round(r(k_t, total), 2),
        "discordant_net": round(r(k_s - k_t, total), 2),
        "mixed_change": round(mixed_change, 2),
        "controlled_gain": round(tg.net_gain(loaded, support, "mixed"), 2),
    }


def per_round(loaded, support, panel, n_rounds=5):
    return [tg.accuracy(loaded[panel], support, rd) for rd in range(n_rounds)]


def main() -> int:
    rows = []
    # New same-family rung.
    sup_new, loaded_new = matched_support_no_adv(NEW)
    rows.append(row(NEW, sup_new, loaded_new, with_adv=False))
    # Existing fixed-receiver rungs, recomputed identically for comparison.
    for d in panels.DIRECTIONS:
        if not d.get("fixed_receiver"):
            continue
        sup, loaded = td.matched_support(d)
        rows.append(row(d, sup, loaded, with_adv=True))

    hdr = f"{'direction':24s} {'n':>5s} {'A0_D':>6s} {'A0_P':>6s} {'Gamma':>7s} {'rho':>6s} {'rho 95% CI':>14s} {'n_rho':>6s} {'unaided':>8s} {'gain':>7s} {'G*rho':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['direction']:24s} {r['n_items']:5d} {r['A0_D']:6.1f} {r['A0_P']:6.1f} "
              f"{r['Gamma']:+7.1f} {r['rho']:6.1f} "
              f"{'[%.1f, %.1f]' % (r['rho_lo'], r['rho_hi']):>14s} {r['rho_n']:6d} "
              f"{r['unaided']:8.1f} {r['gain']:+7.1f} {r['Gamma_rho']:7.1f}")

    print("\nPer-round defender-slot accuracy, same-family direction (matched support):")
    print("  mixed [qw7b,qw7b,qw72b]: " +
          "  ".join(f"R{i}={a:.1f}" for i, a in enumerate(per_round(loaded_new, sup_new, "mixed"))))
    print("  d_homo [qw7b x3]:        " +
          "  ".join(f"R{i}={a:.1f}" for i, a in enumerate(per_round(loaded_new, sup_new, "d_homo"))))

    # Discordant-state mass decomposition, same three rungs.
    mass = [mass_row(NEW, sup_new, loaded_new)]
    for d in panels.DIRECTIONS:
        if not d.get("fixed_receiver"):
            continue
        sup, loaded = td.matched_support(d)
        mass.append(mass_row(d, sup, loaded))

    mhdr = (f"\n{'direction':24s} {'exp':>5s} {'Gamma':>7s} {'S_mass':>7s} {'rho':>6s} {'S*rho':>7s} "
            f"{'T_mass':>7s} {'eta':>6s} {'T*eta':>7s} {'net':>7s} {'change':>7s} {'ctrl':>7s}")
    print(mhdr)
    print("-" * (len(mhdr) - 1))
    for m in mass:
        print(f"{m['direction']:24s} {m['n_exposures']:5d} {m['Gamma']:+7.2f} {m['S_mass']:7.2f} "
              f"{m['rho']:6.2f} {m['S_rho']:7.2f} {m['T_mass']:7.2f} {m['eta']:6.2f} "
              f"{m['T_eta']:7.2f} {m['discordant_net']:+7.2f} {m['mixed_change']:+7.2f} "
              f"{m['controlled_gain']:+7.2f}")

    mass_out = RESULTS / "same_family_discordant_flows.csv"
    traces.write_csv(mass_out, list(mass[0]), mass)

    out = RESULTS / "same_family_ladder.csv"
    fields = ["direction", "n_items", "A0_D", "A0_P", "Gamma", "H", "rho", "rho_n",
              "rho_lo", "rho_hi", "unaided", "unaided_n", "gain", "Gamma_rho", "uptake", "adv_gain"]
    traces.write_csv(out, fields, [{k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in r.items()} for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
