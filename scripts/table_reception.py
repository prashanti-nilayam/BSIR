#!/usr/bin/env python3
"""Reception power per direction.

  rho     = P(defender correct at R1 | defender wrong at R0, honest peer
            correct at R0)  -- the receiver-side transfer rate
  n       = eligible defender exposures behind rho
  unaided = the same recovery rate when the peer was also wrong at R0
  uptake  = P(defender wrong at R1 | defender correct at R0, adversarial peer
            committed wrong at R0)
  gain    = controlled R0->R1 accuracy change of the honest panel

Output: results/reception.csv
"""
from artifact_paths import RESULTS
import bootstrap
import panels
import table_directions as td
import table_gate as tg
import traces

OUT = RESULTS / "reception.csv"


def exposures_by_cluster(loaded, support, panel, benchmark,
                         peer_correct: bool, from_correct: bool):
    """{benchmark-item cluster: (successes, denominator)} for one channel."""
    reps = dict(loaded[panel])
    out = {}
    for rep, items in support.items():
        recs = reps.get(rep) or loaded[panel][0][1]
        for item in items:
            peer = recs.get((item, 0, panels.PEER_SLOT))
            if peer is None or bool(peer["ok"]) is not peer_correct:
                continue
            key = bootstrap.cluster_key(benchmark, item)
            n, k = out.get(key, (0.0, 0.0))
            for s in panels.DEFENDER_SLOTS:
                r0, r1 = recs.get((item, 0, s)), recs.get((item, 1, s))
                if r0 is None or r1 is None:
                    continue
                if bool(r0["ok"]) is not from_correct:
                    continue
                n += 1
                k += int((not r1["ok"]) if from_correct else r1["ok"])
            out[key] = (n, k)
    # stored as (denominator, successes); return (successes, denominator)
    return {c: (k, n) for c, (n, k) in out.items()}


def exposures(loaded, support, panel, peer_correct: bool, from_correct: bool):
    """Count (eligible, transitioned) defender exposures.

    from_correct=False -> recovery  (R0 wrong  -> R1 correct)
    from_correct=True  -> corruption(R0 correct-> R1 wrong)
    peer_correct filters on the peer's own R0 state.
    """
    reps = dict(loaded[panel])
    elig = hit = 0
    for rep, items in support.items():
        recs = reps.get(rep) or loaded[panel][0][1]
        for item in items:
            peer = recs.get((item, 0, panels.PEER_SLOT))
            if peer is None or bool(peer["ok"]) is not peer_correct:
                continue
            for s in panels.DEFENDER_SLOTS:
                r0, r1 = recs.get((item, 0, s)), recs.get((item, 1, s))
                if r0 is None or r1 is None:
                    continue
                if bool(r0["ok"]) is not from_correct:
                    continue
                elig += 1
                hit += int((not r1["ok"]) if from_correct else r1["ok"])
    return elig, hit


def main() -> None:
    rows = []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(d)
        gamma = (td.r0_accuracy(loaded["p_homo"], support)
                 - td.r0_accuracy(loaded["d_homo"], support))

        n_rho, k_rho = exposures(loaded, support, "mixed", True, False)
        by_cluster = exposures_by_cluster(loaded, support, "mixed",
                                          d["benchmark"], True, False)
        boot_lo, boot_hi = bootstrap.ratio_interval(by_cluster,
                                                    list(by_cluster))
        n_un, k_un = exposures(loaded, support, "mixed", False, False)
        n_up, k_up = exposures(loaded, support, "adv", False, True)

        rows.append({
            "direction": d["label"],
            "Gamma": round(gamma, 1),
            "rho": round(100.0 * k_rho / n_rho, 1),
            "n": n_rho,
            "unaided": round(100.0 * k_un / n_un, 1),
            "uptake": round(100.0 * k_up / n_up, 1),
            "gain": round(tg.net_gain(loaded, support, "mixed"), 1),
            "adv_gain": round(tg.net_gain(loaded, support, "adv"), 2),
            "boot_lo": round(boot_lo, 1),
            "boot_hi": round(boot_hi, 1),
        })
        r = rows[-1]
        print(f"  {d['label']:28s} G={r['Gamma']:+6.1f} rho={r['rho']:5.1f} "
              f"n={r['n']:4d} unaided={r['unaided']:5.1f} "
              f"uptake={r['uptake']:5.1f} gain={r['gain']:+6.1f} "
              f"boot=[{r['boot_lo']:.1f}, {r['boot_hi']:.1f}]")

    traces.write_csv(OUT, ["direction", "Gamma", "rho", "n", "boot_lo",
                           "boot_hi", "unaided", "uptake", "gain",
                           "adv_gain"], rows)


if __name__ == "__main__":
    main()
