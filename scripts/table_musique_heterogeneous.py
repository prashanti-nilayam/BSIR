#!/usr/bin/env python3
"""Matched heterogeneous MuSiQue-Ans scope replication.

The primary comparison holds 300 open-answer multi-hop QA items fixed across
four panels:

  [G,G,G]       Gemma-3-12B homogeneous baseline
  [Q,Q,Q]       Qwen-7B homogeneous capability reference
  [G,G,Q]       honest heterogeneous panel
  [G,G,Q_adv]   adversarial heterogeneous panel

Two additional controls are catalogued as well: [G,G,G_adv] tests a
same-family adversary, and [G,G_adv,Q] adds an honest heterogeneous peer to an
already contaminated panel.  All correctness labels are alias-normalized exact
match recomputed from raw answers.  Uncertainty is a 4,000-draw benchmark-item
Poisson bootstrap shared across panels.

Output: results/musique_heterogeneous.csv
"""
from __future__ import annotations

import numpy as np

from artifact_paths import RESULTS
import bootstrap
import traces

OUT = RESULTS / "musique_heterogeneous.csv"
RUNS = {
    "d_homo": "musique_gemma3_homo_5round_rep1",
    "p_homo": "musique_qwen7b_homo_rep1",
    "honest": "musique_mixed_ggq_honest_rep1",
    "adv_peer": "musique_mixed_ggq_advpeer_rep1",
    "same_family_adv": "musique_gemma3_homo_adv_rep1",
    "contaminated": "musique_mixed_ggq_advdef_rep1",
}
DEFENDERS = (0, 1)


def _complete_support(loaded: dict[str, dict]) -> list[str]:
    sets = [traces.complete_items(records, 3, range(5))
            for records in loaded.values()]
    items = sorted(set.intersection(*sets))
    if len(items) != 300:
        raise ValueError(f"expected 300 fully matched MuSiQue items, found {len(items)}")
    return items


def _flags(records: dict, items: list[str], rd: int, slots=DEFENDERS) -> np.ndarray:
    return np.array([[float(records[(item, rd, slot)]["ok"]) for slot in slots]
                     for item in items], dtype=float)


def _draw_mean(weights: np.ndarray, per_item: np.ndarray) -> np.ndarray:
    return (weights @ per_item) / weights.sum(axis=1)


def _rate(flags0: np.ndarray, flags1: np.ndarray, event: str):
    if event == "recovery":
        eligible = 1.0 - flags0
        hits = eligible * flags1
    elif event == "corruption":
        eligible = flags0
        hits = eligible * (1.0 - flags1)
    else:
        raise ValueError(event)
    return hits.sum(axis=1), eligible.sum(axis=1)


def _conditional(def0: np.ndarray, def1: np.ndarray, peer0: np.ndarray,
                 defender_correct: bool, peer_correct: bool, outcome_correct: bool):
    d0 = def0 if defender_correct else 1.0 - def0
    p0 = peer0[:, None] if peer_correct else 1.0 - peer0[:, None]
    eligible = d0 * p0
    outcome = def1 if outcome_correct else 1.0 - def1
    return (eligible * outcome).sum(axis=1), eligible.sum(axis=1)


def _ratio_point_draws(num: np.ndarray, den: np.ndarray, weights: np.ndarray):
    point = 100.0 * num.sum() / den.sum()
    draws_num = weights @ num
    draws_den = weights @ den
    valid = draws_den > 0
    return point, 100.0 * draws_num[valid] / draws_den[valid], int(den.sum())


def compute() -> list[dict]:
    loaded = {name: traces.load(run, "musique") for name, run in RUNS.items()}
    items = _complete_support(loaded)
    clusters = [bootstrap.cluster_key("musique", item) for item in items]
    order, weights = bootstrap.weight_matrix(clusters)
    ordered_items = [cluster.split(":", 1)[1] for cluster in order]
    if ordered_items != items:
        items = ordered_items

    values = {name: {rd: _flags(records, items, rd)
                     for rd in range(5)}
              for name, records in loaded.items()}

    rows: list[dict] = []

    def add(metric: str, point: float, draws: np.ndarray, n: int, unit: str):
        rows.append({
            "metric": metric,
            "estimate": point,
            "ci_low": float(np.percentile(draws, bootstrap.LOW)),
            "ci_high": float(np.percentile(draws, bootstrap.HIGH)),
            "n": n,
            "unit": unit,
        })

    # Same-task capability supply, Qwen minus Gemma, measured at R0.
    gamma_items = (_flags(loaded["p_homo"], items, 0, (0, 1, 2)).mean(axis=1)
                   - _flags(loaded["d_homo"], items, 0, (0, 1, 2)).mean(axis=1)) * 100.0
    add("capability_gap_qwen_minus_gemma", float(gamma_items.mean()),
        _draw_mean(weights, gamma_items), len(items), "pp")

    # Controlled accuracy changes: change in the candidate panel minus the
    # simultaneous change in the Gemma homogeneous baseline.
    base_change = {}
    for rd in (1, 4):
        base_change[rd] = (values["d_homo"][rd].mean(axis=1)
                           - values["d_homo"][0].mean(axis=1)) * 100.0
        for condition, label in (("honest", "honest"),
                                 ("adv_peer", "adversarial_peer"),
                                 ("same_family_adv", "same_family_adversary")):
            change = (values[condition][rd].mean(axis=1)
                      - values[condition][0].mean(axis=1)) * 100.0
            controlled = change - base_change[rd]
            add(f"r{rd}_controlled_gain_{label}", float(controlled.mean()),
                _draw_mean(weights, controlled), len(items), "pp")

    # Transition rates and differences on defender slots.
    rates = {}
    for condition in ("d_homo", "honest", "adv_peer", "same_family_adv"):
        for event in ("recovery", "corruption"):
            num, den = _rate(values[condition][0], values[condition][1], event)
            point, draws, n = _ratio_point_draws(num, den, weights)
            rates[(condition, event)] = (point, draws, n)
            add(f"r1_{event}_{condition}", point, draws, n, "percent")

    for condition, label, event in (
        ("honest", "honest_minus_homogeneous", "recovery"),
        ("honest", "honest_minus_homogeneous", "corruption"),
        ("adv_peer", "adversarial_peer_minus_homogeneous", "corruption"),
        ("same_family_adv", "same_family_adversary_minus_homogeneous", "corruption"),
    ):
        point = rates[(condition, event)][0] - rates[("d_homo", event)][0]
        draws = rates[(condition, event)][1] - rates[("d_homo", event)][1]
        add(f"r1_{event}_difference_{label}", point, draws,
            min(rates[(condition, event)][2], rates[("d_homo", event)][2]), "pp")

    # Reception, unaided revision, and adversarial uptake.
    honest_peer0 = _flags(loaded["honest"], items, 0, (2,)).ravel()
    for metric, args in (
        ("honest_reception", (False, True, True)),
        ("honest_unaided_revision", (False, False, True)),
    ):
        num, den = _conditional(values["honest"][0], values["honest"][1],
                                honest_peer0, *args)
        point, draws, n = _ratio_point_draws(num, den, weights)
        add(metric, point, draws, n, "percent")

    adv_peer0 = _flags(loaded["adv_peer"], items, 0, (2,)).ravel()
    num, den = _conditional(values["adv_peer"][0], values["adv_peer"][1],
                            adv_peer0, True, False, False)
    point, draws, n = _ratio_point_draws(num, den, weights)
    add("adversarial_uptake", point, draws, n, "percent")

    # Contaminated-panel control: slot 0 is the sole honest Gemma defender in
    # [G,G_adv,Q]. Compare it with the same slot in [G,G,G_adv].
    same0 = {rd: _flags(loaded["same_family_adv"], items, rd, (0,))
             for rd in range(5)}
    contam0 = {rd: _flags(loaded["contaminated"], items, rd, (0,))
               for rd in range(5)}
    for rd in (1, 4):
        contrast = ((contam0[rd] - contam0[0])
                    - (same0[rd] - same0[0])).ravel() * 100.0
        add(f"r{rd}_contaminated_honest_qwen_effect", float(contrast.mean()),
            _draw_mean(weights, contrast), len(items), "pp")

    return rows


def main() -> None:
    rows = compute()
    display = []
    for row in rows:
        display.append({
            "metric": row["metric"],
            "estimate": round(row["estimate"], 4),
            "ci_low": round(row["ci_low"], 4),
            "ci_high": round(row["ci_high"], 4),
            "n": row["n"],
            "unit": row["unit"],
        })
        print(f"  {row['metric']:52s} {row['estimate']:8.2f} "
              f"[{row['ci_low']:7.2f}, {row['ci_high']:7.2f}]  n={row['n']}")
    traces.write_csv(OUT, list(display[0]), display)


if __name__ == "__main__":
    main()
