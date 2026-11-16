#!/usr/bin/env python3
"""Discordant-state accounting for honest heterogeneous panels.

For each receiver exposure, the peer and receiver R0 states define four cells.
This table isolates the two discordant cells:

  supply mass S = P(receiver wrong, peer correct)
  opposing mass T = P(receiver correct, peer wrong)
  rho            = P(receiver correct at R1 | S)
  eta            = P(receiver wrong at R1 | T), with an honest peer

The realized correction and corruption flows are S*rho and T*eta.  These are
descriptive components of the R0->R1 transition, not independent predictors of
that transition.  Discovery rows use the primary matched R0/R1 support; the two
fixed-receiver rows use their corresponding matched support; MuSiQue uses the
fully matched 300-item scope cohort.

Output: results/discordant_flows.csv
"""
from __future__ import annotations

from artifact_paths import RESULTS
import panels
import table_directions as td
import table_gate as tg
import table_musique_heterogeneous as tm
import table_reception as tr
import traces

OUT = RESULTS / "discordant_flows.csv"


def _rate(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def _panel_row(direction: dict) -> dict:
    support, loaded = td.matched_support(direction)
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

    return {
        "cohort": "fixed receiver" if direction.get("fixed_receiver") else "discovery",
        "direction": direction["label"],
        "Gamma": round(gamma, 2),
        "n_exposures": total,
        "S_mass": round(_rate(n_s, total), 2),
        "rho": round(_rate(k_s, n_s), 2),
        "S_rho": round(_rate(k_s, total), 2),
        "T_mass": round(_rate(n_t, total), 2),
        "mixed_gap": round(_rate(n_s - n_t, total), 2),
        "eta": round(_rate(k_t, n_t), 2),
        "T_eta": round(_rate(k_t, total), 2),
        "discordant_net": round(_rate(k_s - k_t, total), 2),
        "mixed_change": round(mixed_change, 2),
        "controlled_gain": round(tg.net_gain(loaded, support, "mixed"), 2),
    }


def _musique_row() -> dict:
    loaded = {name: traces.load(run, "musique") for name, run in tm.RUNS.items()}
    items = tm._complete_support(loaded)
    honest = loaded["honest"]
    total = len(items) * len(tm.DEFENDERS)
    counts = {(d, p): [0, 0] for d in (False, True) for p in (False, True)}
    for item in items:
        peer0 = bool(honest[(item, 0, 2)]["ok"])
        for slot in tm.DEFENDERS:
            d0 = bool(honest[(item, 0, slot)]["ok"])
            d1 = bool(honest[(item, 1, slot)]["ok"])
            counts[(d0, peer0)][0] += 1
            counts[(d0, peer0)][1] += int((not d1) if d0 else d1)

    n_s, k_s = counts[(False, True)]
    n_t, k_t = counts[(True, False)]
    n_bw, k_bw = counts[(False, False)]
    n_bc, k_bc = counts[(True, True)]
    if n_s + n_t + n_bw + n_bc != total:
        raise ValueError("discordant cells do not partition MuSiQue")

    def accuracy(panel: str, rd: int) -> float:
        values = [loaded[panel][(item, rd, slot)]["ok"]
                  for item in items for slot in tm.DEFENDERS]
        return 100.0 * sum(values) / len(values)

    # Match the manuscript's capability-gap definition: all three slots in the
    # peer- and receiver-homogeneous panels on the fully matched support.
    def homogeneous_accuracy(panel: str) -> float:
        values = [loaded[panel][(item, 0, slot)]["ok"]
                  for item in items for slot in (0, 1, 2)]
        return 100.0 * sum(values) / len(values)

    gamma = homogeneous_accuracy("p_homo") - homogeneous_accuracy("d_homo")
    mixed_change = 100.0 * (k_s + k_bw - k_t - k_bc) / total
    direct_change = accuracy("honest", 1) - accuracy("honest", 0)
    if abs(mixed_change - direct_change) > 1e-9:
        raise ValueError("transition accounting failed for MuSiQue")
    controlled = direct_change - (accuracy("d_homo", 1) - accuracy("d_homo", 0))

    return {
        "cohort": "MuSiQue scope",
        "direction": "Gemma-3-12B <- Qwen-7B",
        "Gamma": round(gamma, 2),
        "n_exposures": total,
        "S_mass": round(_rate(n_s, total), 2),
        "rho": round(_rate(k_s, n_s), 2),
        "S_rho": round(_rate(k_s, total), 2),
        "T_mass": round(_rate(n_t, total), 2),
        "mixed_gap": round(_rate(n_s - n_t, total), 2),
        "eta": round(_rate(k_t, n_t), 2),
        "T_eta": round(_rate(k_t, total), 2),
        "discordant_net": round(_rate(k_s - k_t, total), 2),
        "mixed_change": round(mixed_change, 2),
        "controlled_gain": round(controlled, 2),
    }


def main() -> None:
    rows = [_panel_row(direction) for direction in panels.DIRECTIONS]
    rows.append(_musique_row())
    for row in rows:
        print(
            f"  {row['cohort']:14s} {row['direction']:28s} "
            f"G={row['Gamma']:+6.2f} S={row['S_mass']:5.2f} "
            f"rho={row['rho']:5.2f} S*rho={row['S_rho']:5.2f} "
            f"T={row['T_mass']:5.2f} eta={row['eta']:5.2f} "
            f"T*eta={row['T_eta']:5.2f} net={row['discordant_net']:+6.2f}"
        )
    traces.write_csv(OUT, list(rows[0]), rows)


if __name__ == "__main__":
    main()
