#!/usr/bin/env python3
"""Generate full-precision numerical inputs for all four manuscript figures.

Presentation CSVs intentionally round values for tables.  Figures must never
read those rounded cells: this script recomputes their coordinates directly
from the traces and writes a separate, auditable plot-data layer.
"""
from __future__ import annotations

from artifact_paths import RESULTS
import panels
import prose_quantities as pq
import table_confirmatory as tc
import table_directions as td
import table_rates as tr
import table_rounds
import table_specifications as ts
import table_reception as trc
import traces


def discovery_rows() -> list[dict]:
    by_key = {r["key"]: r for r in pq.per_direction()}
    rows = []
    for d in (x for x in panels.DIRECTIONS if not x.get("fixed_receiver")):
        support, loaded = td.matched_support(d)
        n_un, k_un = trc.exposures(loaded, support, "mixed", False, False)
        cor_hom = tr.transitions(loaded["d_homo"], support)[1]
        cor_adv = tr.transitions(loaded["adv"], support)[1]
        r = by_key[d["key"]]
        rows.append({
            "key": d["key"], "direction": d["label"],
            "gamma": r["gamma"], "gain": r["gain"],
            "adv_gain": r["adv_gain"], "rho": r["rho"],
            "unaided": 100.0 * k_un / n_un,
            "uptake": r["uptake"],
            "adv_excess_corruption": cor_adv - cor_hom,
        })
    return rows


def specification_rows(discovery: list[dict]) -> list[dict]:
    gain = [r["gain"] for r in discovery]
    predictors = {
        "Gamma+": [max(0.0, r["gamma"]) for r in discovery],
        "Gamma+ x rho": [max(0.0, r["gamma"]) * r["rho"] / 100.0
                           for r in discovery],
    }
    rows = []
    for name, xs in predictors.items():
        slope = ts.ols_slope(xs, gain)
        intercept = sum(gain) / len(gain) - slope * sum(xs) / len(xs)
        rows.append({"predictor": name, "slope": slope,
                     "intercept": intercept,
                     "r_squared": ts.r_squared(xs, gain)})
    return rows


def ceiling_rows() -> list[dict]:
    rows = []
    for label, key in table_rounds.SETTINGS:
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        values, n = table_rounds.series(d)
        best = {name: max(v for v in rounds.values() if v is not None)
                for name, rounds in values.items()}
        rows.append({"setting": label, "n": n,
                     "peer_in_panel": best["peer_in_panel"],
                     "panel_two": best["panel_two"],
                     "oracle": best["oracle"]})
    return rows


def confirmatory_rows() -> list[dict]:
    cols = tc.compute()
    rows = []
    for peer in ("Qwen", "Gemma", "OSS-120B"):
        c = cols[peer]
        rows.append({
            "peer": peer, "gamma": c["Gamma"],
            "r1_gain": c["r1_gain"], "r4_gain": c["r4_gain"],
            "rho": "" if c["rho"] is None else c["rho"],
            "rho_lo": "" if c["rho_lo"] is None else c["rho_lo"],
            "rho_hi": "" if c["rho_hi"] is None else c["rho_hi"],
            "upsilon": "" if c["upsilon"] is None else c["upsilon"],
            "upsilon_lo": "" if c["upsilon_lo"] is None else c["upsilon_lo"],
            "upsilon_hi": "" if c["upsilon_hi"] is None else c["upsilon_hi"],
        })
    return rows


def main() -> None:
    discovery = discovery_rows()
    outputs = [
        ("plot_discovery.csv", discovery),
        ("plot_specifications.csv", specification_rows(discovery)),
        ("plot_ceiling.csv", ceiling_rows()),
        ("plot_confirmatory.csv", confirmatory_rows()),
    ]
    for name, rows in outputs:
        traces.write_csv(RESULTS / name, list(rows[0]), rows)


if __name__ == "__main__":
    main()
