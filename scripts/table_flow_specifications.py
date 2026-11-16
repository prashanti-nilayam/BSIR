#!/usr/bin/env python3
"""Descriptive alternatives to the aggregate-gap product specification.

All four specifications use the same nine discovery directions and controlled
R0->R1 gain as the outcome.  S and T are the receiver-wrong/peer-correct and
receiver-correct/peer-wrong masses in the honest heterogeneous panel; rho and
eta are their observed conditional correction and corruption rates.  Because
rho and eta use R1 outcomes, these are accounting diagnostics rather than
independent predictive models.

Output: results/flow_specifications.csv
"""
from __future__ import annotations

from artifact_paths import RESULTS
import panels
import table_directions as td
import table_gate as tg
import table_reception as tr
import table_specifications as ts
import traces

OUT = RESULTS / "flow_specifications.csv"


def _fit(name: str, xs: list[float], ys: list[float]) -> dict:
    slope = ts.ols_slope(xs, ys)
    intercept = sum(ys) / len(ys) - slope * sum(xs) / len(xs)
    predictions = [intercept + slope * x for x in xs]
    return {
        "predictor": name,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(ts.r_squared(xs, ys, slope), 4),
        "max_residual": round(max(abs(y - prediction)
                                  for y, prediction in zip(ys, predictions)), 2),
    }


def main() -> None:
    predictors = {
        "Gamma+ x rho": [],
        "mixed_gap+ x rho": [],
        "S x rho": [],
        "S x rho - T x eta": [],
    }
    gains = []
    for direction in panels.DIRECTIONS:
        if direction.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(direction)
        total = sum(len(items) for items in support.values()) * len(panels.DEFENDER_SLOTS)
        gamma = (td.r0_accuracy(loaded["p_homo"], support)
                 - td.r0_accuracy(loaded["d_homo"], support))
        n_s, k_s = tr.exposures(loaded, support, "mixed", True, False)
        n_t, k_t = tr.exposures(loaded, support, "mixed", False, True)
        rho = k_s / n_s
        mixed_gap = 100.0 * (n_s - n_t) / total
        useful_flow = 100.0 * k_s / total
        opposing_flow = 100.0 * k_t / total
        predictors["Gamma+ x rho"].append(max(0.0, gamma) * rho)
        predictors["mixed_gap+ x rho"].append(max(0.0, mixed_gap) * rho)
        predictors["S x rho"].append(useful_flow)
        predictors["S x rho - T x eta"].append(useful_flow - opposing_flow)
        gains.append(tg.net_gain(loaded, support, "mixed"))

    rows = [_fit(name, values, gains) for name, values in predictors.items()]
    for row in rows:
        print(
            f"  {row['predictor']:24s} slope={row['slope']:7.4f} "
            f"intercept={row['intercept']:+7.4f} R2={row['r_squared']:.4f} "
            f"max-residual={row['max_residual']:.2f}"
        )
    traces.write_csv(OUT, list(rows[0]), rows)


if __name__ == "__main__":
    main()
