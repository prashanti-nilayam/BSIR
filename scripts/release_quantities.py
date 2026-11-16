#!/usr/bin/env python3
"""Compute camera-ready additions without changing the original fitted populations."""
from __future__ import annotations
import csv
import statistics
from artifact_paths import REPO, RESULTS
import traces


def read(name):
    return list(csv.DictReader((RESULTS / f"{name}.csv").open()))


def quantities():
    same = read("same_family_ladder")[0]
    gaps = {r["direction"]: float(r["Gamma"]) for r in read("directions")}
    gaps[same["direction"]] = float(same["Gamma"])
    positive = [r for r in read("token_cost") if gaps[r["direction"]] > 0]
    ratios = [float(r["total_token_ratio"]) for r in positive]
    values = {f"same_family_{k}": float(same[k]) for k in
              ["n_items", "Gamma", "rho", "rho_n", "rho_lo", "rho_hi", "gain"]}
    values.update(token_positive_directions=len(positive),
                  token_ratio_min=min(ratios), token_ratio_max=max(ratios),
                  token_ratio_median=statistics.median(ratios),
                  token_debate_wins=sum(float(r["debate_best_majority_accuracy"])
                                       > float(r["direct_peer_R0_accuracy"])
                                       for r in positive))
    return values


def main():
    values = quantities()
    claims = list(csv.DictReader((REPO / "claims/release_claims.csv").open()))
    rows = []
    for claim in claims:
        q, digits = claim["quantity"], int(claim["decimals"])
        measured = round(values[q], digits)
        expected = round(float(claim["paper"]), digits)
        if measured != expected:
            raise ValueError(f"camera-ready claim mismatch: {q}: {measured} vs {expected}")
        rows.append(dict(quantity=q, value=f"{measured:.{digits}f}",
                         paper=claim["paper"], agrees="YES"))
    traces.write_csv(RESULTS / "release_quantities.csv",
                     ["quantity", "value", "paper", "agrees"], rows)
    print(f"PASS: {len(rows)} camera-ready quantities match named claims")


if __name__ == "__main__":
    main()
