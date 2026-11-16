#!/usr/bin/env python3
"""Competing specifications for honest gain across the nine discovery directions.

Each predictor is fit by least squares with an intercept, with
R^2 the squared Pearson correlation. Significance is an exact permutation test -- all 9!
pairings of predictor to outcome, two-sided on the Pearson correlation. The
LOO column is the range of slopes when each direction is dropped in turn.

  Gamma+        peer capability advantage, floored at zero (pp)
  rho           reception power (fraction)
  Gamma+ x rho  their product

Output: results/specifications.csv
"""
import itertools
import math

from artifact_paths import RESULTS
import panels
import table_directions as td
import table_gate as tg
import table_reception as trc
import traces

OUT = RESULTS / "specifications.csv"


def ols_slope(xs, ys):
    """Least-squares slope with an intercept."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            if sxx else float("nan"))


def r_squared(xs, ys, slope=None):
    """Coefficient of determination for the fit with an intercept."""
    r = pearson(xs, ys)
    return r * r


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def exact_permutation_p(xs, ys):
    """Two-sided p over all n! pairings, on the Pearson correlation."""
    n = len(xs)
    mx = sum(xs) / n
    xc = [x - mx for x in xs]
    my = sum(ys) / n
    yc = [y - my for y in ys]
    obs = abs(sum(a * b for a, b in zip(xc, yc)))
    total = extreme = 0
    for perm in itertools.permutations(yc):
        total += 1
        if abs(sum(a * b for a, b in zip(xc, perm))) >= obs - 1e-12:
            extreme += 1
    return extreme / total


def loo_slopes(xs, ys):
    out = []
    for i in range(len(xs)):
        sx = xs[:i] + xs[i + 1:]
        sy = ys[:i] + ys[i + 1:]
        out.append(ols_slope(sx, sy))
    return min(out), max(out)


def main() -> None:
    gammas, rhos, gains = [], [], []
    for d in panels.DIRECTIONS:
        if d.get("fixed_receiver"):
            continue
        support, loaded = td.matched_support(d)
        gamma = (td.r0_accuracy(loaded["p_homo"], support)
                 - td.r0_accuracy(loaded["d_homo"], support))
        n_rho, k_rho = trc.exposures(loaded, support, "mixed", True, False)
        gammas.append(max(0.0, gamma))
        rhos.append(k_rho / n_rho)
        gains.append(tg.net_gain(loaded, support, "mixed"))

    specs = {
        "Gamma+": gammas,
        "rho": rhos,
        "Gamma+ x rho": [g * r for g, r in zip(gammas, rhos)],
    }

    rows = []
    for name, xs in specs.items():
        slope = ols_slope(xs, gains)
        n = len(xs)
        intercept = sum(gains) / n - slope * sum(xs) / n
        lo, hi = loo_slopes(xs, gains)
        rows.append({
            "predictor": name,
            "slope": round(slope, 3),
            "intercept": round(intercept, 6),
            "r_squared": round(r_squared(xs, gains, slope), 4),
            "exact_p": round(exact_permutation_p(xs, gains), 4),
            "loo_slope_min": round(lo, 2),
            "loo_slope_max": round(hi, 2),
        })
        r = rows[-1]
        print(f"  {name:14s} slope={r['slope']:8.3f}  R2={r['r_squared']:.4f}  "
              f"p={r['exact_p']:.4f}  LOO=[{r['loo_slope_min']:.2f}, "
              f"{r['loo_slope_max']:.2f}]")

    traces.write_csv(OUT, list(rows[0].keys()), rows)


if __name__ == "__main__":
    main()
