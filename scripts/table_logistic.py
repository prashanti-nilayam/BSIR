#!/usr/bin/env python3
"""Item-level logistic models for the role contrasts.

Three specifications, each fit on defender exposures pooled across the nine
discovery directions:

  pair fe three condition -- direction fixed effects, homogeneous as reference,
                             honest and adversarial indicators
  pair fixed effects      -- direction fixed effects, honest as reference,
                             adversarial indicator (adversarial vs honest only)
  gap                     -- adversarial indicator interacted with the
                             direction's capability gap, per ten points

Variance is a direction-clustered sandwich with the usual small-cluster
correction, and intervals use t with (clusters - 1) degrees of freedom.

Every predictor is categorical at the direction level, so the data collapse to
one cell per (direction, condition, outcome) without any loss -- the fit below
is exact, not an approximation.

Output: results/logistic.csv
"""
import math

from artifact_paths import RESULTS
import panels
import table_directions as td
import traces

OUT = RESULTS / "logistic.csv"
BENCHMARK = [d["benchmark"] for d in panels.DIRECTIONS
             if not d.get("fixed_receiver")]
CONDITIONS = ("d_homo", "mixed", "adv")


# ---------- small dense linear algebra -------------------------------------

def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            raise ValueError("singular design")
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col] / p
            if f:
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def inverse(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ---------- t distribution --------------------------------------------------

def _betacf(a, b, x, iters=300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Two-sided tail probability."""
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df, p=0.975, lo=0.0, hi=100.0):
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - t_sf(mid, df) / 2.0 < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- weighted logistic with clustered sandwich ----------------------

def fit(cells, n_params, n_clusters):
    """cells: list of (x, cluster, n_success, n_fail)."""
    beta = [0.0] * n_params
    for _ in range(200):
        XtWX = [[0.0] * n_params for _ in range(n_params)]
        score = [0.0] * n_params
        for x, _c, n1, n0 in cells:
            eta = sum(b * xi for b, xi in zip(beta, x))
            p = 1.0 / (1.0 + math.exp(-eta)) if eta > -700 else 0.0
            w = (n1 + n0) * p * (1 - p)
            r = n1 - (n1 + n0) * p
            for i in range(n_params):
                if not x[i]:
                    continue
                score[i] += x[i] * r
                for j in range(n_params):
                    if x[j]:
                        XtWX[i][j] += w * x[i] * x[j]
        step = solve(XtWX, score)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-11:
            break

    bread = inverse(XtWX)
    meat = [[0.0] * n_params for _ in range(n_params)]
    by_cluster = {}
    for x, c, n1, n0 in cells:
        eta = sum(b * xi for b, xi in zip(beta, x))
        p = 1.0 / (1.0 + math.exp(-eta))
        r = n1 - (n1 + n0) * p
        acc = by_cluster.setdefault(c, [0.0] * n_params)
        for i in range(n_params):
            acc[i] += x[i] * r
    for acc in by_cluster.values():
        for i in range(n_params):
            for j in range(n_params):
                meat[i][j] += acc[i] * acc[j]

    n_obs = sum(n1 + n0 for _x, _c, n1, n0 in cells)
    g = n_clusters
    corr = (g / (g - 1.0)) * ((n_obs - 1.0) / (n_obs - n_params))
    var = [[corr * sum(bread[i][k] * meat[k][l] * bread[l][j]
                       for k in range(n_params) for l in range(n_params))
            for j in range(n_params)] for i in range(n_params)]
    se = [math.sqrt(var[i][i]) for i in range(n_params)]
    return beta, se, n_obs


# ---------- data ------------------------------------------------------------

def exposure_cells(outcome):
    """[(direction_index, condition, gamma, n_success, n_fail)]"""
    out = []
    for idx, d in enumerate(x for x in panels.DIRECTIONS
                            if not x.get("fixed_receiver")):
        support, loaded = td.matched_support(d)
        gamma = (td.r0_accuracy(loaded["p_homo"], support)
                 - td.r0_accuracy(loaded["d_homo"], support))
        for cond in CONDITIONS:
            reps = dict(loaded[cond])
            n1 = n0 = 0
            for rep, items in support.items():
                recs = reps.get(rep) or loaded[cond][0][1]
                for item in items:
                    for s in panels.DEFENDER_SLOTS:
                        r0, r1 = recs.get((item, 0, s)), recs.get((item, 1, s))
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
                        n1 += y
                        n0 += 1 - y
            out.append((idx, cond, gamma, n1, n0))
    return out


def main() -> None:
    n_dir = sum(1 for x in panels.DIRECTIONS if not x.get("fixed_receiver"))
    rows = []

    for outcome in ("recovery", "corruption"):
        cells = exposure_cells(outcome)

        # --- pair fe, three conditions (homogeneous as reference) -----------
        design = []
        for idx, cond, _g, n1, n0 in cells:
            x = [1.0] + [1.0 if idx == k else 0.0 for k in range(1, n_dir)]
            x += [1.0 if cond == "mixed" else 0.0,
                  1.0 if cond == "adv" else 0.0]
            design.append((x, idx, n1, n0))
        beta, se, n_obs = fit(design, n_dir + 2, n_dir)
        tc = t_crit(n_dir - 1)
        for name, k in (("honest vs homogeneous", n_dir),
                        ("adversarial vs homogeneous", n_dir + 1)):
            rows.append(_row(outcome, "pair fe three condition", name,
                             n_obs, beta[k], se[k], tc, n_dir))

        # --- pair fe, adversarial vs honest ---------------------------------
        sub = [(i, c, g, n1, n0) for i, c, g, n1, n0 in cells if c != "d_homo"]
        design = []
        for idx, cond, _g, n1, n0 in sub:
            x = [1.0] + [1.0 if idx == k else 0.0 for k in range(1, n_dir)]
            x += [1.0 if cond == "adv" else 0.0]
            design.append((x, idx, n1, n0))
        beta, se, n_obs = fit(design, n_dir + 1, n_dir)
        rows.append(_row(outcome, "pair fixed effects",
                         "adversarial vs honest", n_obs,
                         beta[n_dir], se[n_dir], tc, n_dir))

        # --- gap specification ----------------------------------------------
        # Includes a benchmark indicator: the directions span two datasets, and
        # without it the dataset difference loads onto the gap terms.
        design = []
        for idx, cond, g, n1, n0 in sub:
            adv = 1.0 if cond == "adv" else 0.0
            gsm8k = 1.0 if BENCHMARK[idx] == "gsm8k" else 0.0
            x = [1.0, adv, g / 10.0, adv * g / 10.0, gsm8k]
            design.append((x, idx, n1, n0))
        beta, se, n_obs = fit(design, 5, n_dir)
        rows.append(_row(outcome, "gap", "adversarial vs honest", n_obs,
                         beta[1], se[1], tc, n_dir))
        rows.append(_row(outcome, "gap", "adversarial x Gamma", n_obs,
                         beta[3], se[3], tc, n_dir))

    order = {("recovery", "pair fe three condition"): 0,
             ("corruption", "pair fe three condition"): 1,
             ("recovery", "pair fixed effects"): 2,
             ("corruption", "pair fixed effects"): 3,
             ("recovery", "gap"): 4, ("corruption", "gap"): 5}
    rows.sort(key=lambda r: order[(r["outcome"], r["specification"])])

    traces.write_csv(OUT, ["outcome", "specification", "term", "n", "odds_ratio",
                           "ci_lo", "ci_hi", "p"], rows)
    for r in rows:
        print(f"  {r['outcome']:11s} {r['specification']:24s} {r['term']:28s} "
              f"n={r['n']:6d} OR={r['odds_ratio']:7.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] p={r['p']:.4f}")


def _row(outcome, spec, term, n, b, se, tc, n_dir):
    return {
        "outcome": outcome, "specification": spec, "term": term, "n": n,
        "odds_ratio": round(math.exp(b), 3),
        "ci_lo": round(math.exp(b - tc * se), 3),
        "ci_hi": round(math.exp(b + tc * se), 3),
        "p": round(t_sf(abs(b / se), n_dir - 1), 4),
    }


if __name__ == "__main__":
    main()
