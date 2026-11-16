"""Item-clustered Poisson bootstrap, reproducible from the seed alone.

Resampling is at the benchmark-item level: one Poisson(1) weight per cluster
per replicate, shared across every direction and condition that item appears
in, so a direction's replicates are never resampled independently of one
another.

Determinism
-----------
Weights are assigned to clusters in **sorted cluster order**. The cluster key
is "{benchmark}:{item}", exactly as elsewhere in the analysis. Sorting is the
whole point: it makes the weight-to-item mapping a function of the seed and the
data, so anyone holding the traces and the seed gets identical endpoints. An
insertion-ordered index would instead depend on the order rows happened to be
assembled, which is a property of one implementation rather than of the data.

numpy is required here and only here. Poisson draws from PCG64 are what make
the interval reproducible across machines; the rest of the result scripts are
standard library only.
"""
from __future__ import annotations

SEED = 260719826
REPS = 4000
LOW, HIGH = 2.5, 97.5


def cluster_key(benchmark: str, item) -> str:
    return f"{benchmark}:{item}"


def weight_matrix(clusters: list[str], seed: int = SEED, reps: int = REPS):
    """(reps x len(clusters)) Poisson weights, indexed by sorted cluster order."""
    import numpy as np
    order = sorted(clusters)
    rng = np.random.default_rng(seed)
    return order, rng.poisson(1.0, size=(reps, len(order))).astype(float)


def ratio_interval(per_cluster, clusters, seed: int = SEED, reps: int = REPS):
    """Percentile interval for a ratio of cluster-summed counts.

    per_cluster maps cluster -> (numerator, denominator).
    """
    import numpy as np
    order, weights = weight_matrix(clusters, seed, reps)
    num = np.array([per_cluster.get(c, (0.0, 0.0))[0] for c in order])
    den = np.array([per_cluster.get(c, (0.0, 0.0))[1] for c in order])
    draws_num = weights @ num
    draws_den = weights @ den
    valid = draws_den > 0
    draws = draws_num[valid] / draws_den[valid]
    return (float(np.percentile(draws, LOW)) * 100.0,
            float(np.percentile(draws, HIGH)) * 100.0)


def mean_of_direction_stats(per_direction, clusters, stat,
                            seed: int = SEED, reps: int = REPS):
    """Percentile interval for an equal-weight mean across directions.

    per_direction: list of {cluster: counts-tuple}, one entry per direction.
    stat: callable(list-of-resampled-count-arrays) -> per-direction value array.
    """
    import numpy as np
    order, weights = weight_matrix(clusters, seed, reps)
    pos = {c: i for i, c in enumerate(order)}
    per_dir_draws = []
    for counts in per_direction:
        arrays = []
        n_fields = len(next(iter(counts.values())))
        for f in range(n_fields):
            v = np.zeros(len(order))
            for c, tup in counts.items():
                v[pos[c]] = tup[f]
            arrays.append(weights @ v)
        per_dir_draws.append(stat(arrays))
    stacked = np.vstack(per_dir_draws)              # (directions, reps)
    means = np.nanmean(stacked, axis=0)
    means = means[np.isfinite(means)]
    return float(np.percentile(means, LOW)), float(np.percentile(means, HIGH))
