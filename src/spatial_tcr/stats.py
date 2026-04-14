import numpy as np
from scipy.spatial import cKDTree


def overlap_count(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    return int(np.sum(a & b))


def _local_candidates(coords, radius):
    coords = np.asarray(coords, float)
    tree = cKDTree(coords)
    neigh = tree.query_ball_point(coords, r=radius)
    cand = []
    for i, lst in enumerate(neigh):
        lst = [j for j in lst if j != i]
        if not lst:
            raise ValueError("Increase radius: some nodes have no swap candidates.")
        cand.append(np.array(lst, dtype=int))
    return cand


def _local_swap_perm(x, candidates, n_swaps, rng):
    x = x.copy()
    n = x.size
    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.choice(candidates[i])
        x[i], x[j] = x[j], x[i]
    return x


def colocalization_local_swap_test(
    a, b, coords, radius, n_perm=5000, swaps_per_perm=None, seed=0, symmetric=True
):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1D arrays over the same nodes")

    rng = np.random.default_rng(seed)
    cand = _local_candidates(coords, radius)
    if swaps_per_perm is None:
        swaps_per_perm = 10 * a.size

    obs = overlap_count(a, b)
    null = np.empty(n_perm, dtype=int)

    for k in range(n_perm):
        if symmetric:
            a_perm = _local_swap_perm(a, cand, swaps_per_perm, rng)
            b_perm = _local_swap_perm(b, cand, swaps_per_perm, rng)
            null[k] = overlap_count(a_perm, b_perm)
        else:
            b_perm = _local_swap_perm(b, cand, swaps_per_perm, rng)
            null[k] = overlap_count(a, b_perm)

    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return {
        "observed_overlap": int(obs),
        "null_mean": float(null.mean()),
        "p_value": float(p),
        "null": null,
    }
