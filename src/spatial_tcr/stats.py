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


def count_test(pos_1, neg_1, pos_2, neg_2, test="auto"):
    table = np.array([[pos_1, neg_1], [pos_2, neg_2]])
    if test == "auto":
        if np.sum(table) < 1000:
            test = "fisher"
        else:
            test = "chisquare"
    if test == "chisquare":
        from scipy.stats import chi2_contingency

        p_value = chi2_contingency(table)[1]
    elif test == "fisher":
        from scipy.stats import fisher_exact

        p_value = fisher_exact(table)[1]
    return p_value


def wilcoxon_significance(df, target_ct, group_key="group"):
    from scipy.stats import mannwhitneyu

    assert target_ct in df.columns, f"{target_ct} not in {df.columns}"
    assert group_key in df.columns, f"{group_key} not in {df.columns}"

    # perform significance test
    groups = df[group_key].unique()

    ct_cols = [ct for ct in df.columns if ct != group_key]
    row_sums = df[ct_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Row sums should be close to 1, got max of {row_sums.max()}"
    )

    group_values = [df[df[group_key] == g][target_ct].values for g in groups]

    stat, p = mannwhitneyu(group_values[0], group_values[1], alternative="two-sided")
    return p
