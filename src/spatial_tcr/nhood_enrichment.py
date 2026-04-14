import matplotlib.pyplot as plt
import nichepca as npc
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm.auto import tqdm


def fill_triangle(X):
    X = X + X.T - np.diag(np.diag(X))
    return X


def graph_conv(A, X, n_iter=1):
    for _ in range(n_iter):
        X = (A @ X).T @ X
    return npc.utils.to_numpy(X)


def nhood_enrichment_helper(X, A, seed, permute_per_col=False):
    rng = np.random.RandomState(seed)
    if not permute_per_col:
        perm = rng.permutation(X.shape[0])
        X_perm = X[perm]
    else:
        # X_perm = scipy.sparse.lil_matrix(X.shape)
        # # Permute each column independently
        # for col in range(X.shape[1]):
        #     perm = rng.permutation(X.shape[0])  # Generate a permutation of row indices
        #     X_perm[:, col] = X[perm, col]  # Apply the permutation to each column
        # X_perm = X_perm.tocsr()
        col_indices = np.arange(X.shape[1])
        row_perms = np.empty(X.shape, dtype=int)
        for i in range(X.shape[1]):
            row_perms[:, i] = rng.permutation(X.shape[0])
        X_perm = X[row_perms, col_indices]
    counts = graph_conv(A, X_perm, n_iter=1)
    return counts


def nhood_enrichment(
    adata, obs_key=None, var_names=None, binarize=False, n_perms=1000, n_jobs=-1, c=0.1
):
    if obs_key is not None:
        dummies = pd.get_dummies(adata.obs[obs_key])
        labels = dummies.columns.tolist()
        X = scipy.sparse.csr_matrix(dummies.values.astype(int))
        permute_per_col = False
        A = adata.obsp["spatial_connectivities"].astype(int)
    elif var_names is not None:
        # genes are continuous and can be expressed in the same cell
        X = scipy.sparse.csr_matrix(adata[:, var_names].X)
        if binarize:
            X = (X > 0).astype(int)
        labels = var_names if isinstance(var_names, list) else [var_names]
        permute_per_col = True
        # modify A to distinguish self from other via a dist kernel
        A = adata.obsp["spatial_distances"].copy()
        A.data = np.exp(-A.data * c)
        print(A.data.max(), A.data.min())
    else:
        raise ValueError("Either obs_key or var_names must be provided.")

    seeds = np.arange(n_perms)

    counts_obs = graph_conv(A, X, n_iter=1)

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(nhood_enrichment_helper)(
                X, A, seed, permute_per_col=permute_per_col
            )
            for seed in tqdm(seeds)
        )

    counts_sim = np.stack(results)
    E_sim = np.mean(counts_sim, axis=0)
    std_sim = np.std(counts_sim, axis=0)
    z_scores = (counts_obs - E_sim) / std_sim

    indices = np.triu_indices_from(z_scores, k=0)

    p_values_high = ((counts_sim >= counts_obs).sum(axis=0) + 1) / (n_perms + 1)
    p_values_low = ((counts_sim < counts_obs).sum(axis=0) + 1) / (n_perms + 1)
    p_values = np.minimum(p_values_high, p_values_low)

    # TODO multiple testing correction
    p_values_high_flat = p_values_high[indices]
    p_values_low_flat = p_values_low[indices]

    q_values_high_flat = multipletests(p_values_high_flat, method="fdr_bh")[1]
    q_values_low_flat = multipletests(p_values_low_flat, method="fdr_bh")[1]

    q_values_high = np.zeros_like(p_values)
    q_values_low = np.zeros_like(p_values)

    q_values_high[indices] = q_values_high_flat
    q_values_low[indices] = q_values_low_flat

    # make symmetric
    q_values_high = fill_triangle(q_values_high)
    q_values_low = fill_triangle(q_values_low)

    q_values = np.minimum(q_values_high, q_values_low)

    adata.uns["nhood_enrichment"] = {
        "counts_obs": pd.DataFrame(counts_obs, columns=labels, index=labels),
        "counts_sim": counts_sim,
        "z_scores": pd.DataFrame(z_scores, columns=labels, index=labels),
        "p_values": pd.DataFrame(p_values, columns=labels, index=labels),
        "q_values": pd.DataFrame(q_values, columns=labels, index=labels),
    }


def nhood_enrichment_dotplot(adata, figsize=(6, 4)):
    z_scores = adata.uns["nhood_enrichment"]["z_scores"]
    q_values = adata.uns["nhood_enrichment"]["q_values"]
    # p_values = adata.uns["nhood_enrichment"]["p_values"]

    z_scores_melted = z_scores.reset_index().melt(
        id_vars="index", var_name="col", value_name="z_score"
    )
    q_values_melted = q_values.reset_index().melt(
        id_vars="index", var_name="col", value_name="q_value"
    )

    df = pd.merge(z_scores_melted, q_values_melted, on=["index", "col"])
    df["size"] = 1 / df["q_value"]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x="col",
        y="index",
        size="size",
        hue="z_score",
        palette="coolwarm",
        sizes=(20, 200),
        legend=False,
        ax=ax,
    )

    return ax


def p_values_to_significance(p_values):
    """
    Convert an array of p-values to an array of significance strings.

    Parameters
    ----------
    p_values (array-like): Array of p-values.

    Returns
    -------
    significance (array-like): Array of significance strings.
    """
    significance = np.empty(
        p_values.shape, dtype=object
    )  # Create an empty array of the same shape

    for i in range(p_values.size):
        if p_values.flat[i] <= 0.001:
            significance.flat[i] = "***"  # Highly significant
        elif p_values.flat[i] <= 0.01:
            significance.flat[i] = "**"  # Significant
        elif p_values.flat[i] <= 0.05:
            significance.flat[i] = "*"  # Marginally significant
        else:
            significance.flat[i] = "ns"  # Not significant

    return significance


def nhood_enrichment_heatmap(
    adata, figsize=(6, 4), cmap="coolwarm", center=0, vmin=None, vmax=None
):
    z_scores = adata.uns["nhood_enrichment"]["z_scores"]
    q_values = adata.uns["nhood_enrichment"]["q_values"]

    if vmin is not None:
        z_scores = np.clip(z_scores, a_min=vmin, a_max=None)
    if vmax is not None:
        z_scores = np.clip(z_scores, a_max=vmax, a_min=None)

    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            z_scores,
            annot=False,
            cmap=cmap,
            ax=ax,
            center=center,
            square=True,
            cbar_kws={"label": "z-score"},
        )

        # plot significance on top
        significance = p_values_to_significance(q_values.values)
        for i in range(z_scores.shape[0]):
            for j in range(z_scores.shape[1]):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    significance[i, j],
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    return ax
