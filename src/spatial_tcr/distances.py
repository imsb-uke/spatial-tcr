import warnings
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.spatial import KDTree
from tqdm.auto import tqdm


def distance_matrix(
    adata, obs_key, sample_key=None, labels=None, cutoff=0.05, n_jobs=1, **kwargs
):
    # check if obs_key is iterable
    if not isinstance(obs_key, Iterable) or isinstance(obs_key, str):
        obs_keys = [obs_key]
    else:
        obs_keys = obs_key

    # create a temporary adata object with the obs_keys to avoid memory issues
    ad_tmp = sc.AnnData(
        obs=adata.obs[obs_keys], obsm={"spatial": adata.obsm["spatial"]}
    )

    if sample_key is not None:
        ad_tmp.obs[sample_key] = adata.obs[sample_key]
        samples = ad_tmp.obs[sample_key].unique()
    else:
        samples = ["0"]

    # limit n_jobs to the number of samples
    n_jobs = min(n_jobs, len(samples))

    all_results_per_cell = []

    if len(obs_keys) == 1:
        if labels is None:
            labels = ad_tmp.obs[obs_keys[0]].unique()
    else:
        labels = obs_keys

    # iterate over samples
    for sample in tqdm(samples):
        if len(samples) == 1:
            ad_sub = ad_tmp
        else:
            ad_sub = ad_tmp[ad_tmp.obs[sample_key] == sample].copy()

        # handle categorical vs multiple continuous obs_keys
        if len(obs_keys) == 1:
            coords = ad_sub.obsm["spatial"]
            obs = ad_sub.obs[obs_keys]
            # rename to column
            obs.columns = ["column"]
        else:
            label_props = ad_sub.obs[obs_keys]
            mask = label_props > cutoff
            melted = pd.melt(
                mask.reset_index(),
                id_vars="index",  # Keeps the index as a column
                var_name="column",  # Names for melted column
                value_name="value",  # Names for melted values
            )
            obs = melted[melted.value].reset_index(drop=True)
            coords = ad_sub[obs["index"]].obsm["spatial"].toarray()

        result_per_cell = distance_matrix_fn(obs, coords, labels, **kwargs)
        result_per_cell["sample"] = sample
        all_results_per_cell.append(result_per_cell)
    # Catch warnings in the block
    with warnings.catch_warnings(record=True):
        # Filter the specific warning
        warnings.simplefilter("always", FutureWarning)  # Catch only FutureWarnings
        warnings.simplefilter("always", RuntimeWarning)  # Catch RuntimeWarnings

        # Code that might trigger warnings
        results_combined = pd.concat(all_results_per_cell, axis=0)

        result_per_sample = results_combined.groupby(["sample", "cell_type"]).median()
        result = (
            results_combined.drop(columns=["sample"]).groupby(["cell_type"]).median()
        )
        # reorder index to match column order
        result = result.reindex(labels)

    return result, result_per_sample, results_combined


def nearest_from_B(A: np.ndarray, B: np.ndarray, k=1, agg="median"):
    tree = KDTree(B)
    dist, idx = tree.query(A, k=k)
    if k > 1:
        if agg == "median":
            dist = np.median(dist, axis=1)
        elif agg == "mean":
            dist = np.mean(dist, axis=1)
        else:
            raise ValueError(f"Invalid aggregation method: {agg}")
    return idx, dist


def distance_matrix_fn(df, coords, labels, key="column", **kwargs):
    # now we are back in the "single-cell" case
    # calc all distances
    # distances = scipy.spatial.distance_matrix(coords, coords)
    key_value_per_cell = df[key].values

    # Pre-allocate result DataFrame
    result_per_cell = pd.DataFrame(
        index=range(len(coords)), columns=labels, dtype=float
    )

    # Create boolean masks for each cell type
    for ct in labels:
        mask = key_value_per_cell == ct
        # handle case where there are no cells of type ct
        if mask.sum() == 0:
            result_per_cell[ct] = np.nan
        else:
            # For each cell, find distance to nearest cell of type ct
            # result_per_cell[ct] = np.min(distances[:, mask], axis=1)
            idx, dist = nearest_from_B(coords, coords[mask])
            result_per_cell[ct] = dist

    result_per_cell["cell_type"] = df[key].values
    return result_per_cell


def plot_celltype_distances(ct, dist_mat):
    other_cts = [c for c in dist_mat.columns if c != ct]

    plot_df = pd.melt(
        dist_mat.loc[[ct], other_cts].reset_index(),
        id_vars="cell_type",
        var_name="other_cell_type",
        value_name="distance",
    )
    plot_df = plot_df.sort_values(by="distance", ascending=True)
    sns.set_theme(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Normalize the distance values to [0,1] for the colormap
    norm = plt.Normalize(plot_df["distance"].min(), plot_df["distance"].max())
    colors = list(plt.cm.coolwarm_r(norm(plot_df["distance"])))

    sns.barplot(
        data=plot_df,
        x="other_cell_type",
        y="distance",
        hue="other_cell_type",
        # hue="cell_type",
        ax=ax,
        palette=colors,
    )
    # rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Distance [microns]")
    ax.set_xlabel("Other Cell Type")
    ax.set_title(f"Estimated distance from {ct} to other cell types")
    # ax.legend(title="Cell Type", loc="upper left", frameon=False)

    # remove legend
    # ax.legend_.remove()

    sns.despine()
    plt.show()
