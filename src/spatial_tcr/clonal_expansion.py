import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nichepca as npc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
from sklearn.cluster import DBSCAN
from spatialtools.spatial import spatial_sample_split
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm.auto import tqdm

from .spatial import annotate_ccs, to_networkx
from .tcr import get_tcr_genes


def get_tv_gene_sets(
    adata,
    av_excluded=None,
    bv_excluded=None,
):
    if av_excluded is None:
        av_excluded = ["TRAV41"]
    if bv_excluded is None:
        bv_excluded = ["TRBV20-1"]
    av_genes = [
        g for g in adata.var_names if g.startswith("TRAV") and (g not in av_excluded)
    ]
    bv_genes = [
        g for g in adata.var_names if g.startswith("TRBV") and (g not in bv_excluded)
    ]
    dv_genes = [g for g in adata.var_names if g.startswith("TRDV")]
    gv_genes = [g for g in adata.var_names if g.startswith("TRGV")]
    tv_genes = av_genes + bv_genes + dv_genes + gv_genes
    return av_genes, bv_genes, dv_genes, gv_genes, tv_genes


def merge_cols(x):
    x = [str(i) for i in x if not pd.isna(i)]
    if len(x) == 0:
        return None
    return "_".join(x)


def clonal_expansion_clusters(
    adata,
    genes=None,
    gene_tuples=None,
    prefix="",
    min_cells=4,
    max_diameter=100,
    radius=None,
    verbose=True,
    save_clusters_per_gene=False,
    prog_bar=True,
    merge_columns=True,
    fillna=False,
    sample_key=None,
):
    if radius is not None:
        if sample_key is None:
            npc.gc.distance_graph(adata, radius=radius, remove_self_loops=False)
        else:
            npc.gc.construct_multi_sample_graph(
                adata, radius=radius, sample_key=sample_key
            )
        npc.gc.to_squidpy(adata)

    if genes is not None:
        gene_tuples = [[gene] for gene in genes]

    df = pd.DataFrame(
        data=None,
        index=adata.obs_names,
        columns=[f"{'+'.join(genes)}_cc" for genes in gene_tuples],
    )

    for genes in tqdm(gene_tuples, disable=not prog_bar):
        genes_str = "+".join(genes)

        mask = np.ones(adata.shape[0], dtype=bool)
        for gene in genes:
            mask *= adata[:, gene].X.toarray().flatten() > 0
        # mask = adata[:, gene].X.toarray().flatten() > 0
        if not np.any(mask):
            if verbose:
                print(f"No cells with {genes_str} expression, skipping...")
            continue

        ad_sub = adata[mask, genes].copy()

        npc.gc.from_squidpy(ad_sub)
        G = to_networkx(ad_sub)

        # get connected components
        annotate_ccs(ad_sub, G)

        cc_sizes = ad_sub.obs["cc"].value_counts()
        relevant_ccs = cc_sizes[cc_sizes >= min_cells].index
        # print(len(relevant_ccs))

        # calculate the diameter of the connected components
        if max_diameter is not None:
            tmp_ccs = []
            for cc in relevant_ccs:
                coords = ad_sub[ad_sub.obs["cc"] == cc].obsm["spatial"]
                # calculate max distance between any two points
                max_dist = scipy.spatial.distance.pdist(coords).max()
                if max_dist < max_diameter:
                    tmp_ccs.append(cc)
            relevant_ccs = tmp_ccs

        if len(relevant_ccs) == 0:
            if verbose:
                print(
                    f"No connected components with {genes_str} expression, skipping..."
                )
            continue
        if verbose:
            print(
                f"{genes_str}, num ccs: {len(relevant_ccs)}, max_size: {cc_sizes[relevant_ccs].max()}, avg_size: {cc_sizes[relevant_ccs].mean().round(2)}"
            )

        # get the indices
        mask = ad_sub.obs["cc"].isin(relevant_ccs)
        indices = ad_sub.obs[mask].index
        if merge_columns:
            df.loc[indices, f"{genes_str}_cc"] = f"{genes_str}_" + ad_sub.obs["cc"]
        else:
            df.loc[indices, f"{genes_str}_cc"] = ad_sub.obs["cc"]

    if save_clusters_per_gene:
        try:
            del adata.obs[df.columns]
        except KeyError:
            pass
        adata.obs.loc[df.index, df.columns] = df

    if merge_columns:
        mask = (~df.isna()).sum(axis=1) > 1
        # set mask rows to nan
        df.loc[mask] = None

        df["merged"] = df.apply(merge_cols, axis=1)

        # (df.merged.isna() == df["TRBV23-1_cc"].isna()).all()

        # filter out all ccs with less t
        mask = df["merged"].value_counts() >= min_cells
        cc_names = mask[mask].index
        ccs = df.loc[df["merged"].isin(cc_names), ["merged"]]

        fill_value = "NA" if fillna else None

        try:
            del adata.obs[f"{prefix}cc"]
            del adata.obs[f"{prefix}cc_class"]
        except KeyError:
            pass

        adata.obs[f"{prefix}cc"] = fill_value
        adata.obs.loc[ccs.index, f"{prefix}cc"] = ccs["merged"]

        adata.obs[f"{prefix}cc_class"] = fill_value
        adata.obs[f"{prefix}cc_class"] = adata.obs[f"{prefix}cc"].apply(
            lambda x: x.split("_")[0] if not pd.isna(x) else fill_value
        )


def aggr_clonal_clusters(adata, cc_key, dist_key):
    df = (
        adata.obs.groupby(cc_key, observed=True)[dist_key]
        .mean()
        .to_frame()
        .reset_index()
    )
    df[f"{cc_key}_class"] = df[cc_key].apply(lambda x: x.split("_")[0])
    return df


def dist_hist(
    adata,
    obs_keys,
    dist_col="dist_to_lob_border",
    title=None,
    xlabel=None,
    ylabel=None,
    bins=50,
    save_path=None,
    return_df=False,
):
    # mask = adata.obs[dist_col].isna()
    if isinstance(obs_keys, str):
        obs_keys = [obs_keys]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for obs_key in obs_keys:
        df = aggr_clonal_clusters(adata, obs_key, dist_col)
        df[dist_col].hist(bins=bins, ax=ax, label=obs_key)
    ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if return_df:
        return df


def filter_ccs(adata, cc_key, filter_key=None, filter_val=None, mask=None):
    # ensure that all ccs are completely inside the masked region
    if mask is None:
        assert filter_key is not None and filter_val is not None
        mask = adata.obs[filter_key] == filter_val

    ccs_to_remove = adata[~mask].obs[cc_key].unique()
    ccs_to_remove = [cc for cc in ccs_to_remove if not pd.isna(cc)]
    adata.obs[f"{cc_key}_filtered"] = adata.obs[cc_key]
    adata.obs.loc[adata.obs[cc_key].isin(ccs_to_remove), f"{cc_key}_filtered"] = None


def calc_statistics(adata, df, dist_key, n_bins=50, n_perms=1000, verbose=True):
    hist_obs = np.histogram(df[dist_key].values[~df[dist_key].isna()], bins=n_bins)
    bins_obs = hist_obs[1]
    counts_obs = hist_obs[0]

    n_clonal_clusters = len(df)

    counts_sim = np.zeros((n_perms, n_bins))
    for i in tqdm(range(n_perms)):
        seed = i
        # sample n_clonal_cluster cells
        rng = np.random.RandomState(seed)
        # random choice
        indices = rng.choice(adata.obs.index, n_clonal_clusters, replace=False)
        distances = adata[indices].obs[dist_key].values
        distances = distances[~np.isnan(distances)]
        hist = np.histogram(distances, bins=bins_obs)
        counts_sim[i] = hist[0]
        if i < 10 and verbose:
            adata[indices].obs[dist_key].hist(bins=n_bins)
    if verbose:
        df[dist_key].hist(bins=n_bins, color="red", alpha=0.5)
        plt.show()

    E_sim = counts_sim.mean(axis=0)
    std_sim = counts_sim.std(axis=0)
    z_scores = (counts_obs - E_sim) / std_sim
    z_scores_sim = (counts_sim - E_sim) / std_sim

    p_values_high = ((z_scores_sim >= z_scores).sum(axis=0) + 1) / (n_perms + 1)
    p_values_low = ((z_scores_sim < z_scores).sum(axis=0) + 1) / (n_perms + 1)

    p_values_high = ((counts_sim >= counts_obs).sum(axis=0) + 1) / (n_perms + 1)
    p_values_low = ((counts_sim < counts_obs).sum(axis=0) + 1) / (n_perms + 1)

    p_values = np.minimum(p_values_high, p_values_low)

    q_values = multipletests(p_values, method="fdr_bh")[1]

    return z_scores, p_values, q_values, bins_obs


def calc_empirical_p_values(counts_obs, counts_sim, n_perms):
    # counts_obs.shape: (n_obs,)
    # counts_sim.shape: (n_perms, n_obs)
    n_perms = len(counts_sim)

    E_sim = counts_sim.mean(axis=0)
    std_sim = counts_sim.std(axis=0)
    z_scores = (counts_obs - E_sim) / std_sim

    p_values_high = ((counts_sim >= counts_obs).sum(axis=0) + 1) / (n_perms + 1)
    p_values_low = ((counts_sim < counts_obs).sum(axis=0) + 1) / (n_perms + 1)

    print(f"p_values_high: {p_values_high}")
    print(f"p_values_low: {p_values_low}")

    p_values = np.minimum(p_values_high, p_values_low)

    q_values = multipletests(p_values, method="fdr_bh")[1]

    return z_scores, p_values, q_values


def compute_cc_info(adata, obs_key, var_names, layer="counts"):
    # make sure obs_key is a list
    if not isinstance(obs_key, list):
        obs_key = [obs_key]

    if layer is not None:
        X = npc.utils.to_numpy(adata[:, var_names].layers[layer])
    else:
        X = npc.utils.to_numpy(adata[:, var_names].X)

    df = adata.obs[obs_key].copy()
    df.loc[:, var_names] = X

    df_bool = adata.obs[obs_key].copy()
    df_bool.loc[:, var_names] = X.astype(bool).astype(float)

    mean_expression = df.groupby(obs_key, observed=True)[var_names].mean()
    fraction_of_cells = df_bool.groupby(obs_key, observed=True)[var_names].mean()

    return mean_expression, fraction_of_cells


# plot specific component in detail
def get_component_view(
    adata, cc, cc_key, expansion, show_bbox=False, sample_key=None, **kwargs
):
    # get cc sample
    if sample_key is not None:
        sample = adata[adata.obs[cc_key] == cc].obs[sample_key].values[0]
        adata = adata[adata.obs[sample_key] == sample].copy()

    x_center, y_center = adata[adata.obs[cc_key] == cc].obsm["spatial"].mean(axis=0)
    # Define the bounds of the square
    x_min = x_center - expansion
    x_max = x_center + expansion
    y_min = y_center - expansion
    y_max = y_center + expansion

    points = adata.obsm["spatial"]
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    ad_sub = adata[mask].copy()

    if show_bbox:
        # get the sample that this belongs to
        if sample_key is None:
            ad_sample = adata
        else:
            ad_sample = adata[adata.obs[sample_key] == sample].copy()
        axs = sc.pl.spatial(ad_sample, color=cc_key, spot_size=20, show=False, **kwargs)
        ax = axs[0]
        # plot bounding box on top
        # ax.plot(
        #     [x_min, x_max, x_max, x_min, x_min],
        #     [y_min, y_min, y_max, y_max, y_min],
        #     color="red",
        #     linestyle="--",
        # )
        # remove legend
        ax.get_legend().remove()
        # Add a rectangle to represent the bounding box
        rect = patches.Rectangle(
            (x_min, y_min),  # Bottom-left corner
            x_max - x_min,  # Width
            y_max - y_min,  # Height
            linewidth=0.5,  # Border thickness
            edgecolor="red",  # Border color
            linestyle="--",  # Border style
            facecolor="none",  # Transparent fill
        )
        ax.add_patch(rect)  # Add the rectangle to the plot
        pass
    return ad_sub


def find_clones(adata, clone_key="clone", excluded_genes=None, fillna=None, layer=None):
    av_genes, bv_genes, dv_genes, gv_genes, tv_genes = get_tcr_genes(adata)

    if excluded_genes is not None:
        av_genes = [g for g in av_genes if g not in excluded_genes]
        bv_genes = [g for g in bv_genes if g not in excluded_genes]
        dv_genes = [g for g in dv_genes if g not in excluded_genes]
        gv_genes = [g for g in gv_genes if g not in excluded_genes]
        tv_genes = [g for g in tv_genes if g not in excluded_genes]

    # avbv_tuples = list(itertools.product(av_genes, bv_genes))

    if layer is None:
        X_av = npc.utils.to_numpy(adata[:, av_genes].X)
        X_bv = npc.utils.to_numpy(adata[:, bv_genes].X)
    else:
        X_av = npc.utils.to_numpy(adata[:, av_genes].layers[layer])
        X_bv = npc.utils.to_numpy(adata[:, bv_genes].layers[layer])

    # df_av = pd.DataFrame(X_av, columns=av_genes, index=adata.obs_names)
    # df_bv = pd.DataFrame(X_bv, columns=bv_genes, index=adata.obs_names)

    # this calculates all possible combinations of AV-BV genes efficiently
    X_avbv = (X_av[:, :, None] * X_bv[:, None, :]).reshape(X_av.shape[0], -1)

    # Create column names for each AV-BV combination
    avbv_cols = [f"{av}_{bv}" for av in av_genes for bv in bv_genes]

    # Build the DataFrame
    df_avbv = pd.DataFrame(
        X_avbv,
        index=adata.obs_names,
        columns=avbv_cols,  # same index as original
    )
    # convert to binary
    # TODO later filter based on the expression more detailed
    df_avbv = df_avbv.astype(bool).astype(int)

    summed = df_avbv.sum(axis=1)
    mask = summed == 1
    df_avbv = df_avbv[mask].copy()

    df_melted = pd.melt(
        df_avbv.reset_index(), id_vars=["index"], value_vars=df_avbv.columns
    )
    df_melted = df_melted[df_melted["value"] == 1].set_index("index")
    adata.obs[clone_key] = df_melted.loc[df_melted.index, "variable"]
    print(f"Found {adata.obs[clone_key].notna().sum()} clones")
    if fillna is not None:
        adata.obs.fillna({clone_key: fillna}, inplace=True)


def get_avbv_assignment_counts(
    adata, av_genes, bv_genes, layer="counts", sample_key=None
):
    if sample_key is None:
        # Single sample analysis
        if layer is None:
            X_av = npc.utils.to_numpy(adata[:, av_genes].X)
            X_bv = npc.utils.to_numpy(adata[:, bv_genes].X)
        else:
            X_av = npc.utils.to_numpy(adata[:, av_genes].layers[layer])
            X_bv = npc.utils.to_numpy(adata[:, bv_genes].layers[layer])

        X_av_mask = X_av > 0
        X_bv_mask = X_bv > 0

        # Check if any av or bv gene is expressed per cell
        av_expressed = X_av_mask.any(axis=1)  # shape: (n_cells,)
        bv_expressed = X_bv_mask.any(axis=1)  # shape: (n_cells,)

        # All cells expressing both av and bv genes
        both_av_bv = av_expressed & bv_expressed

        only_av = av_expressed & ~bv_expressed
        only_bv = bv_expressed & ~av_expressed

        # All cells expressing no av or bv genes
        neither_av_nor_bv = ~(av_expressed | bv_expressed)

        # Return counts as dict
        return {
            "both_av_bv": both_av_bv.sum(),
            "only_av": only_av.sum(),
            "only_bv": only_bv.sum(),
            "neither_av_nor_bv": neither_av_nor_bv.sum(),
        }
    else:
        # Multi-sample analysis
        results = []
        for sample in adata.obs[sample_key].unique():
            ad_sub = adata[adata.obs[sample_key] == sample].copy()
            counts = get_avbv_assignment_counts(
                ad_sub, av_genes, bv_genes, layer=layer, sample_key=None
            )
            counts["sample"] = sample
            results.append(counts)

        # Convert to DataFrame with samples as index
        df = pd.DataFrame(results)
        df = df.set_index("sample")
        return df


def plot_avbv_assignment_counts_stacked(
    df_counts, figsize=(8, 6), save_path=None, title=None
):
    """
    Plot normalized AV/BV assignment counts as stacked horizontal bars.

    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with samples as index and columns: 'both_av_bv', 'only_av',
        'only_bv', 'neither_av_nor_bv'. Typically output from
        get_avbv_assignment_counts() with sample_key provided.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Overall title for the figure. If None, no title is added.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Ensure we have the expected columns
    expected_cols = ["both_av_bv", "only_av", "only_bv", "neither_av_nor_bv"]
    for col in expected_cols:
        if col not in df_counts.columns:
            raise ValueError(f"Expected column '{col}' not found in df_counts")

    # Normalize each row to sum to 1
    df_plot = df_counts[expected_cols].div(df_counts[expected_cols].sum(axis=1), axis=0)

    # Define column names and corresponding labels/colors
    plot_configs = [
        ("both_av_bv", "Both AV and BV"),
        ("only_av", "Only AV"),
        ("only_bv", "Only BV"),
        ("neither_av_nor_bv", "Neither AV nor BV"),
    ]

    # Get samples
    samples = df_plot.index.tolist()
    y_pos = np.arange(len(samples))

    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=figsize)

    # Colors for each category
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_configs)))

    # Create stacked horizontal bars
    left = np.zeros(len(samples))
    bars = []
    labels = []

    for idx, (col, label) in enumerate(plot_configs):
        values = df_plot[col].values
        bar = ax.barh(
            y_pos, values, left=left, align="center", label=label, color=colors[idx]
        )
        bars.append(bar)
        labels.append(label)
        left += values

    # Set y-axis ticks and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(samples)
    ax.set_ylabel("Sample", fontsize=12)

    # Set x-axis label and limits
    ax.set_xlabel("Proportion of cells", fontsize=11)
    ax.set_xlim(0, 1)

    # Add legend
    ax.legend(loc="best", fontsize=10)

    # Add grid for better readability
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    sns.despine(top=True, right=True, ax=ax)

    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_avbv_assignment_counts(
    df_counts, figsize=(16, 6), save_path=None, title=None, normalize=False
):
    """
    Plot AV/BV assignment counts across samples.

    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with samples as index and columns: 'both_av_bv', 'only_av',
        'only_bv', 'neither_av_nor_bv'. Typically output from
        get_avbv_assignment_counts() with sample_key provided.
    figsize : tuple, optional
        Figure size (width, height). Default is (16, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Overall title for the figure. If None, no title is added.
    normalize : bool, optional
        If True, normalize each row (sample) so that it sums to 1 and plot
        as stacked bars. This converts counts to proportions. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes or np.ndarray
        The matplotlib axes object(s). Single axis if normalize=True, array if False.
    """
    # If normalize is True, create stacked bar plot with absolute counts on the right
    if normalize:
        # Ensure we have the expected columns
        expected_cols = ["both_av_bv", "only_av", "only_bv", "neither_av_nor_bv"]
        for col in expected_cols:
            if col not in df_counts.columns:
                raise ValueError(f"Expected column '{col}' not found in df_counts")

        # Normalize each row to sum to 1 for proportions
        df_plot = df_counts[expected_cols].div(
            df_counts[expected_cols].sum(axis=1), axis=0
        )

        # Calculate total counts per sample
        total_counts = df_counts[expected_cols].sum(axis=1)

        # Define column names and corresponding labels/colors
        plot_configs = [
            ("both_av_bv", "Both AV and BV"),
            ("only_av", "Only AV"),
            ("only_bv", "Only BV"),
            ("neither_av_nor_bv", "Neither AV nor BV"),
        ]

        # Get samples
        samples = df_plot.index.tolist()
        y_pos = np.arange(len(samples))

        # Create figure with 2 subplots side by side, right axes 20% width of left
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [5, 1]}
        )
        fig.subplots_adjust(wspace=0.3)

        ax_left = axes[0]
        ax_right = axes[1]

        # Colors for each category
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_configs)))

        # Left subplot: stacked horizontal bars (proportions)
        left = np.zeros(len(samples))
        for idx, (col, label) in enumerate(plot_configs):
            values = df_plot[col].values
            ax_left.barh(
                y_pos, values, left=left, align="center", label=label, color=colors[idx]
            )
            left += values

        # Set y-axis ticks and labels
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(samples, fontsize=8)
        ax_left.set_ylabel("Sample", fontsize=10)

        # Set x-axis label and limits
        ax_left.set_xlabel("Proportion of cells", fontsize=10)
        ax_left.tick_params(axis="x", labelsize=8)
        ax_left.set_xlim(0, 1)

        # Add legend
        ax_left.legend(loc="best", fontsize=10)

        # Add grid for better readability
        ax_left.grid(axis="x", alpha=0.3, linestyle="--")
        sns.despine(top=True, right=True, ax=ax_left)

        # Right subplot: absolute counts
        muted_red = "#c44e52"  # Muted red color
        ax_right.barh(y_pos, total_counts.values, align="center", color=muted_red)
        ax_right.set_yticks(y_pos)
        ax_right.set_yticklabels(samples, fontsize=8)
        ax_right.set_xlabel("Number of cells", fontsize=10)
        ax_right.set_title("Total cells", fontsize=10, fontweight="bold")
        ax_right.tick_params(axis="x", labelsize=8)
        ax_right.grid(axis="x", alpha=0.3, linestyle="--")
        sns.despine(top=True, right=True, ax=ax_right)

        # Add overall title if provided
        if title is not None:
            fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes

    # Ensure we have the expected columns
    expected_cols = ["both_av_bv", "only_av", "only_bv", "neither_av_nor_bv"]
    for col in expected_cols:
        if col not in df_counts.columns:
            raise ValueError(f"Expected column '{col}' not found in df_counts")

    df_plot = df_counts[expected_cols]
    xlabel = "Number of cells"

    # Create figure with 4 subplots side by side
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    fig.subplots_adjust(wspace=0.3)

    # Define column names and corresponding subtitles
    plot_configs = [
        ("both_av_bv", "Both AV and BV"),
        ("only_av", "Only AV"),
        ("only_bv", "Only BV"),
        ("neither_av_nor_bv", "Neither AV nor BV"),
    ]

    # Get samples in reverse order for top-to-bottom display
    samples = df_plot.index.tolist()

    # Plot each category
    for idx, (col, subtitle) in enumerate(plot_configs):
        ax = axes[idx]
        values = df_plot[col].values

        # Create horizontal bar plot
        y_pos = np.arange(len(samples))
        ax.barh(y_pos, values, align="center")

        # Set y-axis ticks and labels - show sample names on all subplots
        ax.set_yticks(y_pos)
        ax.set_yticklabels(samples)
        if idx == 0:
            # Add y-axis label only on the leftmost subplot
            ax.set_ylabel("Sample", fontsize=12)

        # Set x-axis label
        ax.set_xlabel(xlabel, fontsize=11)

        # Set subtitle
        ax.set_title(subtitle, fontsize=12, fontweight="bold")

        # Add grid for better readability
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        sns.despine(top=True, right=True, ax=ax)

    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def annotate_singlets(adata, genes, out_key, allow_multiplets=False, layer=None):
    if layer is None:
        X = npc.utils.to_numpy(adata[:, genes].X)
    else:
        X = npc.utils.to_numpy(adata[:, genes].layers[layer])

    X_mask = X > 0
    # number of genes expressed in each cell
    X_sum = X_mask.sum(axis=1)

    mask_single = X_sum == 1
    mask_multi = X_sum > 1

    print(f"Found {mask_single.sum()} singlets")
    print(f"Found {mask_multi.sum()} multiplets")

    X_mask = X_mask & mask_single[:, None]

    if allow_multiplets:
        # resolve multiple
        # For cells with multiple genes, keep only the highest expressed one
        X_multi = X[mask_multi]

        # further remove those cells where two genes have the highest expression

        # Get indices of top 2 expressed genes for each cell
        top_2_indices = np.argsort(X_multi, axis=1)[:, ::-1][:, :2]
        # Get the actual expression values for top 2 genes
        top_2_values = np.take_along_axis(X_multi, top_2_indices, axis=1)

        # Only keep cells where highest > second highest
        mask_top_2 = top_2_values[:, 0] > top_2_values[:, 1]
        mask_valid = np.zeros(X_mask.shape[0], dtype=bool)
        mask_valid[mask_multi] = mask_top_2

        print(f"Found {mask_valid.sum()} valid multiplets")

        # Create mask for valid multi-gene cells
        mask_multi_valid = np.zeros_like(X_mask)

        # For valid cells, set True for highest expressed gene
        highest_gene_indices = top_2_indices[mask_top_2, 0]

        # For valid cells, set True for highest expressed gene
        mask_multi_valid[mask_valid, highest_gene_indices] = True

        X_mask = X_mask | mask_multi_valid

    mask_none = X_mask.sum(axis=1) == 0
    adata.obs[out_key] = None
    X_mask_idx = np.argmax(X_mask[~mask_none], axis=1)
    values = np.array(genes)[X_mask_idx]
    adata.obs.loc[~mask_none, out_key] = values


def find_avbv_clones(adata, av_genes, bv_genes, allow_multiplets=True, layer=None):
    annotate_singlets(
        adata,
        genes=av_genes,
        out_key="av_clone",
        allow_multiplets=allow_multiplets,
        layer=layer,
    )
    annotate_singlets(
        adata,
        genes=bv_genes,
        out_key="bv_clone",
        allow_multiplets=allow_multiplets,
        layer=layer,
    )
    adata.obs["avbv_clone"] = (
        adata.obs["av_clone"].astype(str) + "+" + adata.obs["bv_clone"].astype(str)
    )
    # remove None values
    adata.obs.loc[adata.obs["av_clone"].isna(), "avbv_clone"] = None
    adata.obs.loc[adata.obs["bv_clone"].isna(), "avbv_clone"] = None

    print(f"Found {adata.obs['avbv_clone'].notna().sum()} avbv clones")


def find_gvdv_clones(adata, gv_genes, dv_genes, allow_multiplets=True, layer=None):
    annotate_singlets(
        adata,
        genes=gv_genes,
        out_key="gv_clone",
        allow_multiplets=allow_multiplets,
        layer=layer,
    )
    annotate_singlets(
        adata,
        genes=dv_genes,
        out_key="dv_clone",
        allow_multiplets=allow_multiplets,
        layer=layer,
    )
    adata.obs["gvdv_clone"] = (
        adata.obs["gv_clone"].astype(str) + "+" + adata.obs["dv_clone"].astype(str)
    )
    # remove None values
    adata.obs.loc[adata.obs["gv_clone"].isna(), "gvdv_clone"] = None
    adata.obs.loc[adata.obs["dv_clone"].isna(), "gvdv_clone"] = None

    print(f"Found {adata.obs['gvdv_clone'].notna().sum()} gvdv clones")


def get_avbv_expression(
    adata, av_genes=None, bv_genes=None, excluded_genes=None, layer="counts"
):
    if av_genes is None:
        av_genes = get_tcr_genes(adata)[0]
    if bv_genes is None:
        bv_genes = get_tcr_genes(adata)[1]

    if excluded_genes is not None:
        av_genes = [g for g in av_genes if g not in excluded_genes]
        bv_genes = [g for g in bv_genes if g not in excluded_genes]

    if layer is None:
        X_av = npc.utils.to_numpy(adata[:, av_genes].X)
        X_bv = npc.utils.to_numpy(adata[:, bv_genes].X)
    else:
        X_av = npc.utils.to_numpy(adata[:, av_genes].layers[layer])
        X_bv = npc.utils.to_numpy(adata[:, bv_genes].layers[layer])

    # this calculates all possible combinations of AV-BV genes efficiently
    X_avbv = (X_av[:, :, None] * X_bv[:, None, :]).reshape(X_av.shape[0], -1)

    # convert to sparse
    X_avbv = scipy.sparse.csr_matrix(X_avbv)

    # Create column names for each AV-BV combination
    avbv_cols = [f"{av}+{bv}" for av in av_genes for bv in bv_genes]

    # Build the sparse DataFrame
    df_avbv = pd.DataFrame.sparse.from_spmatrix(
        X_avbv,
        index=adata.obs_names,
        columns=avbv_cols,
    )
    return df_avbv


def remove_minority_avbv_expression(
    adata, av_genes, bv_genes, out_layer, in_layer="counts"
):
    if in_layer is None:
        X_in = adata.X.copy()
    else:
        X_in = adata.layers[in_layer].copy()

    # check if X is sparse
    if scipy.sparse.issparse(X_in):
        is_sparse = True
        X_in = X_in.toarray()
    else:
        is_sparse = False

    print(f"Sum of X_in before removal: {X_in.sum()}")

    for gene_list in [av_genes, bv_genes]:
        gene_idx = np.where(adata.var_names.isin(gene_list))[0]
        X = X_in[:, gene_idx]

        X_mask = X > 0
        X_sum = X_mask.sum(axis=1)
        mask_multi = X_sum > 1
        multi_indices = np.where(mask_multi)[0]

        for i in multi_indices:
            # Get the row we want to modify
            row = X[i]
            # Find max expression value
            max_expr = np.max(row)
            # Create mask for non-maximum values
            mask = row < max_expr

            # Set non-maximum values to 0 directly in X_in
            X_in[i, gene_idx[mask]] = 0

        # # Get all rows we want to modify at once
        # rows = X[multi_indices]
        # # Find max expression values for each row
        # max_expr = rows.max(axis=1, keepdims=True)
        # # Create mask for non-maximum values
        # mask = rows < max_expr

        # # Set non-maximum values to 0 directly in X_in
        # X_in[multi_indices, gene_idx[mask]] = 0

    print(f"Sum of X_in after removal: {X_in.sum()}")
    if is_sparse:
        X_out = scipy.sparse.csr_matrix(X_in)
    else:
        X_out = X_in

    adata.layers[out_layer] = X_out


# allows overlapping gene expression
def identify_clonal_clusters(
    adata,
    av_genes,
    bv_genes,
    ct_key=None,
    sample_key=None,
    spatial_split_key=None,
    tcell_keys=None,
    max_dist=100,
    min_cells=2,
    verbose=True,
    layer=None,
):
    if sample_key is not None and spatial_split_key is None:
        # make sure coordinates are split by sample
        spatial_sample_split(adata, sample_key, displacement=1000)
        spatial_split_key = "spatial_split"
    else:
        spatial_split_key = "spatial"

    # first restrict all calculations to relevant cells
    if tcell_keys is not None:
        ad_sub = adata[adata.obs[ct_key].isin(tcell_keys)].copy()
    else:
        ad_sub = adata

    if verbose:
        print(f"Considering {ad_sub.shape[0]} cells for clonal expansion analysis")

    if layer is None:
        mask_av = np.array(ad_sub[:, av_genes].X.sum(axis=1) > 0).flatten()
        mask_bv = np.array(ad_sub[:, bv_genes].X.sum(axis=1) > 0).flatten()
    else:
        mask_av = np.array(ad_sub[:, av_genes].layers[layer].sum(axis=1) > 0).flatten()
        mask_bv = np.array(ad_sub[:, bv_genes].layers[layer].sum(axis=1) > 0).flatten()
    mask_avbv = mask_av & mask_bv

    ad_sub = ad_sub[mask_avbv].copy()

    if verbose:
        print(f"Found {ad_sub.shape[0]} cells expressing at least one AV and BV gene")

    df_avbv = get_avbv_expression(ad_sub, av_genes, bv_genes)
    col_mask = df_avbv.sum(axis=0) > 0
    df_avbv = df_avbv.loc[:, col_mask]

    if verbose:
        print(f"Found {df_avbv.shape[1]} distinct AVBV combinations")

    # convert to longformat, keeping original index
    df_avbv = pd.melt(
        df_avbv.reset_index(),
        id_vars=["index"],
        var_name="avbv",
        value_name="expression",
    )

    # remove all zero rows
    df_avbv = df_avbv[df_avbv["expression"] > 0]
    df_avbv.index = df_avbv.index.astype(str)

    ad_tmp = sc.AnnData(
        obs=df_avbv,
        obsm={"spatial": ad_sub[df_avbv["index"].values].obsm[spatial_split_key]},
    )
    ad_tmp.obs = ad_tmp.obs.join(adata.obs[["cc"]], on="index")

    spatial_sample_split(ad_tmp, "avbv", displacement=1000)

    coords = ad_tmp.obsm["spatial_split"]
    dbs = DBSCAN(eps=max_dist, min_samples=min_cells)
    # dbs = HDBSCAN(min_cluster_size=3)
    dbs.fit(coords)
    labels = [f"{label}" if label != -1 else None for label in dbs.labels_]

    ad_tmp.obs["avbv_label"] = labels

    if verbose:
        print(f"Found {ad_tmp.obs['avbv_label'].nunique()} clonal clusters")
        print(
            f"Found {ad_tmp.obs['avbv_label'].notna().sum()} cells in clonal clusters"
        )

    clone_df = pd.pivot(ad_tmp.obs, index="index", columns="avbv", values="avbv_label")
    return clone_df


def identify_clonal_clusters_single_chain(
    adata,
    trv_genes,
    ct_key=None,
    sample_key=None,
    spatial_split_key=None,
    tcell_keys=None,
    max_dist=100,
    min_cells=2,
    verbose=True,
    layer=None,
    clonal_cluster_key="trv_cluster",
):
    if sample_key is not None and spatial_split_key is None:
        # make sure coordinates are split by sample
        spatial_sample_split(adata, sample_key, displacement=1000)
        spatial_split_key = "spatial_split"

    # first restrict all calculations to relevant cells
    if tcell_keys is not None:
        ad_sub = adata[adata.obs[ct_key].isin(tcell_keys)].copy()
    else:
        ad_sub = adata

    if layer is None:
        mask_trv = np.array(ad_sub[:, trv_genes].X.sum(axis=1) > 0).flatten()
    else:
        mask_trv = np.array(
            ad_sub[:, trv_genes].layers[layer].sum(axis=1) > 0
        ).flatten()

    ad_sub = ad_sub[mask_trv].copy()

    if verbose:
        print(f"Found {ad_sub.shape[0]} cells expressing at least one TRV gene")

    # add column to identify different TRV genes
    X_trv = pd.DataFrame(
        npc.utils.to_numpy(ad_sub[:, trv_genes].X),
        index=ad_sub.obs_names,
        columns=trv_genes,
    )
    ad_sub.obs["trv_gene"] = X_trv.idxmax(axis=1)

    if verbose:
        print(f"Found {ad_sub.obs['trv_gene'].nunique()} unique TRV genes")

    spatial_sample_split(ad_sub, "trv_gene", displacement=1000)

    coords = ad_sub.obsm["spatial_split"]
    dbs = DBSCAN(eps=max_dist, min_samples=min_cells)
    # dbs = HDBSCAN(min_cluster_size=3)
    dbs.fit(coords)
    labels = [f"{label}" if label != -1 else None for label in dbs.labels_]

    ad_sub.obs[clonal_cluster_key] = labels
    ad_sub = ad_sub[ad_sub.obs[clonal_cluster_key].notna()].copy()
    ad_sub.obs[clonal_cluster_key] = (
        ad_sub.obs["trv_gene"].astype(str)
        + "_C"
        + ad_sub.obs[clonal_cluster_key].astype(str)
    )

    if verbose:
        print(f"Found {ad_sub.obs[clonal_cluster_key].nunique()} clonal clusters")
        print(
            f"Found {ad_sub.obs[clonal_cluster_key].notna().sum()} cells in clonal clusters"
        )

    adata.obs[clonal_cluster_key] = None
    adata.obs.loc[ad_sub.obs_names, clonal_cluster_key] = ad_sub.obs[clonal_cluster_key]

    filter_clonal_clusters_by_cell_type(
        adata,
        ct_key=ct_key,
        prohibited_combinations=[("CD4+", "CD8+")],
        in_key=clonal_cluster_key,
        out_key=f"{clonal_cluster_key}_filtered",
        verbose=verbose,
        prog_bar=verbose,
    )


def merge_clonal_clusters(clone_df, verbose=True):
    tmp = clone_df.copy()
    # merge into unique label per cell
    # go through each row and put the clusters into one column
    avbv_cluster = []

    while len(avbv_cluster) < tmp.shape[0]:
        # run until we have to modify the clone_df
        for k, row in tqdm(tmp.iloc[len(avbv_cluster) :].iterrows(), disable=True):
            if row.notna().sum() == 0:
                avbv_cluster.append(None)
                continue
            elif row.notna().sum() == 1:
                mask = row.notna()
                row = row[mask]
                fill_value = f"{row.index[0]}_C{row.values[0]}"
                avbv_cluster.append(fill_value)
                # break
            else:
                mask = row.notna()
                row = row[mask]
                c_sizes = [tmp[c].value_counts()[v] for c, v in row.items()]
                max_idx = np.argmax(c_sizes)
                fill_value = f"{row.index[max_idx]}_C{row.values[max_idx]}"
                avbv_cluster.append(fill_value)

                # remove non max idx elements from clone df
                for c, _v in row.items():
                    if c != row.index[max_idx]:
                        tmp.loc[k, c] = None
                # print(row)
                # print(f"sizes: {c_sizes}")
                # print(f"Filled {fill_value}")

                break
    avbv_cluster = pd.Series(avbv_cluster, index=clone_df.index)
    # remove entries with value count 1
    mask = avbv_cluster.value_counts() > 1
    entries = avbv_cluster.value_counts()[mask].index.tolist()
    avbv_cluster[~avbv_cluster.isin(entries)] = None

    if verbose:
        print(
            f"Total number of clonal clusters after merging: {avbv_cluster.nunique()}"
        )

    return avbv_cluster


def filter_clonal_clusters_by_cell_type(
    adata,
    ct_key,
    prohibited_combinations,
    in_key,
    out_key,
    verbose: int = 0,
    prog_bar: bool = True,
):
    adata.obs[out_key] = adata.obs[in_key].copy()

    for cc in tqdm(
        adata.obs[in_key].dropna().unique(), disable=not prog_bar, desc="Filtering"
    ):
        cc_cells = adata[adata.obs[in_key] == cc]
        cc_ct = cc_cells.obs[ct_key].value_counts()
        for pair in prohibited_combinations:
            if set(pair).issubset(set(cc_ct.index)):
                # Find the cell type with fewer cells in the pair and remove those cells from the cluster
                ct_counts = cc_ct[list(pair)]
                ct_to_remove = ct_counts.idxmin()
                cells_to_remove = cc_cells.obs[
                    cc_cells.obs[ct_key] == ct_to_remove
                ].index
                # Remove these cells from the cluster assignment
                adata.obs.loc[cells_to_remove, out_key] = None
                # Check if the remaining number of cells in the cluster is greater than 2
                remaining_cells = adata.obs[adata.obs[out_key] == cc]
                if remaining_cells.shape[0] < 2:
                    # If not, remove the cluster assignment for all remaining cells
                    adata.obs.loc[remaining_cells.index, out_key] = None
                    if verbose > 1:
                        print(
                            f"Removed cluster {cc} because it had less than 2 cells after filtering for {pair}"
                        )
    if verbose > 0:
        print(
            f"Total number of clonal clusters after filtering: {adata.obs[out_key].nunique()}"
        )


def find_clonal_clusters(adata, clone_key, out_key=None):
    out_key = f"{clone_key}_cluster" if out_key is None else out_key

    adata.obs[out_key] = None

    for clone in tqdm(adata.obs[clone_key].unique()):
        if clone == "NA" or pd.isna(clone):
            continue
        # print(f"Processing clone {clone}")
        mask = adata.obs[clone_key] == clone
        coords = adata[mask].obsm["spatial_split"]
        dbs = DBSCAN(eps=50, min_samples=3)
        # dbs = HDBSCAN(min_cluster_size=3)
        dbs.fit(coords)
        labels = [f"{clone}-{label}" if label != -1 else "NA" for label in dbs.labels_]
        adata.obs.loc[mask, out_key] = labels


def plot_topn_clonal_clusters(adata, clone_key, n=3, **kwargs):
    for s in adata.obs["sample"].unique()[0::]:
        ad_sub = adata[adata.obs["sample"] == s].copy()
        topn_clones = ad_sub.obs[clone_key].value_counts().index.tolist()
        topn_clones = [k for k in topn_clones if k != "NA" and not pd.isna(k)][:n]
        if len(topn_clones) < 1:
            continue
        ad_sub.obs["tmp"] = ad_sub.obs[clone_key]
        ad_sub.obs.loc[~ad_sub.obs["tmp"].isin(topn_clones), "tmp"] = None
        ax = sc.pl.spatial(
            ad_sub,
            color=["tmp"],
            show=False,
            spot_size=10,
            na_color="#eef4f0",
            **kwargs,
        )[0]
        ax.set_title(f"Top {n} clonal clusters")
        break
    ad_sub.obs[clone_key].value_counts()


def identify_clonal_clusters_from_unique_clones(
    adata,
    av_genes=None,
    bv_genes=None,
    ct_key="cell_type_no_tcr",
    sample_key="cc",
    spatial_split_key=None,
    tcell_keys=None,
    max_dist=100,
    min_cells=2,
):
    if tcell_keys is None:
        tcell_keys = ["T"]
    if av_genes is not None:
        genes = av_genes
        save_key = "av"
    elif bv_genes is not None:
        genes = bv_genes
        save_key = "bv"
    else:
        raise ValueError("Either av_genes or bv_genes must be provided")

    if sample_key is not None and spatial_split_key is None:
        # make sure coordinates are split by sample
        spatial_sample_split(adata, sample_key, displacement=1000)
        spatial_split_key = "spatial_split"
    else:
        spatial_split_key = "spatial"

    # first restrict all calculations to relevant cells
    if tcell_keys is not None:
        ad_sub = adata[adata.obs[ct_key].isin(tcell_keys)].copy()
    else:
        ad_sub = adata

    mask_genes = np.array(ad_sub[:, genes].X.sum(axis=1) > 0).flatten()
    ad_sub = ad_sub[mask_genes].copy()

    print(
        f"Found {ad_sub.shape[0]} cells expressing at least one of the {save_key} genes"
    )

    df_genes = pd.DataFrame(
        npc.utils.to_numpy(ad_sub[:, genes].X),
        index=ad_sub.obs_names,
        columns=genes,
    )
    # convert to longformat
    df_genes = pd.melt(
        df_genes.reset_index(),
        id_vars=["index"],
        value_vars=df_genes.columns,
        var_name=f"{save_key}_gene",
    )
    df_genes = df_genes[df_genes["value"] > 0]
    df_genes.index = df_genes.index.astype(str)

    ad_tmp = sc.AnnData(obs=df_genes, obsm={"spatial": ad_sub.obsm[spatial_split_key]})
    ad_tmp.obs = ad_tmp.obs.join(adata.obs[["cc"]], on="index")

    spatial_sample_split(ad_tmp, f"{save_key}_gene", displacement=1000)
