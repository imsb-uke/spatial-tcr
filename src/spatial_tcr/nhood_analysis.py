import nichepca as npc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import statsmodels.api as sm
from anndata import AnnData
from scipy.spatial import KDTree
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from tqdm.auto import tqdm

from spatial_tcr.spatial import spatial_sample_split
from spatial_tcr.stats import count_test


def compute_nhood_expression_old(
    adata,
    target_ct,
    ct_key,
    max_dist=50,
    sample_key=None,
    spatial_key="spatial",
    layer="counts",
    verbose=True,
):
    if sample_key is not None:
        spatial_sample_split(adata, sample_key=sample_key, in_key=spatial_key)

    # create the intial data
    ad_tmp = sc.AnnData(adata.X.copy() if layer is None else adata.layers[layer].copy())
    sc.pp.normalize_total(ad_tmp)

    coords = (
        adata.obsm["spatial_split"]
        if sample_key is not None
        else adata.obsm[spatial_key]
    )
    kdtree = KDTree(coords)

    # consider a specific query set
    sub_mask = adata.obs[ct_key] == target_ct
    sub_coords = coords[sub_mask]
    sub_obs_names = adata.obs_names[sub_mask]
    sub_indices = np.where(sub_mask)[0]

    neighbor_indices = kdtree.query_ball_point(sub_coords, r=max_dist)
    # remove self-neighbors
    for i, neighbors in enumerate(neighbor_indices):
        neighbor_indices[i] = np.setdiff1d(neighbors, [sub_indices[i]])

    rows, cols, data = [], [], []
    for i, neighbors in enumerate(neighbor_indices):
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
        data.extend([1] * len(neighbors))

    # Create sparse matrix with shape: (# query cells) x (total cells)
    agg_matrix = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(neighbor_indices), ad_tmp.shape[0])
    )

    n_neighbors = np.array([len(neighbors) for neighbors in neighbor_indices])

    if verbose:
        print(f"Max number of neighbors: {max(n_neighbors)}")
        print(f"Min number of neighbors: {min(n_neighbors)}")
        print(f"Median number of neighbors: {np.median(n_neighbors)}")

    # aggregate expression
    X_sum = agg_matrix.dot(ad_tmp.X).toarray()

    # normalize by number of neighbors
    X_mean = X_sum / n_neighbors[:, None]

    X_sum_log1p = np.log1p(X_sum)
    X_mean_log1p = np.log1p(X_mean)

    df_sum = pd.DataFrame(X_sum, index=sub_obs_names, columns=adata.var_names)
    df_mean = pd.DataFrame(X_mean, index=sub_obs_names, columns=adata.var_names)
    df_sum_log1p = pd.DataFrame(
        X_sum_log1p, index=sub_obs_names, columns=adata.var_names
    )
    df_mean_log1p = pd.DataFrame(
        X_mean_log1p, index=sub_obs_names, columns=adata.var_names
    )

    return df_sum, df_mean, df_sum_log1p, df_mean_log1p


def fit_GLM(adata, input, gene, layer="counts", maxiter=50):
    X = input.values
    if layer is not None:
        y = npc.utils.to_numpy(adata[input.index, gene].layers[layer]).flatten()
    else:
        y = npc.utils.to_numpy(adata[input.index, gene].X).flatten()

    # Optionally, standardize predictors to improve numerical stability.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add an intercept term to the predictors
    X_count = sm.add_constant(X_scaled, has_constant="add")  # (2167, 481)
    X_infl = X_scaled  # (2167, 480)

    # Fit a simpler Poisson model to get starting values for the count component
    poisson_model = sm.GLM(y, X_count, family=sm.families.Poisson())
    poisson_results = poisson_model.fit()
    start_params_count = poisson_results.params

    model = ZeroInflatedNegativeBinomialP(
        y, X_count, exog_infl=X_infl, inflation="logit"
    )

    # For the inflation part, initialize to zeros with length matching model.exog_infl.
    start_params_infl = np.zeros(model.exog_infl.shape[1])
    start_params_disp = 0.1 * np.ones((1,))
    start_params = np.concatenate(
        [start_params_count, start_params_infl, start_params_disp]
    )

    result = model.fit(start_params=start_params, maxiter=maxiter, disp=1)

    return result


def fit_lasso(adata, input, gene, layer=None):
    X = input.values
    if layer is not None:
        y = npc.utils.to_numpy(adata[input.index, gene].layers[layer]).flatten()
    else:
        y = npc.utils.to_numpy(adata[input.index, gene].X).flatten()

    # Scale the features to have zero mean and unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use LASSO with cross-validation to select the most influential genes
    lasso = LassoCV(cv=5, max_iter=100000, n_jobs=-1).fit(X_scaled, y)

    # Get the coefficients for each gene
    coefficients = lasso.coef_

    # Create a DataFrame with gene names and coefficients
    coef_df = pd.DataFrame(
        {"gene": input.columns, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)

    return coef_df


def fit_ridge(adata, input, gene, layer=None):
    X = input.values
    if layer is not None:
        y = npc.utils.to_numpy(adata[input.index, gene].layers[layer]).flatten()
    else:
        y = npc.utils.to_numpy(adata[input.index, gene].X).flatten()

    # Scale the features to have zero mean and unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use Ridge with cross-validation to select the most influential genes
    ridge = RidgeCV(cv=5).fit(X_scaled, y)

    # Get the coefficients for each gene
    coefficients = ridge.coef_

    # Create a DataFrame with gene names and coefficients
    coef_df = pd.DataFrame(
        {"gene": input.columns, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)

    # Print top positive and negative genes
    # n_top = 10  # adjust this number to show more or fewer genes
    # print(f"\nTop {n_top} positive genes:")
    # print(coef_df.head(n_top))
    # print(f"\nTop {n_top} negative genes:")
    # print(coef_df.tail(n_top))

    return coef_df


def correlation_analysis(adata):
    raise NotImplementedError("Not implemented")


def find_neighbors(
    adata,
    obs_key,
    radius=25,
    sample_key=None,
    spatial_key="spatial",
    plot_nhoods=False,
):
    obs_vals = adata.obs[obs_key].dropna().unique()
    samples = adata.obs[sample_key].unique()

    neighbors_per_obs_val = {o: [] for o in obs_vals}

    for sample in tqdm(samples):
        sample_mask = adata.obs[sample_key] == sample
        ad_sample = adata[sample_mask]
        obs_vals = ad_sample.obs[obs_key].dropna().unique()
        for obs_val in obs_vals:
            mask = ad_sample.obs[obs_key] == obs_val

            coords = ad_sample.obsm[spatial_key]
            target_coords = coords[mask]

            # find the neighbors
            tree = KDTree(coords)
            nhoods = tree.query_ball_point(target_coords, r=radius)

            # flatten the list and make unique
            nhoods = [item for sublist in nhoods for item in sublist]
            nhoods = np.unique(nhoods)

            # remove the current target cells
            nhoods = [i for i in nhoods if i not in np.where(mask)[0]]

            # convert to cell ids
            nhoods = ad_sample.obs_names[nhoods]
            neighbors_per_obs_val[obs_val].extend(nhoods)

            if plot_nhoods:
                ad_sample.obs["in_nhood"] = None
                ad_sample.obs.loc[nhoods, "in_nhood"] = "True"
                sc.pl.spatial(
                    ad_sample,
                    color="in_nhood",
                    title=f"{obs_val} in {sample}",
                    show=True,
                    spot_size=10,
                )

    return neighbors_per_obs_val


def compute_nhood_composition(
    adata: AnnData,
    obs_key: str,
    comp_key: str,
    sample_key: str | None = None,
    spatial_key: str = "spatial",
    normalize: bool = False,
):
    assert "graph" in adata.uns.keys(), "Graph must be provided"

    obs_onehot = pd.get_dummies(adata.obs[comp_key]).astype(int)
    ad_tmp = sc.AnnData(
        X=obs_onehot.values,
        var=pd.DataFrame(index=obs_onehot.columns),
        obs=adata.obs[
            [obs_key] + [sample_key] if sample_key is not None else [obs_key]
        ],
        obsm={spatial_key: adata.obsm[spatial_key]},
        uns={"graph": adata.uns["graph"]},
    )

    npc.ne.aggregate(ad_tmp, backend="sparse", aggr="sum")

    ad_tmp = ad_tmp[ad_tmp.obs[obs_key].notna()].copy()

    # remove any nhoods with no cells
    sc.pp.filter_cells(ad_tmp, min_counts=1)
    # drop n_counts col
    ad_tmp.obs.drop(columns=["n_counts"], inplace=True)

    ad_tmp.obsm["ct_counts"] = pd.DataFrame(
        ad_tmp.X, index=ad_tmp.obs_names, columns=ad_tmp.var_names
    )
    ad_tmp.obs[ad_tmp.var_names] = ad_tmp.X

    if normalize:
        sc.pp.normalize_total(ad_tmp, target_sum=1.0)

    return ad_tmp


def compute_nhood_expression(
    adata,
    obs_key,
    radius=25,
    sample_key=None,
    spatial_key="spatial",
    layer="counts",
    construct_graph=True,
):
    if construct_graph:
        assert "graph" in adata.uns.keys(), "Graph must be provided"

    ad_tmp = sc.AnnData(
        adata.X.copy() if layer is None else adata.layers[layer].copy(),
        var=pd.DataFrame(index=adata.var_names),
        obs=adata.obs[
            [obs_key] + [sample_key] if sample_key is not None else [obs_key]
        ],
        obsm={spatial_key: adata.obsm[spatial_key]},
        uns={"graph": adata.uns["graph"]} if not construct_graph else None,
    )
    # construct the neighborhood graph
    if construct_graph:
        if sample_key is not None:
            npc.gc.construct_multi_sample_graph(
                ad_tmp,
                sample_key=sample_key,
                obsm_key=spatial_key,
                radius=radius,
                remove_self_loops=True,
                verbose=False,
            )
        else:
            npc.gc.distance_graph(
                ad_tmp,
                obsm_key=spatial_key,
                radius=radius,
                remove_self_loops=True,
                verbose=False,
            )

    # aggregate the expression
    npc.ne.aggregate(ad_tmp, backend="sparse", aggr="sum")

    ad_tmp = ad_tmp[ad_tmp.obs[obs_key].notna()].copy()

    # remove any nhhods with no expression
    sc.pp.filter_cells(ad_tmp, min_counts=1)

    # if layer is not None:
    #     ad_tmp.layers[layer] = ad_tmp.X.copy()

    return ad_tmp


### Plotting functions


def compare_nhood_composition(
    comp_per_group_norm,
    title="T cell neighborhood composition",
    topn=10,
    order_by_ratio=True,
    show_other=True,
    palette=None,
    save_path=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")

    if order_by_ratio:
        ratios = comp_per_group_norm.iloc[0] / comp_per_group_norm.iloc[1]
        ratios = ratios.sort_values(ascending=True)
        comp_per_group_norm = comp_per_group_norm[ratios.index]

    agg_comp_per_group_norm = (
        comp_per_group_norm[comp_per_group_norm.columns[-topn::]].copy() * 100
    )
    if show_other:
        agg_comp_per_group_norm["other"] = 100 - agg_comp_per_group_norm.sum(axis=1)
        # move other columns to the front
        agg_comp_per_group_norm = agg_comp_per_group_norm[
            ["other"] + list(agg_comp_per_group_norm.columns[:-1])
        ]

    celltypes = agg_comp_per_group_norm.columns
    if palette is None:
        palette = sns.color_palette("tab10", len(celltypes))
        palette = {ct: palette[i] for i, ct in enumerate(celltypes[::-1])}
        palette["other"] = "lightgray"

    fig, ax = plt.subplots(figsize=(1.5, 5))
    agg_comp_per_group_norm.iloc[::-1].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[palette[ct] for ct in agg_comp_per_group_norm.columns],
        linewidth=0,
    )

    ax.set_title(title)
    ax.set_ylabel("Proportion [%]")
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()

    if show_other:
        ax.legend(
            handles[::-1], labels[::-1], bbox_to_anchor=(1, 1), frameon=False, ncol=1
        )
    else:
        ax.legend(
            handles[::-1], labels[::-1], bbox_to_anchor=(1.6, 1), frameon=False, ncol=1
        )
    sns.despine(ax=ax, right=True, top=True)

    # set x-axis labels to ANCA and Control
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_specific_nhood_proportions(
    counts_df,
    value,
    p_value: float = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = sns.color_palette("Greys", n_colors=2)[::-1]

    comp_df = counts_df.div(counts_df.sum(axis=1), axis=0)
    # ensure index has no name
    comp_df.index.name = None
    comp_df_melted = pd.melt(
        comp_df.reset_index(),
        id_vars="index",
        var_name="tmp",
        value_name="prop",
    )

    plot_df = comp_df_melted[comp_df_melted["tmp"] == value]

    plot_df.loc[:, "prop"] = plot_df["prop"] * 100

    fig, ax = plt.subplots(figsize=(1.5, 5))
    sns.barplot(
        x="index",
        y="prop",
        hue="index",
        data=plot_df,
        ax=ax,
        palette=colors,
        width=0.6,
    )
    ax.set_title(f"{value}")
    ax.set_xlabel("")
    ax.set_ylabel("Proportion [%]")

    # Add significance bar and star
    # Convert p-value to star notation
    # extract the relevant counts
    tot_counts = counts_df.sum(axis=1)
    val_counts = counts_df[value]
    other_counts = tot_counts - val_counts

    p_value = (
        count_test(
            val_counts.values[0],
            other_counts.values[0],
            val_counts.values[1],
            other_counts.values[1],
        )
        if p_value is None
        else p_value
    )

    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"
    # Adjust the y-offset based on your data range:
    y_max = np.max(plot_df["prop"].values)
    y = y_max + 0.05 * (y_max)  # starting height of the line
    h = 0.02 * (y_max)  # height of the significance bar

    x1, x2 = 0, 1
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
    ax.text((x1 + x2) * 0.5, y + h, stars, ha="center", va="bottom", color="k")

    # Add a bit more space at the top of the plot for significance annotation
    ax.set_ylim(top=y + h + 0.05 * y_max, bottom=0)

    sns.despine()
    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join(figure_dir, f"{ct}_tls_vs_rest.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    plt.show()


def run_nhood_comparison(
    adata,
    group_1,
    group_2,
    group_key,
    comp_key: str = None,
    sample_key: str = None,
    spatial_key: str = "spatial",
    radius: float = 25,
    group_1_name: str = None,
    group_2_name: str = None,
    construct_graph: bool = True,
    layer: str = "counts",
    **kwargs,
):
    import nichepca as npc
    from statsmodels.stats.multitest import multipletests

    from spatial_tcr.stats import wilcoxon_significance

    if construct_graph:
        if sample_key is not None:
            npc.gc.construct_multi_sample_graph(
                adata,
                sample_key=sample_key,
                obsm_key=spatial_key,
                radius=radius,
                remove_self_loops=True,
                verbose=False,
            )
        else:
            npc.gc.distance_graph(
                adata,
                obsm_key=spatial_key,
                radius=radius,
                remove_self_loops=True,
                verbose=False,
            )

    if comp_key is not None:
        print("Running neighborhood composition analysis ...")

        ad_nhood = compute_nhood_composition(
            adata,
            group_key,
            comp_key,
            # sample_key=sample_key,
            # spatial_key=spatial_key,
            **kwargs,
        )

        ad_nhood_merged = ad_nhood[
            ad_nhood.obs[group_key].isin([group_1, group_2])
        ].copy()

        # normalize the composition
        ct_comp = ad_nhood_merged.obs
        ct_cols = [c for c in ct_comp.columns if c != group_key]
        ct_comp_norm = ct_comp[ct_cols].div(ct_comp[ct_cols].sum(axis=1), axis=0)
        ct_comp_norm[group_key] = ct_comp[group_key]

        # perform the test
        p_values = {
            ct: wilcoxon_significance(ct_comp_norm, ct, group_key=group_key)
            for ct in ct_cols
        }

        # perform multiple testing correction
        p_values_corrected = multipletests(list(p_values.values()), method="fdr_bh")[1]
        p_values = dict(zip(p_values.keys(), p_values_corrected))

        # compute the average composition
        mean_comp = ct_comp_norm.groupby(group_key, observed=True).mean()
        ratios = mean_comp.iloc[0] / mean_comp.iloc[1]
        ratios = ratios.sort_values(ascending=True)
        mean_comp = mean_comp[ratios.index]

        # define new names for the groups if desired
        group_1_name = f"{group_1}\nhood" if group_1_name is None else group_1_name
        group_2_name = f"{group_2}\nhood" if group_2_name is None else group_2_name
        mean_comp.index = mean_comp.index.map(
            {group_1: group_1_name, group_2: group_2_name}
        )
        mean_comp.index.name = None

    print("Running neighborhood expression analysis ...")
    ad_nhood_expr = compute_nhood_expression(
        adata,
        group_key,
        construct_graph=False,
        layer=layer,
    )
    ad_nhood_expr_merged = ad_nhood_expr[
        ad_nhood_expr.obs[group_key].isin([group_1, group_2])
    ].copy()

    return ad_nhood_expr_merged, mean_comp, p_values
