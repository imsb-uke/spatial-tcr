import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


def _p_to_stars(p: float) -> str:
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "n.s."


def per_category_fisher_subset(
    counts_kx2: pd.DataFrame,
    col1: str,
    col2: str,
    adjust: str = "fdr_bh",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    pop = counts_kx2[col1].to_numpy(dtype=int)
    sel = counts_kx2[col2].to_numpy(dtype=int)

    if np.any(pop < 0) or np.any(sel < 0):
        raise ValueError("Counts must be non-negative.")
    if np.any(sel > pop):
        raise ValueError("Subset design requires col2_i <= col1_i for all categories.")

    N = int(pop.sum())
    M = int(sel.sum())
    if N <= 0:
        raise ValueError("Total of col1 must be > 0.")
    if M < 0 or M > N:
        raise ValueError("Total of col2 must satisfy 0 <= sum(col2) <= sum(col1).")

    pvals: list[float] = []
    odds: list[float] = []

    for Ni, ki in zip(pop, sel):
        table = np.array(
            [
                [ki, M - ki],
                [Ni - ki, (N - Ni) - (M - ki)],
            ],
            dtype=int,
        )

        # Guard against degenerate all-zero table (rare, but possible if N==0 handled above)
        if table.sum() == 0:
            pvals.append(1.0)
            odds.append(np.nan)
            continue

        oddsratio, pvalue = fisher_exact(table, alternative=alternative)
        pvals.append(float(pvalue))
        odds.append(float(oddsratio))

    reject, p_adj, _, _ = multipletests(pvals, method=adjust)

    return pd.DataFrame(
        {"oddsratio": odds, "p": pvals, "p_adj": p_adj, "reject": reject},
        index=counts_kx2.index,
    )


def annotate_pairwise_stars(
    ax: plt.Axes,
    order: list,
    stars_by_category: pd.Series,  # index=category -> string
    y_offset_frac: float = 0.02,
    fontsize: int = 9,
    line_width: float = 1.0,
):
    """Adds a bracket + stars above each pair of bars (assumes exactly 2 hue levels)."""
    containers = list(ax.containers)
    if len(containers) < 2:
        return

    bars0 = list(containers[0])
    bars1 = list(containers[1])
    if len(bars0) != len(order) or len(bars1) != len(order):
        return

    ymax = max([b.get_height() for b in (bars0 + bars1)] + [0.0])
    y_offset = max(1e-12, ymax * y_offset_frac)
    h = y_offset * 0.6

    for i, cat in enumerate(order):
        stars = str(stars_by_category.loc[cat])

        b0, b1 = bars0[i], bars1[i]
        x0 = b0.get_x() + b0.get_width() / 2
        x1 = b1.get_x() + b1.get_width() / 2
        y = max(b0.get_height(), b1.get_height()) + y_offset

        ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=line_width, c="black")
        ax.text(
            (x0 + x1) / 2, y + h, stars, ha="center", va="bottom", fontsize=fontsize
        )


def plot_side_by_side_counts(
    df: pd.DataFrame,
    col1: str | None = None,
    col2: str | None = None,
    category_name: str = "category",
    value_name: str = "value",
    ax: plt.Axes | None = None,
    rotate_xticks: int = 90,
    title: str | None = None,
    normalize: bool = True,
    annotate_per_category: bool = True,
    adjust: str = "fdr_bh",
    alternative: str = "two-sided",
    return_stats: bool = False,
    figsize=None,
    save_path: str | None = None,
    ylabel: str | None = None,
):
    if col1 is None or col2 is None:
        col1, col2 = df.columns[:2].tolist()

    counts = df[[col1, col2]].copy()

    # Drop categories that are zero in both settings (no information)
    counts = counts.loc[~((counts[col1] == 0) & (counts[col2] == 0))]

    # Values for plotting
    plot_df = counts.div(counts.sum(axis=0), axis=1) if normalize else counts
    order = plot_df.sort_values(col1, ascending=False).index.tolist()

    long_df = plot_df.reset_index(names=category_name).melt(
        id_vars=category_name, var_name="setting", value_name=value_name
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(8, 0.5 * len(order)), 4) if figsize is None else figsize,
        )

    sns.barplot(
        data=long_df,
        x=category_name,
        y=value_name,
        hue="setting",
        hue_order=[col1, col2],
        order=order,
        dodge=True,
        ax=ax,
    )

    # Reduce gaps at the beginning and end by setting tight x-axis limits
    ax.margins(x=0)
    n_categories = len(order)
    ax.set_xlim(-0.5, n_categories - 0.5)
    ax.set_ylim(0, ax.get_ylim()[1] + 0.01)

    ax.set_xlabel("")
    ax.set_ylabel(("Fraction" if normalize else "Count") if ylabel is None else ylabel)
    ax.legend(title="", frameon=False)
    ax.tick_params(axis="x", rotation=rotate_xticks)

    per_cat = None
    if annotate_per_category:
        per_cat = per_category_fisher_subset(
            counts_kx2=counts,
            col1=col1,
            col2=col2,
            adjust=adjust,
            alternative=alternative,
        )
        stars = per_cat["p_adj"].apply(_p_to_stars)
        annotate_pairwise_stars(ax, order=order, stars_by_category=stars)

    sns.despine(top=True, right=True, ax=ax)

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*")
        plt.tight_layout()
    if return_stats:
        return ax, per_cat
    return ax


def _plot_stacked_bars(
    df_plot: pd.DataFrame,
    figsize: tuple,
    xlabel: str,
    ylabel: str,
    colors_dict: dict[str, str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    ylim_max: float | None = None,
    legend_bbox_to_anchor: tuple = (1.0, 0.5),
    legend_ncol: int = 1,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Helper function to create stacked vertical bar plots.

    Parameters
    ----------
    df_plot : pd.DataFrame
        DataFrame with categories as index and items to stack as columns.
    figsize : tuple
        Figure size (width, height).
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    colors_dict : dict[str, str], optional
        Dictionary mapping column names to hex color codes. If None, uses
        colorcet glasbey colormap for distinct colors.
    title : str, optional
        Overall title for the figure. If None, no title is added.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    ylim_max : float, optional
        Maximum value for y-axis limit. If None, uses max of data.
    legend_bbox_to_anchor : tuple, optional
        Bbox_to_anchor parameter for legend placement. Default is (1.0, 0.5).
    legend_ncol : int, optional
        Number of columns in legend. Default is 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Get categories and items
    categories = df_plot.index.tolist()
    items = df_plot.columns.tolist()
    x_pos = np.arange(len(categories))

    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=figsize)

    # Use standard matplotlib colors if no colors_dict provided, otherwise use colors_dict
    if colors_dict is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        default_colors = prop_cycle.by_key()["color"]
        colors_dict = {
            item: default_colors[idx % len(default_colors)]
            for idx, item in enumerate(items)
        }

    # Get default color cycle as fallback for items not in colors_dict
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]

    # Create stacked vertical bars
    bottom = np.zeros(len(categories))
    bars = []
    labels = []

    for idx, item in enumerate(items):
        values = df_plot[item].values
        # Use color from colors_dict if available, otherwise use default color cycle
        color = colors_dict.get(item, default_colors[idx % len(default_colors)])
        bar = ax.bar(
            x_pos,
            values,
            bottom=bottom,
            align="center",
            label=item,
            color=color,
        )
        bars.append(bar)
        labels.append(item)
        bottom += values

    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)

    # Set y-axis label and limits
    ax.set_ylabel(ylabel)
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    else:
        ax.set_ylim(0, df_plot.sum(axis=1).max() * 1.05)

    # Add legend on the right side, no border
    ax.legend(
        loc="center left",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=10,
        frameon=False,
    )

    # Add grid for better readability
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(top=True, right=True, ax=ax)

    # Set tick label font sizes
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Add overall title if provided
    if title is not None:
        ax.set_title(title, fontsize=10, y=1.005)

    # Adjust layout to accommodate legend on the right
    fig.subplots_adjust(right=0.7)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _plot_stacked_bars_on_ax(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    colors_dict: dict[str, str] | None = None,
    title: str | None = None,
    ylim_max: float | None = None,
    show_legend: bool = True,
    legend_bbox_to_anchor: tuple = (1.0, 0.5),
    legend_ncol: int = 1,
    title_pad: float = 12.0,
    annotate_values: bool = False,
    value_labels_df: pd.DataFrame | None = None,
    value_fmt: str = "{:.0f}",
    value_min_height: float = 0.02,
):
    categories = df_plot.index.tolist()
    items = df_plot.columns.tolist()
    x_pos = np.arange(len(categories))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]

    bottom = np.zeros(len(categories))
    for idx, item in enumerate(items):
        values = df_plot[item].values
        if colors_dict is None:
            color = default_colors[idx % len(default_colors)]
        else:
            color = colors_dict.get(item, default_colors[idx % len(default_colors)])
        bars = ax.bar(
            x_pos,
            values,
            bottom=bottom,
            align="center",
            label=item,
            color=color,
        )
        if annotate_values:
            label_values = (
                value_labels_df[item].values
                if value_labels_df is not None and item in value_labels_df.columns
                else values
            )
            for rect, label_value in zip(bars, label_values):
                height = rect.get_height()
                if height <= 0:
                    continue
                if ylim_max is not None and ylim_max > 0:
                    if height / ylim_max < value_min_height:
                        continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + height / 2,
                    value_fmt.format(label_value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        bottom += values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    else:
        ax.set_ylim(0, df_plot.sum(axis=1).max() * 1.05)

    if show_legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=legend_ncol,
            fontsize=10,
            frameon=False,
        )

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(top=True, right=True, ax=ax)

    if title is not None:
        ax.set_title(title, fontsize=12, y=1.005, pad=title_pad)


def plot_cell_types_per_domain_stacked(
    df_counts: pd.DataFrame,
    figsize: tuple = (2, 6),
    save_path: str | None = None,
    title: str | None = None,
    colors_dict: dict[str, str] | None = None,
):
    """
    Plot normalized annotation proportions per domain as stacked vertical bars.

    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with domains as index and cell types as columns. Typically
        output from `adata.obs.groupby(["domain"], observed=True)["cell_type_l2"]
        .value_counts(normalize=True).unstack()`.
    figsize : tuple, optional
        Figure size (width, height). Default is (2, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Overall title for the figure. If None, no title is added.
    colors_dict : dict[str, str], optional
        Dictionary mapping cell type names to hex color codes. If None, uses
        colorcet glasbey colormap for distinct colors.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Normalize each row to sum to 1 (in case it's not already normalized)
    df_plot = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)

    return _plot_stacked_bars(
        df_plot=df_plot,
        figsize=figsize,
        xlabel="Domain",
        ylabel="Proportion of cell types",
        colors_dict=colors_dict,
        title=title,
        save_path=save_path,
        ylim_max=1.0,
    )


def plot_clone_counts_per_domain_stacked(
    df_counts: pd.DataFrame,
    top_n: int = 10,
    normalize: bool = True,
    figsize: tuple = (2, 6),
    save_path: str | None = None,
    title: str | None = None,
    colors_dict: dict[str, str] | None = None,
    ylim_max: float | None = None,
):
    """
    Plot clone counts per domain as stacked vertical bars, showing top n clones and aggregating the rest as "other alphabeta clones".

    Parameters
    ----------
    df_counts : pd.DataFrame
        DataFrame with domains as index and clone IDs as columns. Typically
        output from `adata.obs.groupby(["domain"], observed=True)["avbv_clone"]
        .value_counts(normalize=False).unstack().fillna(0)`.
    top_n : int, optional
        Number of top clones to show individually. Remaining clones are aggregated
        into "other alphabeta clones". Default is 10.
    normalize : bool, optional
        If True, normalize each row to sum to 1 (proportions). If False, show
        raw counts. Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (2, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Overall title for the figure. If None, no title is added.
    colors_dict : dict[str, str], optional
        Dictionary mapping clone names to hex color codes. If None, uses
        colorcet glasbey colormap for distinct colors. The "other alphabeta clones"
        category will always be colored light gray regardless of this parameter.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Order columns by cumulative abundance (sum across all rows), descending
    df_counts = df_counts.loc[
        :, df_counts.sum(axis=0).sort_values(ascending=False).index
    ]

    # Get top n clones and aggregate the rest
    top_clones = df_counts.columns[:top_n].tolist()
    other_clones = df_counts.columns[top_n:].tolist()

    # Create aggregated dataframe
    df_plot = df_counts[top_clones].copy()
    if len(other_clones) > 0:
        df_plot["other αβ clones"] = df_counts[other_clones].sum(axis=1)

    # Normalize if requested
    if normalize:
        df_plot = df_plot.div(df_plot.sum(axis=1), axis=0).fillna(0)

    # Set up colors: use provided colors_dict for top clones, light gray for "other"
    # Start with provided colors_dict if available, otherwise None
    plot_colors_dict = colors_dict.copy() if colors_dict is not None else None

    # Always use light gray for "other alphabeta clones"
    if "other αβ clones" in df_plot.columns:
        if plot_colors_dict is None:
            plot_colors_dict = {}
        plot_colors_dict["other αβ clones"] = "#D3D3D3"  # light gray

    # Determine ylabel based on normalization
    ylabel = "Proportion of clones" if normalize else "Count of clones"

    return _plot_stacked_bars(
        df_plot=df_plot,
        figsize=figsize,
        xlabel="Domain",
        ylabel=ylabel,
        colors_dict=plot_colors_dict,
        title=title,
        save_path=save_path,
        ylim_max=ylim_max if ylim_max is not None else (1.0 if normalize else None),
    )


def plot_tcell_infiltrate_per_domain_stacked(
    adata,
    domain_col: str = "domain_new",
    infiltrate_col: str = "tcell_infiltrate",
    figsize: tuple = (2, 6),
    save_path: str | None = None,
    title: str | None = None,
    colors_dict: dict[str, str] | None = None,
    rotate_xticks: int = 90,
    exclude_domains: list[str] | None = None,
):
    """
    Plot percentages of cells in T cell infiltrate vs not in infiltrate per domain as stacked vertical bars.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the domain and infiltrate information.
    domain_col : str, optional
        Column name in adata.obs containing domain information. Default is "domain_new".
    infiltrate_col : str, optional
        Column name in adata.obs containing T cell infiltrate information. Default is "tcell_infiltrate".
    figsize : tuple, optional
        Figure size (width, height). Default is (2, 6).
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Overall title for the figure. If None, no title is added.
    colors_dict : dict[str, str], optional
        Dictionary mapping infiltrate status to hex color codes. If None, uses
        colorcet glasbey colormap for distinct colors.
    rotate_xticks : int, optional
        Rotation angle for x-axis tick labels in degrees. Default is 90.
    exclude_domains : list[str], optional
        List of domain names to exclude from the plot. If None, all domains are included.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Get counts of cells per domain and infiltrate status
    df_counts = (
        adata.obs.groupby([domain_col, infiltrate_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    # Exclude specified domains if requested
    if exclude_domains is not None:
        df_counts = df_counts[~df_counts.index.isin(exclude_domains)]

    # Reorder columns so "no infiltrate" comes first (bottom) and "infiltrate" comes last (top)
    desired_order = ["no\ninfiltrate", "infiltrate"]
    available_cols = [col for col in desired_order if col in df_counts.columns]
    df_counts = df_counts[available_cols]

    # Normalize each row to sum to 1 (percentages)
    df_plot = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)

    fig, ax = _plot_stacked_bars(
        df_plot=df_plot,
        figsize=figsize,
        xlabel="Domain",
        ylabel="Proportion of cells",
        colors_dict=colors_dict,
        title=title,
        save_path=None,  # Don't save yet, need to apply modifications first
        ylim_max=1.0,
    )

    # Rotate x-axis tick labels and center-align them
    ax.tick_params(axis="x", rotation=rotate_xticks, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("center")

    # Save the figure after all modifications
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_cd8_clonal_clusters_vs_cells_stacked(
    adata,
    region_col: str = "region",
    cc_col: str = "cd8_dominant_ccs",
    cell_type_col: str = "cell_type_l1.1",
    cell_type: str = "CD8+",
    sample_col: str | None = None,
    normalize: bool = True,
    title: str | None = None,
    annotate_significance: bool = True,
    adjust: str = "fdr_bh",
    alpha: float = 0.05,
    show_raw_pvalues: bool = False,
    show_absolute_counts: bool = False,
    test_method: str = "chi2",
    figsize: tuple = (4, 4),
    ncols: int = 3,
    exclude_regions: list[str] | None = None,
    return_counts: bool = False,
):
    """
    Stacked bar plot comparing CD8 clonal cluster vs CD8 T-cell distributions
    across regions (domains). Optionally facet per sample with FDR-BH on globals.
    """

    def _sample_order(values: pd.Series) -> list:
        if pd.api.types.is_categorical_dtype(values):
            return [v for v in values.cat.categories if v in values.unique()]
        return values.dropna().unique().tolist()

    if sample_col is not None:
        samples = _sample_order(adata.obs[sample_col])
        n_panels = len(samples)
        nrows = int(np.ceil(n_panels / ncols)) if n_panels > 0 else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            sharey=True,
        )
        axes_flat = np.array(axes).ravel()
    else:
        samples = [None]
        fig, ax = plt.subplots(figsize=figsize)
        axes_flat = np.array([ax])

    counts_by_sample: dict[str | None, pd.DataFrame] = {}
    test_stats = []
    legend_handles = None
    legend_labels = None

    for i, sample in enumerate(samples):
        if sample is None:
            adata_sub = adata
            panel_title = title or f"{cell_type} clonal clusters vs {cell_type} T cells"
        else:
            adata_sub = adata[adata.obs[sample_col] == sample]
            panel_title = str(sample)

        counts_df = _cd8_counts_by_region(
            adata=adata_sub,
            region_col=region_col,
            cc_col=cc_col,
            cell_type_col=cell_type_col,
            cell_type=cell_type,
            exclude_regions=exclude_regions,
        )
        counts_by_sample[sample] = counts_df

        global_test = _chi2_homogeneity_test(counts_df, test_method=test_method)
        test_stats.append(
            {
                "sample": sample,
                "chi2": global_test["chi2"],
                "dof": global_test["dof"],
                "p": global_test["p"],
                "method": global_test["method"],
                "oddsratio": global_test.get("oddsratio", np.nan),
            }
        )
        sample_label = "all samples" if sample is None else str(sample)
        print(f"Global test ({global_test['method']}) for {sample_label}")

        stacked_df = counts_df.T
        if normalize:
            stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0).fillna(0)

        ax = axes_flat[i]
        _plot_stacked_bars_on_ax(
            ax=ax,
            df_plot=stacked_df,
            xlabel="Group",
            ylabel="Proportion" if normalize else "Count",
            colors_dict=None,
            title=panel_title,
            ylim_max=1.0 if normalize else None,
            show_legend=(sample_col is None and i == 0),
            annotate_values=show_absolute_counts,
            value_labels_df=counts_df.T,
        )
        if sample_col is not None and i % ncols != 0:
            ax.set_ylabel("")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    stats_df = pd.DataFrame(test_stats).set_index("sample")
    if annotate_significance:
        if not stats_df.empty:
            pvals = stats_df["p"].to_numpy()
            _, p_adj, _, _ = multipletests(pvals, method=adjust)
            stats_df["p_adj"] = p_adj

            for i, sample in enumerate(samples):
                p_use = (
                    stats_df.loc[sample, "p"]
                    if show_raw_pvalues
                    else stats_df.loc[sample, "p_adj"]
                )
                if np.isfinite(p_use):
                    _annotate_global_star(
                        axes_flat[i], pvalue=float(p_use), show_stars=True
                    )

    if sample_col is not None and len(axes_flat) > len(samples):
        for j in range(len(samples), len(axes_flat)):
            axes_flat[j].set_visible(False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*")
        fig.tight_layout()
    if sample_col is not None and legend_handles is not None:
        n_domains = len(legend_labels)
        ncol = min(2, n_domains)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.12)

    if return_counts:
        return fig, axes_flat, counts_by_sample, stats_df
    return fig, axes_flat


def cc_enrichment_by_region(
    observed_ccs: pd.Series,
    t_cells: pd.Series,
    adjust: str = "fdr_bh",
    alternative: str = "two-sided",
    min_expected: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    from scipy.stats import binomtest, chisquare

    df = (
        pd.DataFrame({"obs_cc": observed_ccs, "t_cells": t_cells})
        .fillna(0)
        .astype({"obs_cc": int})
    )

    df = df[df["t_cells"] > 0]
    total_ccs = int(df["obs_cc"].sum())
    total_t = float(df["t_cells"].sum())

    if df.shape[0] < 2 or total_ccs == 0 or total_t == 0:
        global_test = {
            "chi2": np.nan,
            "dof": np.nan,
            "p": 1.0,
            "total_ccs": total_ccs,
            "total_t_cells": total_t,
        }
        stats_df = pd.DataFrame(
            index=df.index, columns=["obs", "exp", "enrichment", "p", "p_adj", "reject"]
        )
        return global_test, stats_df

    p0 = df["t_cells"] / total_t
    exp = p0 * total_ccs

    # Global goodness-of-fit (R-1 df)
    chi2_res = chisquare(f_obs=df["obs_cc"].to_numpy(), f_exp=exp.to_numpy())
    global_test = {
        "chi2": float(chi2_res.statistic),
        "dof": int(df.shape[0] - 1),
        "p": float(chi2_res.pvalue),
        "total_ccs": total_ccs,
        "total_t_cells": total_t,
    }

    # Per-region exact binomial tests (region vs rest)
    pvals = []
    enrichment = []
    for region in df.index:
        k = int(df.loc[region, "obs_cc"])
        e = float(exp.loc[region])
        p_region = float(p0.loc[region])

        enrichment.append((k / e) if e > 0 else np.nan)

        # If expected is tiny, binomtest still works; min_expected is just a flag you can use downstream
        pvals.append(
            float(
                binomtest(k=k, n=total_ccs, p=p_region, alternative=alternative).pvalue
            )
        )

    reject, p_adj, _, _ = multipletests(pvals, method=adjust)

    stats_df = pd.DataFrame(
        {
            "obs": df["obs_cc"].astype(int),
            "exp": exp.astype(float),
            "enrichment": np.array(enrichment, dtype=float),
            "p": np.array(pvals, dtype=float),
            "p_adj": np.array(p_adj, dtype=float),
            "reject": np.array(reject, dtype=bool),
            "expected_ok": (exp >= float(min_expected)).to_numpy(),
        },
        index=df.index,
    )

    return global_test, stats_df


def _cd8_counts_by_region(
    adata,
    region_col: str,
    cc_col: str,
    cell_type_col: str,
    cell_type: str,
    exclude_regions: list[str] | None = None,
) -> pd.DataFrame:
    if exclude_regions is not None:
        adata = adata[~adata.obs[region_col].isin(exclude_regions)]

    adata_cc = adata.obs[adata.obs[cc_col].notna()].copy()
    if len(adata_cc) > 0:
        cc_region_counts = (
            adata_cc.groupby([cc_col, region_col], observed=True)
            .size()
            .reset_index(name="count")
        )
        cc_to_region = cc_region_counts.loc[
            cc_region_counts.groupby(cc_col, observed=True)["count"].idxmax()
        ].set_index(cc_col)[region_col]
        cc_counts = cc_to_region.value_counts()
    else:
        cc_counts = pd.Series(dtype=int)

    cell_type_t_cells_counts = (
        adata.obs[adata.obs[cell_type_col] == cell_type]
        .groupby(region_col, observed=True)
        .size()
    )

    counts_df = pd.DataFrame(
        {
            f"{cell_type}\nclonal clusters": cc_counts,
            f"{cell_type}\nT cells": cell_type_t_cells_counts,
        }
    ).fillna(0)

    region_order = ["Glomerular", "Periglomerular", "Tubulointerstitial"]
    counts_df = counts_df.reindex(
        [r for r in region_order if r in counts_df.index]
        + [r for r in counts_df.index if r not in region_order]
    )

    return counts_df


def _chi2_homogeneity_test(counts_df: pd.DataFrame, test_method: str = "chi2") -> dict:
    if test_method not in {"chi2", "fisher"}:
        raise ValueError("test_method must be 'chi2' or 'fisher'.")
    # counts_df has regions as index and two columns (clusters, T cells)
    counts = counts_df.T
    counts = counts.loc[:, ~((counts == 0).all(axis=0))]

    if counts.shape[1] < 2 or counts.to_numpy().sum() == 0:
        return {"chi2": np.nan, "dof": np.nan, "p": 1.0, "method": test_method}

    if test_method == "fisher":
        if counts.shape[1] == 2:
            oddsratio, p = fisher_exact(counts.to_numpy(), alternative="two-sided")
            return {
                "oddsratio": float(oddsratio),
                "chi2": np.nan,
                "dof": np.nan,
                "p": float(p),
                "method": "fisher",
            }
        print("Requested fisher but found != 2 domains; using chi2 instead.")

    chi2, p, dof, _ = chi2_contingency(counts.to_numpy(), correction=False)
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p), "method": "chi2"}


def _annotate_global_star(
    ax: plt.Axes,
    pvalue: float | None = None,
    stars: str | None = None,
    y_offset_frac: float = 0.04,
    fontsize: int = 11,
    line_width: float = 1.0,
    show_stars: bool = True,
):
    if pvalue is None or not np.isfinite(pvalue):
        if stars is None:
            return
        p_label = None
        stars_label = str(stars)
    else:
        p_label = f"p={float(pvalue):.3g}"
        stars_label = _p_to_stars(float(pvalue)) if show_stars else None

    containers = list(ax.containers)
    if not containers:
        return

    n_bars = len(list(containers[0]))
    if n_bars != 2:
        return

    x_centers = [bar.get_x() + bar.get_width() / 2 for bar in list(containers[0])]
    heights = []
    for i in range(n_bars):
        heights.append(
            sum(
                container[i].get_height()
                for container in containers
                if len(container) > i
            )
        )

    ymax = max(heights + [0.0])
    y_offset = max(1e-12, ymax * y_offset_frac)
    h = y_offset * 0.5

    x0, x1 = x_centers
    y = max(heights) + y_offset
    # Standard bracket above bars, label just below the line
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=line_width, c="black")
    if p_label is not None:
        ax.text(
            (x0 + x1) / 2,
            y + h * 0.80,
            p_label,
            ha="center",
            va="top",
            fontsize=fontsize,
        )
    if stars_label is not None:
        ax.text(
            (x0 + x1) / 2,
            y + h * 1.35,
            stars_label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )
    ax.set_ylim(0, max(ax.get_ylim()[1], y + h * 2.5))


def plot_cd8_clonal_clusters_vs_cells_per_region(
    adata,
    region_col: str = "region",
    cc_col: str = "cd8_dominant_ccs",
    cell_type_col: str = "cell_type_l1.1",
    cell_type: str = "CD8+",
    ax: plt.Axes | None = None,
    normalize: bool = True,
    title: str | None = None,
    return_counts: bool = False,
    exclude_regions: list[str] | None = None,
    annotate_significance: bool = True,
    adjust: str = "fdr_bh",
    alternative: str = "two-sided",
    figsize: tuple = (8, 6),
):
    if exclude_regions is not None:
        adata = adata[~adata.obs[region_col].isin(exclude_regions)]

    # Count unique clonal clusters per region, ensuring each CC is only counted once
    # Assign each CC to the region where it has the most cells
    adata_cc = adata.obs[adata.obs[cc_col].notna()].copy()
    if len(adata_cc) > 0:
        # Count cells per CC per region
        cc_region_counts = (
            adata_cc.groupby([cc_col, region_col], observed=True)
            .size()
            .reset_index(name="count")
        )
        # For each CC, find the region with the most cells
        cc_to_region = cc_region_counts.loc[
            cc_region_counts.groupby(cc_col)["count"].idxmax()
        ].set_index(cc_col)[region_col]
        # Count unique CCs per region
        cc_counts = cc_to_region.value_counts()
    else:
        cc_counts = pd.Series(dtype=int)

    # Count CD8+ T cells per region
    cell_type_t_cells_counts = (
        adata.obs[adata.obs[cell_type_col] == cell_type]
        .groupby(region_col, observed=True)
        .size()
    )

    # Combine into a DataFrame
    counts_df = pd.DataFrame(
        {
            f"{cell_type} clonal clusters": cc_counts,
            f"{cell_type} T cells": cell_type_t_cells_counts,
        }
    ).fillna(0)

    # Ensure consistent ordering of regions
    region_order = ["Glomerular", "Periglomerular", "Tubulointerstitial"]
    counts_df = counts_df.reindex(
        [r for r in region_order if r in counts_df.index]
        + [r for r in counts_df.index if r not in region_order]
    )

    if annotate_significance:
        test_counts_df = pd.DataFrame(
            {
                f"{cell_type} clonal clusters": cc_counts,
                f"{cell_type} T cells": cell_type_t_cells_counts,
            }
        ).fillna(0)

        test_counts_df = test_counts_df.reindex(
            [r for r in region_order if r in test_counts_df.index]
            + [r for r in test_counts_df.index if r not in region_order]
        )

        # Remove regions with zero T cells (undefined expectation)
        test_counts_df = test_counts_df[test_counts_df[f"{cell_type} T cells"] > 0]

        if len(test_counts_df) > 1:
            global_test, stats_df = cc_enrichment_by_region(
                observed_ccs=test_counts_df[f"{cell_type} clonal clusters"],
                t_cells=test_counts_df[f"{cell_type} T cells"],
                adjust=adjust,
                alternative=alternative,
                min_expected=1.0,
            )

            # Optional: only annotate if the global test suggests any deviation.
            if not np.isfinite(global_test["p"]) or global_test["p"] >= 0.05:
                print(f"Global test p-value: {global_test['p']}")
                # stats_df = None
        else:
            stats_df = None
    else:
        stats_df = None

    # Plot using existing function
    if title is None:
        title = f"{cell_type} clonal clusters vs {cell_type} T cells per region"

    # Plot without automatic statistical annotation (we'll add it manually)
    ax = plot_side_by_side_counts(
        df=counts_df,
        col1=f"{cell_type} clonal clusters",
        col2=f"{cell_type} T cells",
        category_name="Region",
        value_name="Count" if not normalize else "Proportion",
        ax=ax,
        normalize=normalize,
        title=title,
        annotate_per_category=False,  # We annotate manually with custom test
        adjust=adjust,
        alternative=alternative,
        figsize=figsize,
        rotate_xticks=0,  # Horizontal x-axis labels
    )

    # Add statistical annotations if requested
    # The test compares the proportion of T cells in clonal clusters across regions
    if annotate_significance and stats_df is not None:
        # Get the order used in the plot (sorted by clonal cluster count)
        order = counts_df.sort_values(
            f"{cell_type} clonal clusters", ascending=False
        ).index.tolist()
        stars = stats_df["p_adj"].apply(_p_to_stars)
        annotate_pairwise_stars(ax, order=order, stars_by_category=stars)

    if return_counts:
        return ax, counts_df
    return ax
