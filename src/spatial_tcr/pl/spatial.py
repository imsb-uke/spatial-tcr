import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from pandas.api.types import is_numeric_dtype
from scipy.sparse import issparse
from shapely.geometry import Polygon

from spatial_tcr.pl.utils import add_spatial_data


def extract_boundary_polygons(df, cell_id_key="cell_id", scaler=1.0):
    grouped = df.groupby(cell_id_key, observed=True)
    # boundary_polygons = {
    #     cell_id: Polygon(list(zip(group["x"], group["y"])))
    #     for cell_id, group in grouped
    # }

    boundary_polygons = {}
    for cell_id, group in grouped:
        coords = np.array(list(zip(group["x"], group["y"])))
        if scaler != 1.0:
            # Calculate centroid
            centroid = coords.mean(axis=0)
            # Scale coordinates relative to centroid
            coords = centroid + (coords - centroid) * scaler
        boundary_polygons[cell_id] = Polygon(coords)

    boundary_patches = {}
    for cell_id, poly in boundary_polygons.items():
        # check if poly is empty
        assert not poly.is_empty, f"Polygon for cell {cell_id} is empty"
        exterior_coords = list(poly.exterior.coords)
        boundary_patches[cell_id] = MplPolygon(exterior_coords, closed=True)
    return boundary_polygons, boundary_patches


def categorical_to_colors(df, categorical, palette="tab10", fill_color="lightgray"):
    # Get unique categories and generate colormap
    if isinstance(palette, dict):
        cmap = palette
    else:
        # create new color map from palette
        categories = df[categorical].astype("category").cat.categories.astype(str)
        if not isinstance(palette, list):
            palette = sns.color_palette(palette, len(categories))
        cmap = dict(zip(categories, palette))

    # Apply the color map to the DataFrame
    colors = df[categorical].astype(str).map(cmap)
    colors = colors.fillna(fill_color)

    # only keep cmap entries that are in the colors
    cmap = {k: v for k, v in cmap.items() if k in df[categorical].unique()}

    return colors, cmap


def numerical_to_colors(values: np.ndarray, cmap="Reds"):
    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(values))
    return colors, cmap, norm


def cmap_to_legend(ax, cmap, title=None, frameon=False, **kwargs):
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=10,
            label=entry,
            linestyle="none",
        )
        for entry, color in cmap.items()
    ]

    # Simply create the legend normally - no complex artist management
    legend = ax.legend(title=title, handles=handles, frameon=frameon, **kwargs)
    return legend


def plot_patches(
    ax,
    patches: dict,
    facecolor="none",
    # cmap=None,
    # legend_title=None,
    edgecolor="gray",
    linewidth=0.1,
    # legend_ax=None,
    rasterized: bool = False,
    **kwargs,
):
    if isinstance(edgecolor, str):
        edgecolor = facecolor if edgecolor == "facecolor" else edgecolor
    keys = list(patches.keys())
    p = PatchCollection(
        [patches[key] for key in keys],
        edgecolor=edgecolor,
        linewidth=linewidth,
        facecolor=facecolor,
        # alpha=kwargs.get("alpha", None),
    )

    # Rasterize the collection if requested to keep vector outputs lightweight
    if rasterized:
        p.set_rasterized(True)
    ax.add_collection(p)


def plot_legend(
    ax,
    cmap,
    norm=None,
    legend_title=None,
    marker="o",
    **kwargs,
):
    if norm is not None:
        # numeric legend
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(
            sm,
            ax=ax,
            pad=0.01,
            shrink=kwargs.get("cbar_shrink", 0.3),
            aspect=kwargs.get("cbar_aspect", 10),
            fraction=kwargs.get("cbar_fraction", 0.075),
        )  # Specify the axis here
        cbar.set_label(kwargs.get("cbar_label", ""), rotation=270, labelpad=15)
        return cbar
    else:
        # categorical legend
        if cmap is not None:
            # Create legend on the figure level to avoid conflicts
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=10,
                    label=entry,
                    linestyle="none",
                )
                for entry, color in cmap.items()
            ]

            # Create figure-level legend
            fig = ax.get_figure()
            # Use fontsize parameter to set legend entry font size
            default_prop = {"size": kwargs.get("fontsize", 10)}
            prop = kwargs.get("prop", default_prop)
            legend = fig.legend(
                handles,
                list(cmap.keys()),
                title=legend_title,
                loc="center left",
                bbox_to_anchor=kwargs.get("bbox_to_anchor", (0.95, 0.5)),
                prop=prop,
                markerscale=kwargs.get("markerscale", 0.5),
                frameon=False,
                title_fontsize=kwargs.get("fontsize", 10),
            )

            return legend


def plot_shapes():
    pass


def plot_points():
    pass


def define_patch_colors(adata, color_key, patches, **kwargs):
    if color_key is not None:
        in_obs = color_key in adata.obs.columns
        in_var = color_key in adata.var_names
        assert in_obs or in_var, (
            f"Cell color {color_key} not found in adata.obs or adata.var"
        )
        assert not (in_obs and in_var), (
            "Cell color found in both adata.obs and adata.var"
        )
        is_numeric = is_numeric_dtype(adata.obs[color_key]) if in_obs else in_var
        if is_numeric:
            values = adata.obs[color_key].values if in_obs else adata[:, color_key].X
            # convert sparse to dense
            values = values.toarray().flatten() if issparse(values) else values

            vmax = kwargs.get("vmax", values.max())
            vmin = kwargs.get("vmin", values.min())
            values = np.clip(values, vmin, vmax)

            colors, cmap, norm = numerical_to_colors(
                values=values, cmap=kwargs.get("cmap", "Reds")
            )
        else:
            colors, cmap = categorical_to_colors(
                adata.obs.loc[patches.keys()],
                color_key,
                palette=kwargs.get("palette", "tab20"),
                fill_color=kwargs.get("fill_color", "lightgray"),
            )
            norm = None
    else:
        colors = "none"
        cmap = None
        norm = None
    return colors, cmap, norm


def plot_multi_channel_image(
    ax, image, channels=None, normalize=True, black_to_white=False, **kwargs
):
    if channels is None:
        channels = image.coords["channel"].values[0:3]
        print(f"No channels specified, using the first 3 channels: {channels}")
    if isinstance(channels, str) or isinstance(channels, int):
        channels = [channels]
    assert len(channels) <= 3, "Only up to 3 channels can be plotted at once"

    image = image.sel(channel=channels).values
    vmax = kwargs.get("vmax", image.max(axis=0).max(axis=0))
    vmax = np.array(vmax) if isinstance(vmax, list) else vmax
    vmin = kwargs.get("vmin", image.min(axis=0).min(axis=0))
    vmin = np.array(vmin) if isinstance(vmin, list) else vmin

    # clip each channel
    image = np.clip(image, vmin, vmax)

    # normalize image per channel
    if normalize:
        image = (image - vmin) / (vmax - vmin)

    if black_to_white:
        image = 1 - image

    cmap = (
        sns.color_palette(kwargs.get("cmap", "gray_r"), as_cmap=True)
        if len(channels) == 1
        else None
    )

    image = image[:, :, 0] if len(channels) == 1 else image

    ax.imshow(image, cmap=cmap)


def plot_spatial(
    adata,
    plot_nuc=False,
    plot_cell=False,
    plot_img=True,
    nuc_color=None,
    cell_color=None,
    genes=None,
    dpi=300,
    img_kwargs=None,
    gene_kwargs=None,
    nuc_kwargs=None,
    cell_kwargs=None,
    show_legend=True,
    figsize=None,
    tick_size=250,  # microns
    grid=False,
    img_channel: int | list[str] = 0,
    frameon=False,
    show_scale_bar=False,
    scale_bar_kwargs=None,
    return_legend=False,
    extra_fn=None,
    legend_positions=None,
    legend_fontsize=10,
    **kwargs,
):
    nuc_boundaries = adata.uns["sdata"]["nuc_boundaries"]
    cell_boundaries = adata.uns["sdata"]["cell_boundaries"]
    transcripts = adata.uns["sdata"]["transcripts"]
    image = adata.uns["sdata"]["image"]
    pixel_size = adata.uns["sdata"]["pixel_size"]

    if img_kwargs is None:
        img_kwargs = {}
    if gene_kwargs is None:
        gene_kwargs = {}
    if nuc_kwargs is None:
        nuc_kwargs = {}
    if cell_kwargs is None:
        cell_kwargs = {}
    if scale_bar_kwargs is None:
        scale_bar_kwargs = {}

    obs_names = adata.obs.index.tolist()

    # Default legend positions to avoid overlap - adjusted for figure-level legends
    if legend_positions is None:
        legend_positions = {
            "cell": {"bbox_to_anchor": (0.9, 0.6)},
            "genes": {"bbox_to_anchor": (0.9, 0.5)},
            "nuc": {"bbox_to_anchor": (0.9, 0.2)},
        }

    sns.set_theme(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if plot_img:
        plot_multi_channel_image(ax, image, channels=img_channel, **img_kwargs)
    else:
        ax.imshow(np.zeros_like(image), cmap="gray_r")

    # save the limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if extra_fn is not None:
        extra_fn(ax)

    legends = {}

    if plot_cell:
        cell_polygons, cell_patches = extract_boundary_polygons(
            cell_boundaries, scaler=cell_kwargs.get("scaler", 1.0)
        )
        assert len(obs_names) == len(cell_patches)

        colors, cmap, norm = define_patch_colors(
            adata, cell_color, cell_patches, **cell_kwargs
        )
        color_edges = cell_kwargs.pop("color_edges", False)
        cell_kwargs["edgecolor"] = (
            colors if color_edges else cell_kwargs.get("edgecolor", "gray")
        )
        cell_kwargs["facecolor"] = (
            colors if not color_edges else cell_kwargs.get("facecolor", "none")
        )
        plot_patches(ax, cell_patches, **cell_kwargs)
        _cmap_str = cell_kwargs.pop("cmap", None)
        # plot_legend(ax, cmap, norm, **cell_kwargs)
        # Add cell legend with custom position
        if cmap is not None and show_legend:
            cell_legend_kwargs = {**cell_kwargs, **legend_positions.get("cell", {})}
            cell_legend = plot_legend(
                ax,
                cmap,
                norm,
                legend_title=cell_kwargs.get("legend_title", "Cells"),
                marker=cell_kwargs.get("marker", "s"),
                loc="center left",
                fontsize=legend_fontsize,
                **cell_legend_kwargs,
            )
            legends["cell"] = cell_legend

    if plot_nuc:
        nuc_polygons, nuc_patches = extract_boundary_polygons(
            nuc_boundaries, scaler=nuc_kwargs.get("scaler", 1.0)
        )
        assert len(obs_names) >= len(nuc_patches)

        colors_nuc, cmap_nuc, norm_nuc = define_patch_colors(
            adata, color_key=nuc_color, patches=nuc_patches, **nuc_kwargs
        )
        plot_patches(
            ax,
            nuc_patches,
            facecolor=colors_nuc,
            **nuc_kwargs,
        )
        _cmap_str = nuc_kwargs.pop("cmap", None)
        # plot_legend(ax, cmap_nuc, norm_nuc, **nuc_kwargs)
        # Add nucleus legend with custom position
        if cmap_nuc is not None and show_legend:
            nuc_legend_kwargs = {**nuc_kwargs, **legend_positions.get("nuc", {})}
            nuc_legend = plot_legend(
                ax,
                cmap_nuc,
                norm_nuc,
                legend_title=nuc_kwargs.get("legend_title", "Nucleus"),
                loc="center left",
                fontsize=legend_fontsize,
                **nuc_legend_kwargs,
            )
            legends["nuc"] = nuc_legend

    if genes is not None:
        if isinstance(genes, dict):
            # show multiple genes as one
            inverted_genes = {
                gene: key for key, gene_list in genes.items() for gene in gene_list
            }
            transcripts_sub = transcripts[
                transcripts["feature_name"].isin(inverted_genes.keys())
            ].copy()
            transcripts_sub["feature_name"] = transcripts_sub["feature_name"].replace(
                inverted_genes
            )
            transcripts_sub["feature_name"] = pd.Categorical(
                transcripts_sub["feature_name"],
                categories=genes.keys(),
                ordered=True,
            )
            colors, color_map_gene = categorical_to_colors(
                transcripts_sub,
                "feature_name",
                palette=gene_kwargs.get("palette", "nipy_spectral"),
            )
        else:
            overlap_genes = [
                g for g in genes if g in transcripts["feature_name"].unique()
            ]
            if len(overlap_genes) == 0:
                print(f"No transcripts found for the given genes: {genes}")
            else:
                print(
                    f"The following genes are present in the transcripts data: {overlap_genes}"
                )
                transcripts_sub = transcripts[
                    transcripts["feature_name"].isin(overlap_genes)
                ].copy()
                transcripts_sub["feature_name"] = pd.Categorical(
                    transcripts_sub["feature_name"],
                    categories=overlap_genes,
                    ordered=True,
                )
                colors, color_map_gene = categorical_to_colors(
                    transcripts_sub,
                    "feature_name",
                    palette=gene_kwargs.get("palette", "nipy_spectral"),
                )
        ax.scatter(
            transcripts_sub["x"],
            transcripts_sub["y"],
            c=colors,
            s=gene_kwargs.get("s", 0.1),
            edgecolors=gene_kwargs.get("edgecolors", "none"),
        )

        # Add gene legend with custom position
        if show_legend:
            gene_legend_pos = legend_positions.get("genes", {})
            gene_legend = plot_legend(
                ax,
                color_map_gene,
                legend_title="Genes",
                loc="center left",
                fontsize=legend_fontsize,
                **gene_legend_pos,
            )
            legends["genes"] = gene_legend

    # frameon = frameon if not show_scale_bar else False

    if frameon:
        # introduce xticks and yticks every n pixels
        n = int(tick_size / pixel_size)
        xticks = np.arange(0, image.shape[1], n)
        ax.set_xticks(xticks)
        yticks = np.arange(0, image.shape[0], n)
        ax.set_yticks(yticks)

        # remove xticklabels and yticklabels
        tick_labels = np.array(list(range(0, 100000, tick_size)))
        ax.set_xticklabels(tick_labels[0 : len(xticks)])
        ax.set_yticklabels(tick_labels[0 : len(yticks)])

        # Reduce size of xticklabels
        if kwargs.get("labelsize", None):
            ax.tick_params(axis="x", labelsize=kwargs["labelsize"])
            ax.tick_params(axis="y", labelsize=kwargs["labelsize"])

        ax.set_xlabel("x-coordinate [µm]")
        ax.set_ylabel("y-coordinate [µm]")

        # make grid lines transparent and very thin
        if grid:
            ax.grid(True, linewidth=0.5, alpha=0.3, linestyle="--")

        sns.despine(top=True, right=True)
    else:
        if show_scale_bar:
            # Calculate scale bar length in pixels
            n = int(tick_size / pixel_size)

            y_offset = scale_bar_kwargs.get("y_offset", 0.05)
            x_offset = scale_bar_kwargs.get("x_offset", 0.05)

            # Position scale bar at bottom left, slightly offset from corner
            bar_y = ylim[0] - (ylim[0] - ylim[1]) * y_offset  # 5% from bottom
            bar_x_start = xlim[0] + (xlim[1] - xlim[0]) * x_offset  # 5% from left

            # Draw scale bar
            ax.plot(
                [bar_x_start, bar_x_start + n],
                [bar_y, bar_y],
                color=scale_bar_kwargs.get("color", "black"),
                linewidth=scale_bar_kwargs.get("linewidth", 2),
            )

            # Add label
            ax.text(
                bar_x_start + n / 2,
                bar_y + 0.25 * y_offset * (ylim[0] - ylim[1]),
                f"{tick_size} µm",
                ha="center",
                va="top",
                fontsize=scale_bar_kwargs.get("fontsize", 10),
                color=scale_bar_kwargs.get("color", "black"),
            )

        # set axis off
        ax.axis("off")

    # crop plot to the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if not show_legend:
        # Remove all legends
        for legend in legends.values():
            if legend is not None:
                try:
                    legend.remove()
                except:
                    pass

    # Adjust figure layout - figure-level legends need less space adjustment
    if show_legend and any(legend is not None for legend in legends.values()):
        # Leave more space for figure-level legends to accommodate longer labels
        fig.subplots_adjust(left=0, right=0.89, top=1, bottom=0)
    else:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if return_legend:
        return fig, ax, legends
    else:
        return fig, ax


def plot_cells_view(
    adata,
    data_dir,
    sample,
    color_key,
    palette,
    cluster=None,
    cell_ids=None,
    gene_palette=None,
    cluster_key="avbv_cluster",
    sample_key="cc",
    ad_sample=None,
    genes=None,
    x_expansion=None,
    y_expansion=None,
    xlength=200,
    ylength=200,
    avbv_map=None,
    tick_size=30,
    gene_kwargs=None,
    img_kwargs=None,
    cell_kwargs=None,
    figsize=(5, 10),
):
    ad_sub = (
        add_spatial_data(
            adata,
            data_dir,
            sample_key=sample_key,
            sample=sample,
            cell_id_key="cell_id",
            level=0,
        )
        if not ad_sample
        else ad_sample
    )

    ad_sub.obs_names = ad_sub.obs["cell_id"].tolist()

    if cell_ids is not None:
        cluster_coords = ad_sub[ad_sub.obs["cell_id"].isin(cell_ids)].obsm[
            "spatial_translated"
        ]
    elif cluster is not None:
        # filter for cluster
        cluster_coords = ad_sub[ad_sub.obs[cluster_key] == cluster].obsm[
            "spatial_translated"
        ]
    else:
        raise ValueError("Either cell_ids or cluster must be provided")

    ymin, ymax = cluster_coords[:, 1].min(), cluster_coords[:, 1].max()
    xmin, xmax = cluster_coords[:, 0].min(), cluster_coords[:, 0].max()

    y_expansion = (ylength - (ymax - ymin)) / 2 if not y_expansion else y_expansion
    x_expansion = (xlength - (xmax - xmin)) / 2 if not x_expansion else x_expansion
    ymin, ymax = ymin - y_expansion, ymax + y_expansion
    xmin, xmax = xmin - x_expansion, xmax + x_expansion

    ylim = [ymin, ymax]
    xlim = [xmin, xmax]

    ad_zoom = add_spatial_data(
        ad_sub,
        data_dir,
        xlim=xlim,
        ylim=ylim,
        cell_id_key="cell_id",
        level=0,
        fit_to_cells=False,
    )
    ad_zoom.obs_names = ad_zoom.obs["cell_id"].tolist()
    if cluster is not None and avbv_map is not None:
        gene_aliases = cluster.split("_C")[0].split("+")
        genes = {g: avbv_map.get(g, [g]) for g in gene_aliases}
        print(f"Plotting cluster {cluster} with genes {genes}")
    else:
        genes = genes

    fig, ax, legends = plot_spatial(
        ad_zoom,
        plot_cell=True,
        dpi=300,
        cell_color=color_key,
        genes=genes,
        img_kwargs={"vmax": 5000} if not img_kwargs else img_kwargs,
        show=False,
        gene_kwargs={"s": 5, "palette": gene_palette}
        if gene_palette
        else {"s": 5}
        if not gene_kwargs
        else gene_kwargs,
        # cell_kwargs={
        #     "palette": palette,
        #     "edgecolor": "lightgray",
        #     "linewidth": 0.2,
        # },
        cell_kwargs={
            "palette": palette,
            "linewidth": 2,
            "color_edges": True,
            "fill_color": "none",
        }
        if not cell_kwargs
        else cell_kwargs,
        figsize=figsize,
        labelsize=5,
        tick_size=tick_size,
        img_channel=[
            # "CD45",
            # "RNA",
            "DAPI",
        ],
        grid=False,
        frameon=False,
        show_scale_bar=True,
        scale_bar_kwargs={"color": "black", "x_offset": 0.075},
        show_legend=True,
        return_legend=True,
    )
    return ad_zoom, fig, ax, legends


#################### experimental ####################


def pixelfy_limits(xlim, ylim, pixel_size):
    xlim = np.array(xlim)
    ylim = np.array(ylim)
    xlim_pixel = np.round(xlim / pixel_size).astype(int)
    ylim_pixel = np.round(ylim / pixel_size).astype(int)
    return xlim_pixel, ylim_pixel


def transform_coordinates(coords, xlim, ylim, scale_factor=1):
    coords_shifted = coords - np.array([xlim[0], ylim[0]])[None, :]
    coords_transformed = coords_shifted * scale_factor
    return coords_transformed


def limits_to_mask(coords, xlim, ylim):
    x, y = coords.T
    mask_x = (x >= xlim[0]) & (x <= xlim[1])
    mask_y = (y >= ylim[0]) & (y <= ylim[1])
    mask = mask_x & mask_y
    return mask


def crop_adata(adata, xlim, ylim):
    xlim, ylim = pixelfy_limits(xlim, ylim, 1.0)
    print(xlim, ylim)

    coords = adata.obsm["spatial"]
    mask = limits_to_mask(coords, xlim, ylim)
    adata = adata[mask].copy()

    # handle the uns data
    nuc_boundaries = adata.uns["nuc_boundaries"]
    cell_boundaries = adata.uns["cell_boundaries"]
    transcripts = adata.uns["transcripts"]
    image = adata.uns["image"]
    _pixel_size = adata.uns["pixel_size"]

    mask = nuc_boundaries["cell_id"].isin(adata.obs_names)
    nuc_boundaries = nuc_boundaries[mask].copy()

    mask = cell_boundaries["cell_id"].isin(adata.obs_names)
    cell_boundaries = cell_boundaries[mask].copy()

    mask = transcripts["x"].between(xlim[0], xlim[1]) & transcripts["y"].between(
        ylim[0], ylim[1]
    )
    transcripts = transcripts[mask].copy()

    image = image[ylim[0] : ylim[1], xlim[0] : xlim[1]].copy()

    # shift all coordinates
    coords = transform_coordinates(coords, xlim, ylim)
    nuc_centroids = nuc_boundaries.groupby("cell_id")[["x", "y"]].mean().values
    nuc_centroids = transform_coordinates(nuc_centroids, xlim, ylim)
    nuc_boundaries[["x", "y"]] = transform_coordinates(
        nuc_boundaries[["x", "y"]].values, xlim, ylim
    )
    cell_boundaries[["x", "y"]] = transform_coordinates(
        cell_boundaries[["x", "y"]].values, xlim, ylim
    )
    transcripts[["x", "y"]] = transform_coordinates(
        transcripts[["x", "y"]].values, xlim, ylim
    )

    adata.uns["nuc_boundaries"] = nuc_boundaries
    adata.uns["cell_boundaries"] = cell_boundaries
    adata.uns["transcripts"] = transcripts
    adata.uns["image"] = image

    try:
        del adata.uns["nuc_patches"]
    except:
        pass
    try:
        del adata.uns["cell_patches"]
    except:
        pass

    return adata
