import json
import os

import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tqdm.auto import tqdm

pixel_sizes = {
    0: 0.2125,
    1: 0.425,
    2: 0.85,
    3: 1.7,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def order_df(df, order, col="cell_id"):
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


def load_additional_data(data_dir):
    # load additional data
    transcripts_path = os.path.join(data_dir, "transcripts.parquet")
    gene_panel_path = os.path.join(data_dir, "gene_panel.json")
    cell_boundaries_path = os.path.join(data_dir, "cell_boundaries.parquet")
    nuc_boundaries_path = os.path.join(data_dir, "nucleus_boundaries.parquet")

    transcripts = pd.read_parquet(transcripts_path)
    gene_panel = load_json(gene_panel_path)
    cell_boundaries = pd.read_parquet(cell_boundaries_path)
    nuc_boundaries = pd.read_parquet(nuc_boundaries_path)

    return transcripts, gene_panel, cell_boundaries, nuc_boundaries


def load_spatial_data_xenium(
    data_dir, cell_ids, cell_coords, level=0, legacy_format=False
):
    _pixel_size = pixel_sizes[level]
    # TODO: optimize this with dask and read only parts of the image
    pass


def add_spatial_data(
    adata,
    data_dir,
    sample_key=None,
    sample=None,
    xlim=None,
    ylim=None,
    img_buffer=0,
    level=0,
    mask_dapi=False,
    legacy_format=False,
    cell_id_key=None,
    load_dapi=True,
    load_stainings=True,
    nuc_radius=2,
    cell_radius=4,
    fit_to_cells=True,
):
    # load image
    pixel_size = pixel_sizes[level]
    if legacy_format:
        img_path = os.path.join(data_dir, "morphology_focus.ome.tif")
        print(f"Loading image from {img_path}")
        img = tifffile.imread(img_path, is_ome=False, level=level)[:, :, None]
        channel_names = ["DAPI"]
    else:
        base_paths = [
            os.path.join(
                data_dir, "morphology_focus", f"morphology_focus_{i:04d}.ome.tif"
            )
            for i in range(4)
        ]
        img_paths = []
        if load_dapi:
            img_paths.extend(base_paths[0:1])
        if load_stainings:
            img_paths.extend(base_paths[1:])

        print(f"Loading images from {img_paths}")
        images = [
            tifffile.imread(img_path, is_ome=False, level=level)
            for img_path in tqdm(img_paths)
        ]
        img = np.stack(images, axis=-1)
        channel_names = ["DAPI", "CD45", "RNA", "Vimentin"]

    print(f"The image has shape: {img.shape} with channel names: {channel_names}")

    transcripts, gene_panel, cell_boundaries, nuc_boundaries = load_additional_data(
        data_dir
    )

    # restrict to a specific sample
    if sample_key is not None and sample is not None:
        mask = adata.obs[sample_key] == sample
        sub_ad = adata[mask].copy()
    else:
        sub_ad = adata

    if xlim is not None:
        mask = (sub_ad.obsm["spatial_translated"][:, 0] >= xlim[0]) & (
            sub_ad.obsm["spatial_translated"][:, 0] <= xlim[1]
        )
        sub_ad = sub_ad[mask].copy()
    if ylim is not None:
        mask = (sub_ad.obsm["spatial_translated"][:, 1] >= ylim[0]) & (
            sub_ad.obsm["spatial_translated"][:, 1] <= ylim[1]
        )
        sub_ad = sub_ad[mask].copy()

    cell_ids_sub = (
        sub_ad.obs_names.tolist()
        if cell_id_key is None
        else sub_ad.obs[cell_id_key].tolist()
    )
    print(f"cell_ids_sub: {cell_ids_sub}")

    sub_nuc_boundaries = nuc_boundaries[
        nuc_boundaries["cell_id"].isin(cell_ids_sub)
    ].copy()
    sub_cell_boundaries = cell_boundaries[
        cell_boundaries["cell_id"].isin(cell_ids_sub)
    ].copy()

    print(f"sub_nuc_boundaries.shape: {sub_nuc_boundaries.shape}")
    print(f"sub_cell_boundaries.shape: {sub_cell_boundaries.shape}")

    if sub_cell_boundaries.shape[0] == 0:
        print("No cell boundaries found, using circle approximation")
        sub_cell_boundaries = vectorized_circle_polygons(sub_ad, radius=cell_radius)
    if sub_nuc_boundaries.shape[0] == 0:
        print("No nucleus boundaries found, using circle approximation")
        sub_nuc_boundaries = vectorized_circle_polygons(sub_ad, radius=nuc_radius)

    # filter transcripts
    names = [t["type"]["data"]["name"] for t in gene_panel["payload"]["targets"]]
    descriptors = [t["type"]["descriptor"] for t in gene_panel["payload"]["targets"]]
    genes = [name for name, desc in zip(names, descriptors) if desc == "gene"]
    transcripts = transcripts[transcripts["feature_name"].isin(genes)]
    transcripts = transcripts[transcripts["qv"] > 20].copy()

    # filter the data in space
    sub_coords = sub_ad.obsm["spatial"]

    if not fit_to_cells and xlim is not None and ylim is not None:
        xlim_0, ylim_0 = (sub_ad.obsm["spatial"] - sub_ad.obsm["spatial_translated"])[0]
        x_min, x_max = np.array(xlim) + xlim_0
        y_min, y_max = np.array(ylim) + ylim_0
    else:
        hull = ConvexHull(sub_coords)
        # Create a Polygon from the hull vertices
        hull_polygon = Polygon(sub_coords[hull.vertices])
        if img_buffer != 0:
            buffered_polygon = hull_polygon.buffer(
                img_buffer
            )  # Adjust the distance here
        else:
            buffered_polygon = hull_polygon

            x_min, y_min, x_max, y_max = buffered_polygon.bounds

    # calc new limits
    xlim = np.array([x_min, x_max])
    ylim = np.array([y_min, y_max])

    # translate into pixel coordinates
    xlim_pixel = np.round(xlim / pixel_size).astype(int)
    xlim = xlim_pixel * pixel_size
    ylim_pixel = np.round(ylim / pixel_size).astype(int)
    ylim = ylim_pixel * pixel_size
    print(f"xlim: {xlim}, ylim: {ylim}")

    sub_img = img[ylim_pixel[0] : ylim_pixel[1], xlim_pixel[0] : xlim_pixel[1]].copy()

    print(f"sub_img.shape: {sub_img.shape}")

    if mask_dapi:
        raise NotImplementedError("DAPI masking not implemented")
        # change coordinates of the polygon to be relative to the new origin
        _translated_polygon = Polygon(
            [
                # ((x - xlim[0]) * 1.0, (y - ylim[0]) * 1.0)
                ((x - xlim[0]), (y - ylim[0]))
                for x, y in buffered_polygon.exterior.coords
            ]
        )
    sub_coords_translated = sub_coords - np.array([xlim[0], ylim[0]])
    sub_coords_scaled = sub_coords_translated / pixel_size

    # handle cell and nucleus boundaries
    sub_nuc_boundaries["x"] = (sub_nuc_boundaries["vertex_x"] - xlim[0]) / pixel_size
    sub_nuc_boundaries["y"] = (sub_nuc_boundaries["vertex_y"] - ylim[0]) / pixel_size
    sub_nuc_boundaries = order_df(sub_nuc_boundaries, cell_ids_sub)

    sub_cell_boundaries["x"] = (sub_cell_boundaries["vertex_x"] - xlim[0]) / pixel_size
    sub_cell_boundaries["y"] = (sub_cell_boundaries["vertex_y"] - ylim[0]) / pixel_size
    sub_cell_boundaries = order_df(sub_cell_boundaries, cell_ids_sub)

    # filter transcripts data
    mask = transcripts["x_location"].between(xlim[0], xlim[1]) & transcripts[
        "y_location"
    ].between(ylim[0], ylim[1])
    sub_transcripts = transcripts[mask].copy()
    sub_transcripts["x"] = (sub_transcripts["x_location"] - xlim[0]) / pixel_size
    sub_transcripts["y"] = (sub_transcripts["y_location"] - ylim[0]) / pixel_size

    print(f"sub_transcripts.shape: {sub_transcripts.shape}")

    # add everything to the anndata object
    sub_ad.obsm["spatial_translated"] = sub_coords_translated
    sub_ad.obsm["spatial_scaled"] = sub_coords_scaled

    sub_ad.uns["sdata"] = {}
    sub_ad.uns["sdata"]["nuc_boundaries"] = sub_nuc_boundaries
    sub_ad.uns["sdata"]["cell_boundaries"] = sub_cell_boundaries
    sub_ad.uns["sdata"]["transcripts"] = sub_transcripts
    sub_ad.uns["sdata"]["image"] = xr.DataArray(
        sub_img, dims=["x", "y", "channel"], coords={"channel": channel_names}
    )
    sub_ad.uns["sdata"]["pixel_size"] = pixel_size

    # save data in scanpy compatible format
    sub_ad.uns["spatial"] = {}
    sub_ad.uns["spatial"]["sample"] = {}
    sub_ad.uns["spatial"]["sample"]["images"] = {}
    sub_ad.uns["spatial"]["sample"]["images"]["hires"] = (
        sub_img if len(sub_img.shape) == 2 else sub_img[:, :, 0]
    )
    sub_ad.uns["spatial"]["sample"]["scalefactors"] = {}
    sub_ad.uns["spatial"]["sample"]["scalefactors"]["tissue_hires_scalef"] = 1

    return sub_ad


def update_spatial_data(adata):
    # TODO: update the spatial data in the anndata object to fit the cell indices
    # important when I subset the data.
    pass


def vectorized_circle_polygons(adata, radius=3, n_vertices=20):
    coords = adata.obsm["spatial"]
    cell_ids = adata.obs_names.tolist()

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    vertex_x = coords[:, 0][:, None] + radius * np.cos(angles)
    vertex_y = coords[:, 1][:, None] + radius * np.sin(angles)
    print(vertex_x.shape, vertex_y.shape)

    # Repeat cell_ids for each vertex
    cell_ids_repeated = np.repeat(cell_ids, n_vertices)
    print(cell_ids_repeated.shape)

    df = pd.DataFrame(
        {
            "cell_id": cell_ids_repeated,
            "vertex_x": vertex_x.ravel(),
            "vertex_y": vertex_y.ravel(),
        }
    )
    return df


def plot_palette_dict(palette_dict, title=None, fontsize=10):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    _labels = list(palette_dict.keys())
    colors = list(palette_dict.values())
    n = len(colors)
    fig, ax = plt.subplots(figsize=(max(6, n), 1.0))
    for i, (label, color) in enumerate(palette_dict.items()):
        # Validate color
        try:
            mcolors.to_rgba(color)
        except ValueError:
            color = "#CCCCCC"  # fallback color for invalid
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(
            i + 0.5,
            1.01,
            label,
            va="bottom",
            ha="center",
            fontsize=fontsize,
            color="black",
            rotation=0,
        )
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        plt.title(title)
    plt.show()
