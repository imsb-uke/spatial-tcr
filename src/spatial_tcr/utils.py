import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nichepca as npc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import csr_matrix, hstack


def remove_background(
    adata, greater=0, genes=None, nonzero_percentile=None, layer_in=None, layer_out=None
):
    genes = genes if genes else adata.var_names
    # convert genes to indices
    gene_indices = [adata.var_names.get_loc(g) for g in genes]

    if layer_in is not None:
        X_init = adata[:, genes].layers[layer_in].copy()
    else:
        X_init = adata[:, genes].X.copy()

    X = npc.utils.to_numpy(X_init)

    if greater:
        mask = X > greater
        X = X * mask
    elif nonzero_percentile:
        for k, _g in enumerate(genes):
            x = X[:, k]
            nonzero_mask = x > 0
            if np.any(nonzero_mask):
                threshold = np.percentile(x[nonzero_mask], nonzero_percentile)
                mask = x > threshold
                X[:, k] = x * mask
    else:
        raise ValueError("Either greater or nonzero_percentile must be provided.")

    X = csr_matrix(X)

    if layer_out is not None:
        # provide the base
        adata.layers[layer_out] = (
            adata.layers[layer_in].copy() if layer_in else adata.X.copy()
        )
        # fill the base with the modified values
        adata.layers[layer_out][:, gene_indices] = X
    else:
        adata.X[:, gene_indices] = X


def create_palette(categories, base_cmap="tab20"):
    # Use the tab20 colormap
    colormap = plt.cm.get_cmap(
        base_cmap, len(categories)
    )  # Ensure enough colors for all categories
    # Create a dictionary mapping each category to a color
    color_dict = {
        category: mcolors.to_hex(colormap(i)) for i, category in enumerate(categories)
    }
    return color_dict


def switch_cwd_to_root():
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "pyproject.toml").is_file():
            os.chdir(d)
            print(f"Changed working directory to {os.getcwd()}")
            return
    raise FileNotFoundError("pyproject.toml")


def setup_plotting(figure_dir=None, display_dpi=None, save_dpi=300):
    import os

    import scanpy as sc
    import seaborn as sns
    from matplotlib import font_manager, rcParams

    sns.set_theme(style="ticks", context="paper")
    sc.settings._vector_friendly = True

    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    arial = font_manager.FontProperties(fname=font_path).get_name()
    if arial == "Arial":
        print("Arial font found!")
        rcParams["font.family"] = arial
    else:
        print(f"Arial font not found, instead found: {arial}! Using default font.")

    if figure_dir is not None:
        os.makedirs(figure_dir, exist_ok=True)
    if display_dpi is not None:
        rcParams["figure.dpi"] = display_dpi
    if save_dpi is not None:
        rcParams["savefig.dpi"] = save_dpi


def _create_mapping_matrix(mapping_dict, var_names):
    target_genes = list(mapping_dict.keys())
    rows = []
    cols = []
    data = []
    for col_idx, target in enumerate(target_genes):
        source_genes = mapping_dict[target]
        for source in source_genes:
            if source in var_names:
                row_idx = var_names.get_loc(source)
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(1)
    mapping_matrix = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(var_names), len(target_genes))
    )
    return mapping_matrix, target_genes


def aggregate_gene_expression(adata, agg_dict, layer_in="counts"):
    if layer_in is not None:
        adata = adata.copy()
        adata.X = adata.layers[layer_in]
    source_genes = []
    for _k, v in agg_dict.items():
        source_genes.extend(v)
    map_mat, target_genes = _create_mapping_matrix(agg_dict, adata.var_names)
    X_agg = adata.X @ map_mat
    genes_to_keep = [g for g in adata.var_names if g not in source_genes]
    X_reduced = adata[:, genes_to_keep].X.copy()
    X_merged = hstack([X_reduced, X_agg])
    var_reduced = adata[:, genes_to_keep].var.copy()
    var_agg = pd.DataFrame(index=target_genes, columns=var_reduced.columns)
    for col in var_reduced.columns:
        if pd.api.types.is_numeric_dtype(var_reduced[col]):
            var_agg[col] = 0
        else:
            var_agg[col] = ""
    var_merged = pd.concat([var_reduced, var_agg], axis=0)
    var_merged.index = var_merged.index.astype(str)
    return sc.AnnData(
        X=X_merged,
        var=var_merged,
        obs=adata.obs,
        obsm=adata.obsm,
        uns=adata.uns,
    )
