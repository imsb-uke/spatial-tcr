import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nichepca as npc
import numpy as np
from scipy.sparse import csr_matrix


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
