import numpy as np
import pandas as pd


def permute_positions(adata, sample_key=None, rng=None, seed=42):
    if rng is None:
        rng = np.random.RandomState(seed=seed)

    coords = adata.obsm["spatial"].copy()
    if sample_key is not None:
        sample_codes = pd.Categorical(adata.obs[sample_key]).codes
        combined = rng.random(adata.shape[0]) + sample_codes
        perm_order = np.argsort(combined)
        group_sort = np.argsort(sample_codes)
        final_perm = np.empty_like(perm_order)
        final_perm[group_sort] = perm_order
    else:
        final_perm = rng.permutation(coords.shape[0])

    adata.obsm["spatial"] = coords[final_perm]
    return adata
