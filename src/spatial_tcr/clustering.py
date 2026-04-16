import numpy as np
import scanpy as sc


def leiden_unique(
    adata,
    use_rep=None,
    resolution=1.0,
    n_neighbors=15,
    key_added="leiden",
    flavor="igraph",
    n_iterations=2,
    **kwargs,
):
    X_rep = adata.obsm[use_rep]
    _, unique_indices, inverse_indices = np.unique(
        X_rep, axis=0, return_index=True, return_inverse=True
    )
    print(f"Found {len(unique_indices)} unique embeddings from a total of {len(X_rep)}")

    ad_sub = adata[unique_indices].copy()
    sc.pp.neighbors(ad_sub, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.leiden(
        ad_sub,
        resolution=resolution,
        flavor=flavor,
        n_iterations=n_iterations,
        key_added=key_added,
        **kwargs,
    )
    labels = ad_sub.obs[key_added].iloc[inverse_indices].copy()
    adata.obs[key_added] = labels.values
