from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm


def _group_cell_indices(
    obs: pd.DataFrame, cell_type_col: str, slice_col: str
) -> dict[tuple[object, object], np.ndarray]:
    # pandas groupby.indices gives {group_key: integer positions}
    return obs.groupby([cell_type_col, slice_col], sort=False, observed=True).indices


def test_gene_codetection_permutation(
    adata,
    genes: Sequence[str],
    cell_type_col: str,
    slice_col: str,
    n_perm: int = 1000,
    seed: int | None = 0,
    min_cells_per_group: int = 40,
    alternative: str = "greater",
) -> pd.DataFrame:
    """
    Permutation test for gene co-detection across groups defined by (cell_type_col, slice_col).

    Input:
      - genes: list of gene names (NO explicit pair input; all unordered pairs are tested)

    Output (one row per unordered pair):
      - gene1, gene2
      - obs_codetections
      - sim_codetections (list length n_perm)
      - p_value (empirical)
      - p_value_adj (BH/FDR via statsmodels)
    """
    if alternative != "greater":
        raise ValueError("Only alternative='greater' is implemented.")

    genes = list(dict.fromkeys(genes))
    if len(genes) < 2:
        raise ValueError("Need at least two genes.")

    # Validate genes exist in adata
    var_names = np.asarray(adata.var_names)
    var_index = {g: i for i, g in enumerate(var_names)}
    missing = [g for g in genes if g not in var_index]
    if missing:
        raise ValueError(f"Genes not found in adata.var_names: {missing}")

    n_genes = len(genes)
    tri = np.triu_indices(n_genes, k=1)
    n_pairs = tri[0].size

    obs = np.zeros(n_pairs, dtype=np.int64)
    sim = np.zeros((n_pairs, n_perm), dtype=np.int64)

    rng = np.random.default_rng(seed)

    # Group cells by (cell_type, slice) using pandas groupby
    obs_df = adata.obs[[cell_type_col, slice_col]]
    group_to_cells = _group_cell_indices(obs_df, cell_type_col, slice_col)

    for _, cell_pos in tqdm(group_to_cells.items(), total=len(group_to_cells)):
        cell_type = obs_df.iloc[cell_pos[0]][cell_type_col]
        slice_value = obs_df.iloc[cell_pos[0]][slice_col]
        cell_idx = np.asarray(cell_pos, dtype=int)
        if cell_idx.size < min_cells_per_group:
            print(
                f"Skipping group {cell_type_col}={cell_type} ({slice_col}={slice_value}) with {cell_idx.size} cells (min {min_cells_per_group})"
            )
            continue

        X_subset = adata[cell_idx, genes].X
        if hasattr(X_subset, "toarray"):
            X_subset = X_subset.toarray()
        det = (X_subset > 0).astype(bool, copy=False)  # (n_cells, n_genes) bool
        det_u8 = det.astype(np.uint8, copy=False)

        # Observed co-detections for all pairs via matrix product
        obs_mat = det_u8.T @ det_u8  # (n_genes, n_genes) counts
        obs += np.asarray(obs_mat)[tri]

        # Permutations: independently permute each gene column within the group
        n_cells_group = det_u8.shape[0]
        detp = np.empty_like(det_u8)

        for k in range(n_perm):
            for j in range(n_genes):
                detp[:, j] = det_u8[rng.permutation(n_cells_group), j]

            sim_mat = detp.T @ detp
            sim[:, k] += np.asarray(sim_mat)[tri]

    E_sim = sim.mean(axis=1)
    std_sim = sim.std(axis=1)
    z_scores = (obs - E_sim) / std_sim

    # Empirical one-sided p-values (add-one smoothing)
    pvals = (1.0 + (sim >= obs[:, None]).sum(axis=1)) / (n_perm + 1.0)
    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    out = pd.DataFrame(
        {
            "gene1": [genes[i] for i in tri[0]],
            "gene2": [genes[j] for j in tri[1]],
            "obs_codetections": obs,
            "sim_codetections": [sim[p, :].tolist() for p in range(n_pairs)],
            "z_score": z_scores,
            "p_value": pvals,
            "p_value_adj": pvals_adj,
        }
    )
    return out
