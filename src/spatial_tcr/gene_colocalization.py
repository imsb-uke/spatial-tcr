import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .plotting import _p_to_stars


def _sum_over_genes(X) -> np.ndarray:
    """Row-wise sum; works for dense or sparse."""
    s = X.sum(axis=1)
    return np.asarray(s).ravel()


def _get_gene_vector(adata, gene: str) -> np.ndarray:
    """Returns 1D dense vector for a single gene (works with sparse .X)."""
    x = adata[:, [gene]].X
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).ravel()


def add_combined_detection(
    adata, genes: list[str], obs_key: str, thresh: float = 0.0
) -> None:
    """Adds obs_key as int {0,1}: any gene in genes is > thresh."""
    if len(genes) == 0:
        raise ValueError(f"{obs_key}: gene list is empty.")
    X = adata[:, genes].X
    adata.obs[obs_key] = (_sum_over_genes(X) > thresh).astype(int)


def codetection_by_group(
    adata,
    group_key: str,
    combined_key: str,
    target_genes: list[str],
    *,
    thresh: float = 0.0,
    min_cells: int = 25,
) -> pd.DataFrame:
    """
    For each group (cell type) and each target gene:
      - trav_rate, target_rate, codetect_rate
      - odds_ratio + Fisher exact p-value for 2x2 table
    """
    if combined_key not in adata.obs:
        raise ValueError(f"{combined_key} not found in adata.obs. Create it first.")

    rows = []
    for ct, idx in adata.obs.groupby(group_key).indices.items():
        a = adata[idx]
        n = a.n_obs
        if n < min_cells:
            continue

        trav = a.obs[combined_key].to_numpy(dtype=bool)
        trav_rate = float(trav.mean())

        for g in target_genes:
            x = _get_gene_vector(a, g)
            tgt = x > thresh
            tgt_rate = float(tgt.mean())

            both = trav & tgt
            codetect_rate = float(both.mean())

            # 2x2: [[both, trav_only],[tgt_only, neither]]
            a11 = int(np.sum(both))
            a10 = int(np.sum(trav & ~tgt))
            a01 = int(np.sum(~trav & tgt))
            a00 = int(np.sum(~trav & ~tgt))
            table = np.array([[a11, a10], [a01, a00]], dtype=int)

            # Fisher exact can be unstable if a row/col is all zeros; handle gracefully
            try:
                odds_ratio, p = stats.fisher_exact(table)
            except Exception:
                odds_ratio, p = np.nan, np.nan

            rows.append(
                {
                    group_key: ct,
                    "gene": g,
                    "n_cells": n,
                    "trav_rate": trav_rate,
                    "target_rate": tgt_rate,
                    "codetect_rate": codetect_rate,
                    "odds_ratio": float(odds_ratio)
                    if np.isfinite(odds_ratio)
                    else np.nan,
                    "p_fisher": float(p) if np.isfinite(p) else np.nan,
                    "a11_both": a11,
                    "a10_trav_only": a10,
                    "a01_tgt_only": a01,
                    "a00_neither": a00,
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # FDR across all (cell_type, gene) tests with valid p-values
    mask = out["p_fisher"].notna()
    q = np.full(len(out), np.nan, dtype=float)
    if mask.any():
        q[mask] = multipletests(out.loc[mask, "p_fisher"].to_numpy(), method="fdr_bh")[
            1
        ]
    out["q_fdr"] = q
    return out


def plot_codetect_heatmap(
    df: pd.DataFrame,
    group_key: str,
    *,
    value_col: str = "codetect_rate",
    sort_genes_by: str | None = "mean",
    figsize_scale: float = 0.45,
) -> None:
    """
    Heatmap of value_col (genes x groups).
    sort_genes_by: "mean" sorts genes by mean value across groups; None keeps original order.
    """
    mat = df.pivot(index="gene", columns=group_key, values=value_col)

    if sort_genes_by == "mean":
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]

    genes = mat.index.to_list()
    groups = mat.columns.to_list()
    Z = mat.to_numpy()

    fig_w = max(8, len(groups) * figsize_scale)
    fig_h = max(3, len(genes) * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(Z, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)

    ax.set_title(f"{value_col} (TRAV combined × target gene)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(value_col)

    plt.tight_layout()
    plt.show()


def plot_codetect_heatmap_with_stars(
    df: pd.DataFrame,
    group_key: str,
    *,
    value_col: str = "codetect_rate",
    p_col: str = "q_fdr",  # "p_fisher" or "q_fdr"
    show_ns: bool = False,
    min_cells: int | None = None,  # optional filter on n_cells
    sort_genes_by: str | None = "mean",
    figsize_scale: float = 0.45,
) -> None:
    d = df.copy()
    if min_cells is not None and "n_cells" in d.columns:
        d = d[d["n_cells"] >= min_cells]

    mat = d.pivot(index="gene", columns=group_key, values=value_col)
    pmat = d.pivot(index="gene", columns=group_key, values=p_col)

    # align
    mat = mat.sort_index()
    pmat = pmat.reindex(index=mat.index, columns=mat.columns)

    if sort_genes_by == "mean":
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
        pmat = pmat.reindex(index=mat.index)

    genes = mat.index.to_list()
    groups = mat.columns.to_list()
    Z = mat.to_numpy()
    P = pmat.to_numpy()

    fig_w = max(8, len(groups) * figsize_scale)
    fig_h = max(3, len(genes) * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(Z, aspect="auto", interpolation="nearest", cmap="Blues")

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)

    ax.set_title(f"{value_col} with significance ({p_col})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(value_col)

    # overlay stars
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            p = P[i, j]
            if not np.isfinite(p):
                continue
            s = _p_to_stars(float(p))
            if (not show_ns) and s == "ns":
                continue
            ax.text(j, i, s, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def guess_depth_key(adata) -> str:
    candidates = [
        "total_counts",
        "n_counts",
        "nCount_RNA",
        "nUMI",
        "umi_counts",
        "counts",
    ]
    for k in candidates:
        if k in adata.obs.columns:
            return k
    raise KeyError(
        "Could not find a depth column in adata.obs. "
        "Common keys: total_counts, n_counts, nCount_RNA, nUMI. "
        "If missing, run sc.pp.calculate_qc_metrics(adata, inplace=True) "
        "or pass the correct depth_key explicitly."
    )


def codetect_with_logreg_pvals(
    adata,
    *,
    celltype_key: str,
    trav_key: str | None = None,
    trbv_key: str | None = None,
    target_genes: list[str],
    depth_key: str | None = None,
    thresh: float = 0.0,
    min_cells: int = 25,
) -> pd.DataFrame:
    """
    Per cell type and target gene:
      - codetect_rate = mean(marker_detected & gene_detected)
      - logistic regression p-value for marker term, adjusted for depth
      - BH-FDR across all tests
    """
    if depth_key is None:
        depth_key = guess_depth_key(adata)
    if (trav_key is None) == (trbv_key is None):
        raise ValueError("Provide exactly one of trav_key or trbv_key.")
    marker_key = trav_key if trav_key is not None else trbv_key
    marker_name = "TRAV" if trav_key is not None else "TRBV"

    rows = []
    for ct, idx in adata.obs.groupby(celltype_key).indices.items():
        a = adata[idx]
        if a.n_obs < min_cells:
            continue

        trav = a.obs[marker_key].to_numpy(dtype=int)
        depth = np.log1p(a.obs[depth_key].to_numpy())

        # skip degenerate marker (all 0 or all 1)
        if trav.mean() in (0.0, 1.0):
            continue

        X = sm.add_constant(np.column_stack([trav, depth]), has_constant="add")

        for g in target_genes:
            y = (_get_gene_vector(a, g) > thresh).astype(int)

            # codetection summary
            both = (trav == 1) & (y == 1)
            codetect_rate = float(both.mean())
            trav_rate = float(trav.mean())
            tgt_rate = float(y.mean())

            # skip degenerate y
            if tgt_rate in (0.0, 1.0):
                rows.append(
                    {
                        celltype_key: ct,
                        "marker_key": marker_key,
                        "marker_name": marker_name,
                        "gene": g,
                        "n_cells": int(a.n_obs),
                        "trav_rate": trav_rate,
                        "target_rate": tgt_rate,
                        "codetect_rate": codetect_rate,
                        "or_trav_adj": np.nan,
                        "p_lr": np.nan,
                    }
                )
                continue

            # logistic regression (Binomial GLM is often more stable than Logit)
            try:
                m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
                coef = float(m.params[1])  # TRAV term
                p = float(m.pvalues[1])
                or_adj = float(np.exp(coef))
            except Exception:
                p = np.nan
                or_adj = np.nan

            rows.append(
                {
                    celltype_key: ct,
                    "marker_key": marker_key,
                    "marker_name": marker_name,
                    "gene": g,
                    "n_cells": int(a.n_obs),
                    "trav_rate": trav_rate,
                    "target_rate": tgt_rate,
                    "codetect_rate": codetect_rate,
                    "or_trav_adj": or_adj,
                    "p_lr": p,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    mask = df["p_lr"].notna()
    q = np.full(len(df), np.nan, dtype=float)
    if mask.any():
        q[mask] = multipletests(df.loc[mask, "p_lr"].to_numpy(), method="fdr_bh")[1]
    df["q_lr"] = q
    return df


def plot_codetect_heatmap_with_lr_stars(
    df: pd.DataFrame,
    *,
    celltype_key: str,
    value_col: str = "codetect_rate",
    p_col: str = "q_lr",  # or "p_lr"
    show_ns: bool = False,
    sort_genes_by_mean: bool = True,
    figsize_scale: float = 0.45,
) -> None:
    mat = df.pivot(index="gene", columns=celltype_key, values=value_col)
    pmat = df.pivot(index="gene", columns=celltype_key, values=p_col)

    if sort_genes_by_mean:
        order = mat.mean(axis=1).sort_values(ascending=False).index
        mat = mat.loc[order]
        pmat = pmat.reindex(index=order, columns=mat.columns)

    genes = mat.index.to_list()
    cts = mat.columns.to_list()
    Z = mat.to_numpy()
    P = pmat.to_numpy()

    fig_w = max(8, len(cts) * figsize_scale)
    fig_h = max(3, len(genes) * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(Z, aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(len(cts)))
    ax.set_xticklabels(cts, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)

    ax.set_title(f"{value_col} with LR significance ({p_col})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(value_col)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            p = P[i, j]
            if not np.isfinite(p):
                continue
            s = _p_to_stars(float(p))
            if (not show_ns) and s == "ns":
                continue
            ax.text(j, i, s, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def add_conditional_codetect_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # denominators
    denom_trav = d["a11_both"] + d["a10_trav_only"]  # TRAV=1
    denom_tgt = d["a11_both"] + d["a01_tgt_only"]  # target=1

    d["p_target_given_trav"] = np.where(
        denom_trav > 0, d["a11_both"] / denom_trav, np.nan
    )
    d["p_trav_given_target"] = np.where(
        denom_tgt > 0, d["a11_both"] / denom_tgt, np.nan
    )

    return d


def plot_heatmap_with_optional_stars(
    df: pd.DataFrame,
    group_key: str,
    *,
    value_col: str,
    p_col: str = "q_fdr",  # or "p_fisher"
    show_stars: bool = True,
    show_ns: bool = False,
    sort_genes_by: str | None = "mean",
    figsize_scale: float = 0.45,
) -> None:
    mat = df.pivot(index="gene", columns=group_key, values=value_col)
    pmat = df.pivot(index="gene", columns=group_key, values=p_col)

    if sort_genes_by == "mean":
        order = mat.mean(axis=1).sort_values(ascending=False).index
        mat = mat.loc[order]
        pmat = pmat.reindex(index=order)

    genes = mat.index.to_list()
    groups = mat.columns.to_list()
    Z = mat.to_numpy()
    P = pmat.to_numpy()

    fig_w = max(8, len(groups) * figsize_scale)
    fig_h = max(3, len(genes) * figsize_scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(Z, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_ylabel("Target Gene")
    ax.set_xlabel("Cell Type")

    ax.set_title(value_col)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(value_col)

    if show_stars:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                p = P[i, j]
                if not np.isfinite(p):
                    continue
                s = _p_to_stars(float(p))
                if (not show_ns) and s == "ns":
                    continue
                ax.text(j, i, s, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()
