import nichepca as npc
import numpy as np
import pandas as pd
import scanpy as sc


def aggregate_trv_expression(adata):
    av_genes, bv_genes, dv_genes, gv_genes, tv_genes = get_tcr_genes(adata)

    trav_expr = npc.utils.to_numpy(adata[:, av_genes].X).sum(axis=1)
    trbv_expr = npc.utils.to_numpy(adata[:, bv_genes].X).sum(axis=1)
    trdv_expr = npc.utils.to_numpy(adata[:, dv_genes].X).sum(axis=1)
    trgv_expr = npc.utils.to_numpy(adata[:, gv_genes].X).sum(axis=1)

    X_tcr = np.stack([trav_expr, trbv_expr, trdv_expr, trgv_expr], axis=1)

    ad_tmp = sc.AnnData(
        X=X_tcr,
        obs=adata.obs,
        var=pd.DataFrame(index=["TRAV", "TRBV", "TRDV", "TRGV"]),
    )
    return ad_tmp


def get_tcr_genes(adata):
    av_genes = [g for g in adata.var_names if g.startswith("TRAV")]
    bv_genes = [g for g in adata.var_names if g.startswith("TRBV")]
    dv_genes = [g for g in adata.var_names if g.startswith("TRDV")]
    gv_genes = [g for g in adata.var_names if g.startswith("TRGV")]
    tv_genes = av_genes + bv_genes + dv_genes + gv_genes
    print(
        f"Found {len(av_genes)} TRAV genes, {len(bv_genes)} TRBV genes, {len(dv_genes)} TRDV genes, {len(gv_genes)} TRGV genes"
    )
    return av_genes, bv_genes, dv_genes, gv_genes, tv_genes


bv_map = {
    "TRBV10": ["TRBV10-1", "TRBV10-2", "TRBV10-3"],
    "TRBV11": ["TRBV11-1", "TRBV11-2"],
    "TRBV12": ["TRBV12-3", "TRBV12-4", "TRBV12-5"],
    "TRBV18_19": ["TRBV18", "TRBV19"],
    "TRBV4": ["TRBV4-1", "TRBV4-2"],
    "TRBV5-6": ["TRBV5-3", "TRBV5-5", "TRBV5-6", "TRBV5-7"],
    "TRBV6": ["TRBV6-1", "TRBV6-2", "TRBV6-5", "TRBV6-7"],
    "TRBV7-2_3": ["TRBV7-2", "TRBV7-3"],
}

av_map = {
    "TRAV19_21": ["TRAV19", "TRAV21"],
    "TRAV12": ["TRAV12-1", "TRAV12-2", "TRAV12-3"],
    "TRAV8-4": ["TRAV8-2", "TRAV8-3", "TRAV8-4", "TRAV8-5", "TRAV8-6"],
    "TRAV9": ["TRAV9-1", "TRAV9-2"],
}

avbv_map = {**av_map, **bv_map}
