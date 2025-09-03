import networkx as nx
import numpy as np


def to_networkx(adata):
    # transform the graph inside the adata object to a networkx graph
    edge_index = adata.uns["graph"]["edge_index"]
    g = nx.Graph()
    g.add_edges_from(edge_index.T)
    return g


def annotate_ccs(adata, g=None, return_ccs=False):
    if g is None:
        g = to_networkx(adata)
    ccs = list(nx.connected_components(g))
    # Pre-allocate a numpy array instead of initializing with 'NA'
    cc_labels = np.full(len(adata.obs), -1, dtype=int)

    # Vectorized assignment instead of loop
    for i, cc in enumerate(ccs):
        cc_labels[list(cc)] = i

    # Single assignment to dataframe
    adata.obs["cc"] = cc_labels.astype(str)
    adata.obs.loc[cc_labels == -1, "cc"] = "NA"
    if return_ccs:
        return ccs


def merge_labels(adata, labels, label_key="cc"):
    mask = adata.obs[label_key].isin(labels)
    adata.obs.loc[mask, label_key] = labels[0]
