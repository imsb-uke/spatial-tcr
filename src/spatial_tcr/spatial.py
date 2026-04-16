import alphashape
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm.auto import tqdm


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


def filter_ccs(adata, cc_key="cc", min_cells=10):
    mask = adata.obs[cc_key].value_counts() >= min_cells
    retained_ccs = mask[mask].index.tolist()
    return adata[adata.obs[cc_key].isin(retained_ccs)].copy()


def annotate_border_band(
    adata, obs_key, obsm_key="spatial", offset=20, alpha=0.0, min_cells=3
):
    """Annotate cells in expanded border bands around area annotations."""
    coords = adata.obsm[obsm_key]
    labels = adata.obs[obs_key].values

    unique_labels = pd.Series(labels).dropna().unique()
    unique_labels = unique_labels[unique_labels != "NA"]

    shapes = {}
    for label in unique_labels:
        mask = labels == label
        if mask.sum() < min_cells:
            print(f"Skipping label {label} because it has less than 3 cells.")
            continue
        points = coords[mask]
        alpha_shape = alphashape.alphashape(points, alpha=alpha)
        if alpha_shape.is_empty or not hasattr(alpha_shape, "geom_type"):
            print(f"Skipping label {label} because it has no shape.")
            continue
        shapes[label] = alpha_shape

    if not shapes:
        print("No shapes found.")
        return

    non_overlapping_shapes = {}
    processed_union = None
    for label, shape in shapes.items():
        if processed_union is not None:
            shape = shape.difference(processed_union)
        if not shape.is_empty:
            non_overlapping_shapes[label] = shape
            if processed_union is None:
                processed_union = shape
            else:
                processed_union = processed_union.union(shape)

    expanded_shapes = {}
    for label, shape in non_overlapping_shapes.items():
        expanded = shape.buffer(offset)
        expanded_shapes[label] = expanded

    expanded_union = None
    final_expanded_shapes = {}
    for label, shape in expanded_shapes.items():
        if expanded_union is not None:
            shape = shape.difference(expanded_union)
        if not shape.is_empty:
            final_expanded_shapes[label] = shape
            if expanded_union is None:
                expanded_union = shape
            else:
                expanded_union = expanded_union.union(shape)

    new_col = f"{obs_key}_expanded"
    adata.obs[new_col] = "NA"

    points = [Point(coord) for coord in coords]
    for label, shape in final_expanded_shapes.items():
        mask = np.array([shape.contains(p) for p in points])
        adata.obs.loc[mask, new_col] = label


def spatial_expansion(
    adata,
    obs_key: str,
    out_key: str | None = None,
    expansion: float = 25.0,
    sample_key: str | None = None,
    spatial_key: str = "spatial_split",
):
    labels = adata.obs[obs_key]
    if out_key is None:
        out_key = f"{obs_key}_expanded"
    adata.obs[out_key] = labels.copy()

    best_dist = np.full(adata.n_obs, np.inf, dtype=float)

    elements = labels.dropna().unique()
    elements = elements[elements != "NA"]

    for element in tqdm(elements):
        ad_element = adata[labels == element]
        if ad_element.n_obs == 0:
            continue

        if sample_key is not None:
            samples = ad_element.obs[sample_key].unique()
            ad_sample = adata[adata.obs[sample_key].isin(samples)]
        else:
            ad_sample = adata

        coords_element = np.asarray(ad_element.obsm[spatial_key])
        coords_sample = np.asarray(ad_sample.obsm[spatial_key])

        tree_element = scipy.spatial.cKDTree(coords_element)

        # For each sample point: distance to nearest element point, capped at expansion
        dist, _ = tree_element.query(coords_sample, k=1, distance_upper_bound=expansion)

        hit = np.isfinite(dist)
        if not np.any(hit):
            continue

        # Map sample indices back to global indices
        sample_idx_global = adata.obs_names.get_indexer(ad_sample.obs_names)
        sample_idx_global = sample_idx_global[hit]
        dist = dist[hit]

        # Only update if this element is closer than the current best
        better = dist < best_dist[sample_idx_global]
        if np.any(better):
            best_dist[sample_idx_global[better]] = dist[better]
            adata.obs.iloc[
                sample_idx_global[better], adata.obs.columns.get_loc(out_key)
            ] = element


def _remove_holes(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(g.exterior) for g in geom.geoms if not g.is_empty])
    return geom  # fallback (GeometryCollection etc.)


def _covers_mask(geom, coords: np.ndarray) -> np.ndarray:
    # Fast path: Shapely 2.x vectorized predicates
    try:
        import shapely

        pts = shapely.points(coords[:, 0], coords[:, 1])
        return shapely.covers(geom, pts)
    except Exception:
        # Fallback: prepared geometry + Python loop
        g = prep(geom)
        return np.fromiter(
            (g.covers(Point(x, y)) for x, y in coords), dtype=bool, count=len(coords)
        )


def fill_annotations(
    adata,
    obs_key: str,
    out_key: str | None = None,
    sample_key: str | None = None,
    spatial_key: str = "spatial",
    alpha: float = 0.0,
    fill_holes: bool = True,
):
    labels = adata.obs[obs_key]
    if out_key is None:
        out_key = f"{obs_key}_filled"
    adata.obs[out_key] = labels.copy()

    elements = pd.Series(labels).dropna().unique()
    elements = [e for e in elements if e != "NA"]

    for element in tqdm(elements, desc="Filling annotations"):
        ad_element = adata[labels == element]
        if ad_element.n_obs == 0:
            print(f"Skipping element {element} because it has no cells.")
            continue

        if sample_key is not None:
            samples = ad_element.obs[sample_key].unique()
            ad_sample = adata[adata.obs[sample_key].isin(samples)]
        else:
            ad_sample = adata

        element_coords = np.asarray(ad_element.obsm[spatial_key])
        sample_coords = np.asarray(ad_sample.obsm[spatial_key])

        geom = alphashape.alphashape(element_coords, alpha=alpha)
        if getattr(geom, "is_empty", True):
            print(f"Skipping element {element} because it has no shape.")
            continue

        # Normalize weird outputs (e.g. GeometryCollection) to a usable surface geometry
        geom = unary_union(geom)

        if fill_holes:
            geom = _remove_holes(geom)

        inside = _covers_mask(geom, sample_coords)
        if not np.any(inside):
            print(
                f"Skipping element {element} because it is not contained in the shape."
            )
            continue

        contained_names = ad_sample.obs_names[inside]
        current = adata.obs.loc[contained_names, out_key]
        unannotated = current.isna() | (current == "NA")
        if np.any(unannotated):
            adata.obs.loc[contained_names[unannotated], out_key] = element


def spatial_sample_split(
    adata, sample_key, displacement=1000, out_key="spatial_split", in_key="spatial"
):
    sample_widths = []
    for cc in adata.obs[sample_key].unique():
        mask = adata.obs[sample_key] == cc
        coords_orig = adata[mask].obsm[in_key]
        x_min = coords_orig[:, 0].min()
        x_max = coords_orig[:, 0].max()
        width = x_max - x_min
        sample_widths.append(width)
    max_sample_width = max(sample_widths)

    adata.obsm[out_key] = adata.obsm[in_key].copy()
    interval = max_sample_width + displacement
    for i, cc in enumerate(adata.obs[sample_key].unique()):
        mask = adata.obs[sample_key] == cc
        coords_orig = adata[mask].obsm[in_key]
        x_min, y_min = coords_orig.min(axis=0)
        coords = coords_orig - np.array([x_min, y_min])[None, :]
        coords[:, 0] += i * interval
        adata.obsm[out_key][mask.values] = coords
