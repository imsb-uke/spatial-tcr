from collections.abc import Iterable

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import (
    TableModel,
    get_table_keys,
)


def filter_sdata(
    sdata,
    elements: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
    tables: Iterable[str] | None = None,
    obs_keys: Iterable[str] | None = None,
    var_keys: Iterable[str] | None = None,
    var_names: Iterable[str] | None = None,
    layers: Iterable[str] | None = None,
) -> SpatialData:
    """
    Filter a SpatialData object to contain only specified elements or table entries.

    Parameters
    ----------
        elements: Names of elements to include. Defaults to [].
        regions: Regions to include in the table. Defaults to regions of all selected elements.
        tables: Names of tables to include. Defaults to ["table"].
        obs_keys: Names of obs columns to include. Defaults to [].
        var_keys: Names of var columns to include. Defaults to [].
        var_names: Names of variables (X columns) to include. Defaults to [].
        layers: Names of X layers to include. Defaults to [].

    Returns
    -------
        A new SpatialData object
    """
    elements = [] if elements is None else list(elements)
    regions = elements if regions is None else regions
    obs_keys = [] if obs_keys is None else obs_keys
    var_keys = [] if var_keys is None else var_keys
    var_names = [] if var_names is None else list(var_names)  # iterable and sized
    tables = ["table"] if tables is None else tables
    layers = [] if layers is None else layers

    sdata_subset = (
        sdata.subset(element_names=elements, filter_tables=True)
        if elements
        else SpatialData()
    )
    # We rely on `subset` returning an unbacked copy, so we don't modifying the original data.
    assert not sdata_subset.is_backed()
    # Further filtering on the tables
    for table_name, table in list(sdata_subset.tables.items()):
        if table_name not in tables:
            del sdata_subset.tables[table_name]
            continue
        _, region_key, instance_key = get_table_keys(table)
        print(region_key, instance_key)
        obs_keys = list(obs_keys)
        if instance_key not in obs_keys:
            obs_keys.insert(0, instance_key)
        if region_key not in obs_keys:
            obs_keys.insert(0, region_key)
        # Preserve order by checking "isin" instead of slicing. Also guarantees no duplicates.
        table_subset = table[
            table.obs[region_key].isin(regions),
            table.var_names.isin(var_names),
        ]
        layers_subset = (
            {key: layer for key, layer in table_subset.layers.items() if key in layers}
            if table_subset.layers is not None and len(var_names) > 0
            else None
        )
        table_subset = TableModel.parse(
            AnnData(
                X=table_subset.X if len(var_names) > 0 else None,
                obs=table_subset.obs.loc[:, table_subset.obs.columns.isin(obs_keys)],
                var=table_subset.var.loc[:, table_subset.var.columns.isin(var_keys)],
                layers=layers_subset,
            ),
            region_key=region_key,
            instance_key=instance_key,
            region=np.unique(table_subset.obs[region_key]).tolist(),
        )
        del sdata_subset.tables[table_name]
        sdata_subset.tables[table_name] = table_subset
    return sdata_subset
