'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import numpy as np
import xarray as xr
import pandas as pd
from .connect_contiguous import Connect

def track_connected_components(d, dims_to_skip=(), PBC_flag=None):
    # track using connected components
    arr = Connect(d>0).get_components(dims_to_skip=dims_to_skip, PBC_flag=PBC_flag)
    labels = xr.DataArray(data=arr, dims=d.dims, coords=d.coords)
    return labels.where(labels>0)


def child_that_overlaps(parent, child):
    child_overlap = child.where(parent>0) # child features coinciding with the parent
    child_overlap = child_overlap.astype(float)
    overlapping_features = np.unique(child_overlap.values[~np.isnan(child_overlap.values)])
    output = child.where(np.isin(child, overlapping_features))
    return output.where(output>0)
    

def union_all(datasets):
    # fill NaN with 0
    for i,d in enumerate(datasets):
        datasets[i] = d.where(d>0).fillna(0)
    # collect value at all points covered
    da_union = datasets[0]
    for i in range(1,len(datasets)):
        da_union = np.maximum(da_union, datasets[i])
    da_union = da_union.where(da_union>0) # 0s go back to NaN
    return da_union

def force_consecutive_labels(da, table_path=None, current_col=None, update_col=None, new_tobac_table=True):
    values = da.fillna(0).values.ravel()
    _, consecutive = np.unique(values, return_inverse=1) # force consecutive numbers
    da = xr.DataArray(data=consecutive.reshape(da.shape), dims=da.dims, coords=da.coords)
    # record mapping, optional
    if isinstance(table_path, str):
        df = pd.read_hdf(table_path, 'table')
        mapping = dict(zip(values, consecutive)) # as dict
        # save
        df[update_col] = df[current_col].map(mapping).fillna(0) # record as new column
        if new_tobac_table:
            outpath = table_path.with_name(f"{table_path.stem}_{update_col}{table_path.suffix}")
        else:
            outpath = table_path
        df.to_hdf(outpath, 'table')
    return da.where(da>0)