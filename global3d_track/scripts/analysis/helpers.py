import xarray as xr
import glob
import re
import pathlib
import logging
import numpy as np
import pandas as pd


def load_stats(files, variables=[], batch=None, size=None, apply=None, sidx_list=None, sidx_ignore=[], n_clouds=1):

    # get file list
    regex = r'_(\d+)\.0.nc$'
    if isinstance(files, str) or isinstance(files, pathlib.Path):
        flist = glob.glob(str(files)+'/*')
        files = sorted(flist, key=lambda x: int(re.search(regex, x).group(1)))

    # choose exact
    if sidx_list is not None:
        get_these = []
        for f in files:
            sidx = int(re.search(regex, pathlib.Path(f).name).group(1))
            if sidx in sidx_list:
                get_these.append(f)
        files = get_these

    # subselect samples
    else:
        if batch is not None:
            files = files[(batch-1)*size:batch*size]
        else:
            files = files[:n_clouds]
        for f in files:
            sidx = int(re.search(regex, pathlib.Path(f).name).group(1))
            if sidx in sidx_ignore:
                files.pop(files.index(f))

    # load
    datasets = []
    for fname in files:
        try:
            ds = xr.open_dataset(fname)
            sidx = int(re.search(regex, pathlib.Path(fname).name).group(1))
            ds['system'] = sidx
            if variables:
                ds = ds[variables]
            if callable(apply):
                ds = apply(ds)
            ds = ds.expand_dims(system=[sidx])
            datasets.append(ds)
        except Exception as e:
            logging.warning(f"skipping {fname} with exception: {e}")
    if not datasets:
        raise ValueError("no datasets loaded successfully")
    return xr.concat(datasets, dim="system")


def normalise_by_lifetime(obj_exists, ds, bins=np.arange(0, 1.05, 0.05)):

    def normalise_by_lifetime_vectorized(data, exists):
        data = data[exists]
        ntimes = data.size
        if ntimes < 2:
            return np.full((len(bins),), np.nan, dtype=float)
        time_percentage = np.linspace(0, 1, ntimes)
        return np.interp(bins, time_percentage, data)

    def apply_normalisation(da, cloud_exists):
        result = xr.apply_ufunc(
            normalise_by_lifetime_vectorized,
            da,
            cloud_exists,
            input_core_dims=[['time'], ['time']],
            output_core_dims=[['interp_time']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={"output_sizes": {"interp_time": len(bins)}},
            output_dtypes=[float],
        )
        return result.assign_coords(interp_time=bins)
    
    result = xr.Dataset(attrs=ds.attrs)

    for name, da in ds.data_vars.items():
        if 'time' in da.dims:
            result[name] = apply_normalisation(da, obj_exists)
        else:
            result[name] = da

    for name, coord in ds.coords.items():
        if 'time' not in coord.dims:
            result = result.assign_coords({name: coord})

    return result