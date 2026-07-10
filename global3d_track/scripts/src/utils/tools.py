''' 
Mathilde Ritman 2023, mathilde.ritman@physics.ox.ac.uk

'''

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import logging
import yaml
import os
import pathlib
import glob
import tempfile
# Set up the logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def sort_files(files):
    ''' Sort files by datetime '''
    # datetimes spanned
    fpaths = files.copy()
    starts, ends = [], []
    for f in fpaths:
        fname = f.replace('.nc','').split('/')[-1]
        str_segs = fname.split('_')
        fdates = [datetime.strptime(x, "%Y%m%dT%H%M") for x in str_segs if 'T' in x]
        starts.append(min(fdates))
        ends.append(max(fdates))
    # find file that starts when preceeding file ends
    sorted_files = []
    next_start = min(starts)
    while len(fpaths) > 0:
        for i, f in enumerate(fpaths):
            if starts[i] == next_start:
                sorted_files.append(f)
                next_start = ends[i]
                fpaths.pop(i)
                starts.pop(i)
                ends.pop(i)
                found = True
        if len(fpaths) and next_start >= max(ends): 
            break
    return sorted_files

def check_file_dates(fpath, start, end):
    ''' Check if file is within time range '''
    # datetimes spanned
    fname = fpath.split('/')[-1]
    str_segs = fname.split('_')
    fdates = [datetime.strptime(x, "%Y%m%dT%H%M") for x in str_segs if 'T' in x]
    fstart = min(fdates)
    fend = max(fdates)
    # is within time range
    is_within = False
    if start <= fstart and fstart < end:
        is_within = True
    if start < fend and fend <= end:
        is_within = True
    return is_within

def compress_and_save(ds, fpath):
    ''' Compress and save output '''
    # Add compression encoding
    logging.info(f"{datetime.now()} Compressing output")
    comp = dict(zlib=True, complevel=5, shuffle=True)
    if isinstance(ds, xr.Dataset):
        for var in ds.data_vars:
            var_type = ds[var].dtype
            if np.issubdtype(var_type, np.integer) or np.issubdtype(var_type, np.floating):
                ds[var].encoding.update(comp)
    else:
        var_type = ds.dtype
        if np.issubdtype(var_type, np.integer) or np.issubdtype(var_type, np.floating):
            ds.encoding.update(comp)
    # save
    logging.info(f"{datetime.now()} Saving output: {fpath}")
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    ds.to_netcdf(fpath)
    ds.close()

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        di = yaml.safe_load(f)
    return di

def version_name(yaml, use_tobac_version=False, start_date=None):
    if start_date is None:
        start_date = datetime.strptime(yaml['start_date'], "%Y-%m-%d %H:%M:%S")
    datestr = start_date.strftime('%Y%m%d')
    if not use_tobac_version:
        name = pathlib.Path(yaml['version_name'], yaml['region'], datestr)
    else:
        name = pathlib.Path(yaml['tobac_version'], yaml['region'], datestr)
    return name

def collect_tobac_features(sdir, feature_type, remove=False):
    ''' Collect features from multiple files '''    

    # filesystem
    fdir = pathlib.Path(sdir, feature_type)
    all_features = sorted(fdir.glob('*/features.h5'))
    all_masks = sorted(fdir.glob('*/segmented_mask.nc'))

    # check whether this is needed (are there any files to collect?)
    if len(all_features) == 0 or len(all_masks) == 0:
        return

    # collect table of all features in directory
    li = []
    for f in all_features:
        li.append(pd.read_hdf(f, 'table'))
    df = tobac.utils.general.combine_feature_dataframes(li)
    df.to_hdf(fdir / 'features.h5', 'table')

    # collect all masks as one
    curr_m = xr.open_dataset(all_masks[0])
    for i in range(len(all_masks)-1):
        highest_label = curr_m.feature.max()
        next_m = xr.open_dataset(all_masks[1+i])
        next_m['feature'] = (next_m.feature + highest_label).where(next_m.feature > 0, 0)
        curr_m = xr.concat((curr_m, next_m), dim='time')
    curr_m.to_netcdf(fdir / 'segmented_mask.nc')
    curr_m.close()

    # remove individual files if desired
    if remove:
        for subdir in fdir.iterdir():
            if subdir.is_dir():
                for f in subdir.iterdir():
                    if f.is_file():
                        f.unlink()
                subdir.rmdir()


def make_directories(dirs):
    ''' Make directories if they do not exist '''
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_chunks(dims, sizes=None):
    if isinstance(dims, xr.DataArray) or isinstance(dims, xr.Dataset):
        sizes = dims.sizes
        dims = dims.dims
        chunks = {}
        for dim in dims:
            if dim == "time":
                chunks[dim] = 1
            elif dim in ("lat", "lon"):
                chunks[dim] = min(sizes[dim], 128*2)
            else:
                chunks[dim] = sizes[dim]
        return chunks
    else:
        chunks = []
        for dim in dims:
            if dim == "time":
                chunks.append(1)
            elif dim in ("lat", "lon"):
                chunks.append(min(sizes[dim], 128*2))
            else:
                chunks.append(sizes[dim])
        return tuple(chunks)

def save_xarray(data, path, engine="h5netcdf", fill_value=0):
    ''' Save xarray dataset or dataarray with compression and a fill value of 0 (assumes data is label data) '''
    # apply and ensure dtype is positive integer (assumes label data)
    if isinstance(data, xr.DataArray):
        if data.name is None:
            data = data.rename("data")
        if not np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.uint32)
        encoding_params = dict(zlib=True, complevel=4, chunksizes=get_chunks(data.dims, data.sizes))
        encoding = {data.name: encoding_params}
    else:
        encoding = {}
        for var in data.data_vars:
            if not np.issubdtype(data[var].dtype, np.integer):
                data[var] = data[var].astype(np.uint32)
            encoding_params = dict(zlib=True, complevel=4, chunksizes=get_chunks(data[var].dims, data[var].sizes))
            encoding[var] = encoding_params
    # atomic write: write to a temporary file and replace
    path = pathlib.Path(path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=str(path.parent))
    os.close(tmp_fd)
    try:
        data.fillna(fill_value).to_netcdf(tmp_path, encoding=encoding, engine=engine)
        os.replace(tmp_path, str(path))
    finally:
        # ensure tmp removed if something failed
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
