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

def version_name(yaml, use_tobac_version=False):
    datestr = datetime.strptime(yaml['start_date'], "%Y-%m-%d %H:%M:%S").strftime('%Y%m%d')
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
