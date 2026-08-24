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

def version_name(yaml, use_tobac_version=False, start_date=None, datestr=None):
    if datestr is None:
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

#### load records ####
            

def load_single_record(rundir, tbcdir=None):
    ''' Collects the tracking record keeping. Only inlude tbcdir if this fails. '''

    # load track records for the day
    updrafts = pd.read_hdf(list(rundir.glob('*updraft_tracks.h5'))[0], 'table').rename(columns={'tracks':'updraft'})
    clouds = pd.read_hdf(list(rundir.glob('*frozen_tracks.h5'))[0], 'table').rename(columns={'tracks':'frozen'})
    
    if tbcdir:
        tbc_updrafts = pd.read_hdf(tbcdir / 'updraft/tracked_features.h5','table')
        updrafts = tbc_updrafts.merge(updrafts, left_on='cell', right_on='cell', how='left')
        tbc_clouds = pd.read_hdf(tbcdir / 'frozen/tracked_features.h5','table')
        clouds = tbc_clouds.merge(clouds, left_on='cell', right_on='cell', how='left')
        
    # load overall system maps
    system_umaps = pd.read_hdf(list(rundir.glob('*system_label_maps.h5'))[0], 'table') # maps: updraft -> anvil
    system_amaps = pd.read_hdf(list(rundir.glob('*system_extra_maps.h5'))[0], 'table').rename(columns={'frozen_new':'dcc'}) # maps: anvil -> DCC
    system_maps = system_umaps.merge(system_amaps, left_on='frozen', right_on='frozen', how='left').fillna(0) # maps: updraft ->  anvil -> dcc
    
    # merge variable records with overall tracking results
    updrafts = updrafts.merge(system_maps[['updraft','dcc']], left_on='updraft', right_on='updraft', how='left').fillna(0)
    clouds = clouds.merge(system_maps[['frozen','dcc']], left_on='frozen', right_on='frozen', how='left').fillna(0)
    dcc_times = pd.concat((updrafts[['time','dcc','updraft']], clouds[['time','dcc',]])).fillna(0)

    return {'w':updrafts, 'cld':clouds, 'df':dcc_times}

''' Collects the tracking record keeping, like above, but for tracked datasets that span multiple days and have been linked up. '''
    
def collect_link_maps(link_maps, var, new_name):
    flag = 0
    for col in link_maps.columns:
        if 'secondary' in col:
            flag = 1
    if flag:
        link_maps = link_maps[[var,f'{var}_linked']].merge(link_maps[[f'{var}_secondary',f'{var}_linked_secondary']], left_on=f'{var}_linked', right_on=f'{var}_secondary', how='left').fillna(0)
        link_maps = link_maps.drop(columns=[f'{var}_linked',f'{var}_secondary']).rename(columns={var:new_name,f'{var}_linked_secondary':f'{new_name}_linked'})
    else:
        link_maps = link_maps.rename(columns={var:new_name,f'{var}_linked':f'{new_name}_linked'})
    return link_maps

def apply_linking(df, link_maps, var='dcc'):
    shift_val = link_maps[var].where(link_maps[var]>0).min() - 1
    df[var] = (df[var] + shift_val).where(df[var] > 0, 0)
    return df.merge(link_maps, left_on=var, right_on=var, how='left')

def load_records(track_dir, day, tbcdir=None):
    # load track records for the day
    updrafts = pd.read_hdf(list(track_dir.glob('%s/*updraft_tracks.h5' %day))[0], 'table').rename(columns={'tracks':'updraft'})
    clouds = pd.read_hdf(list(track_dir.glob('%s/*frozen_tracks.h5' %day))[0], 'table').rename(columns={'tracks':'frozen'})
    if tbcdir:
        tbc_updrafts = pd.read_hdf(tbcdir / 'updraft/tracked_features.h5','table')
        updrafts = tbc_updrafts.merge(updrafts, left_on='cell', right_on='cell', how='left')
        tbc_clouds = pd.read_hdf(tbcdir / 'frozen/tracked_features.h5','table')
        clouds = tbc_clouds.merge(clouds, left_on='cell', right_on='cell', how='left')
    # load overall system maps
    system_umaps = pd.read_hdf(list(track_dir.glob('%s/*system_label_maps.h5' %day))[0], 'table') # maps: updraft -> anvil
    system_amaps = pd.read_hdf(list(track_dir.glob('%s/*system_extra_maps.h5' %day))[0], 'table').rename(columns={'frozen_new':'dcc'}) # maps: anvil -> DCC
    system_maps = system_umaps.merge(system_amaps, left_on='frozen', right_on='frozen', how='left').fillna(0) # maps: updraft ->  anvil -> dcc
    # merge variable records with overall tracking results
    updrafts = updrafts.merge(system_maps[['updraft','dcc']], left_on='updraft', right_on='updraft', how='left').fillna(0)
    clouds = clouds.merge(system_maps[['frozen','dcc']], left_on='frozen', right_on='frozen', how='left').fillna(0)
    # load linking
    link_maps = pd.read_hdf(list(track_dir.glob('%s/*linked_system_maps.h5' %day))[0], 'table')
    updraft_maps = pd.read_hdf(list(track_dir.glob('%s/*linked_u_tracks_maps.h5' %day))[0], 'table')
    link_maps = collect_link_maps(link_maps, 'system', 'dcc')
    updraft_maps = collect_link_maps(updraft_maps, 'u_tracks', 'updraft')
    # link the updrafts in time
    updrafts = apply_linking(updrafts, updraft_maps, var='updraft')    
    # link the overall results in time
    updrafts = apply_linking(updrafts, link_maps, 'dcc')
    clouds = apply_linking(clouds, link_maps, 'dcc')
    # return
    updrafts = updrafts.rename(columns={'dcc':'dcc_daily','dcc_linked':'dcc','updraft':'updraft_daily','updraft_linked':'updraft'})
    clouds = clouds.rename(columns={'dcc':'dcc_daily','dcc_linked':'dcc'})
    return {'w':updrafts, 'cld':clouds}

def load_linked_records(track_dir, days, tbcdir=None):
    updrafts = None
    clouds = None
    for d in days:
        di = load_records(track_dir, d, tbcdir)
        updrafts = pd.concat((updrafts, di['w']))
        clouds = pd.concat((clouds, di['cld']))
    dcc_times = pd.concat((updrafts[['time','dcc','updraft']], clouds[['time','dcc',]]))
    return {'w':updrafts, 'cld':clouds, 'df':dcc_times}