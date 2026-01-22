''' 
Mathilde Ritman 2023, mathilde.ritman@physics.ox.ac.uk
Adapted from William Jones: https://github.com/w-k-jones/tobac_icon_hackathon.git
'''

import intake
import xarray as xr
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from . import regrid
import glob
from pathlib import Path
import logging

# ---------- these ones help prep the field data ---------------- #

def preprocess_for_tobac(dataset):

    # kill stratosphere
    if 'level_full' in dataset.dims:
        dataset = dataset.sel(level_full=slice(40,90))
    if 'level_half' in dataset.dims:
        dataset = dataset.sel(level_half=slice(41,91))
    
    # force shared model levels using linear interpolation to estimate variables defined on half levels at full levels
    dataset['level_half'] = dataset.level_full.values + .5 # shift index value for correct linear interpolation
    for v in ['wa_phy', 'zghalf']:
        if v in dataset.data_vars or v in dataset.coords:
            dataset[v] = dataset[v].interp(level_half=dataset.level_full, method="linear", kwargs={"fill_value": "extrapolate"},)
    
    # drop dims
    dataset = dataset.drop_dims('level_half')
    if 'crs' in dataset.dims:
        dataset = dataset.drop_dims('crs')
        
    # demote height coords as these will confuse tobac
    dataset = dataset.reset_coords(['zg','zghalf'])
    # drop zghalf as this is not equal to zg
    dataset = dataset.drop('zghalf')
        
    return dataset

def add_height_data(dataset, height_values):
    dataset['height'] = ('level_full', height_values)
    return dataset.swap_dims({'level_full': 'height'})


def load_tobac_data(variables, region, start_date, end_date, model_version='4008a'):
    # load data
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    if model_version=='4008a':
        dataset = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().sel(time=slice(start_date, end_date-timedelta(minutes=1)))
    else:
        raise ValueError("Model version not implemented")
    # process data
    # ensure no repeats in the variables
    variables = list(set(list(variables) + ['zghalf','zg']))
    relevant_data = regrid.Regrid(region).perform(dataset[variables], zoom=9, resolution=0.1)
    # relevant_data = xr.concat(li, dim='time') - no longer needed regrid outputs ds, not list of ds
    data = preprocess_for_tobac(relevant_data)
    if 'cli' in data.data_vars and 'clw' in data.data_vars:
        data['cl'] = data.cli + data.clw

    return data


def load_corresponding_data(mask, region=None, variables=['cli','clw'], preceeding_mins=0):
    # times
    start = pd.to_datetime(mask.time[0].values)
    end = pd.to_datetime(mask.time[-1].values)
    # get some preceeding times
    if preceeding_mins > 0:
        start = start - timedelta(minutes=preceeding_mins)
    # load data
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    dataset = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().sel(time=slice(start, end))
    # process data
    if not region:
        region = (mask.lon.min().item(), mask.lon.max().item()+.1, mask.lat.min().item(), mask.lat.max().item()+.1)
    if variables is None:
        variables = list(dataset.data_vars)
    data = regrid.Regrid(region).perform(dataset[variables], zoom=9, resolution=0.1)
    return data


def file_contains_dates(fpath, start, end):
    ''' Check if file is within time range '''
    # datetimes spanned
    fname = fpath.split('/')[-1]
    str_segs = fname.split('_')
    fdates = [datetime.strptime(x, "%Y%m%dT%H%M") for x in str_segs if 'T' in x]
    fstart = min(fdates)
    fend = max(fdates)
    # is within time range
    is_within = False
    # starts within file
    if fstart <= start and start <= fend:
        is_within = True
    # ends within file
    if fstart <= end and end <= fend:
        is_within = True
    return is_within

def grab_system(s, data_dir):

    data_dir = Path(data_dir)

    def system_start(s):
        c = data_dir / 'data_filtering_stats/system_core_exists-20210701T0000_20210707T2345.csv'
        c = pd.read_csv(c, index_col='system_id', low_memory=False)
        core_exists = (c == 1) + (c == '1.0') + (c == 'True')
        return pd.to_datetime(core_exists.idxmax(axis=1).loc[s])

    def str_to_td(value):
        parts = value.split(" days ")
        days = int(parts[0])  # Extract the number of days
        time_part = parts[1]  # Extract the HH:MM:SS part
        t = datetime.strptime(time_part, "%H:%M:%S")  # Parse time
        return timedelta(days=days, hours=t.hour, minutes=t.minute, seconds=t.second)

    def system_duration(s):
        lifetime = data_dir / 'data_filtering_stats/system_lifetime.csv'
        lifetime = pd.read_csv(lifetime, index_col='system_id')
        return lifetime.iloc[:,0].apply(str_to_td).loc[s]

    def temporal_ext(s):
        start = system_start(s)
        duration = system_duration(s)
        end = start + duration
        return start, end

    def grab_times(s, data_dir):
        # find system time range
        start, end = temporal_ext(s)
        files = glob.glob(str(data_dir / '*/*_proc.nc'))
        # query files for time range
        files_to_search = [f for f in files if file_contains_dates(f, start, end)]
        # query datasets for time range
        ds = xr.open_mfdataset(files_to_search, chunks={"time": 1, "lat": 100, "lon": 100})
        return ds.sel(time=slice(start, end))

    def spatial_ext(ds, s):
        ds_sys2d = (ds.system == s).max(('time','level_full')).compute()
        # logging.info(f"{ds_sys2d=}")
        lats = ds.lat[ds_sys2d.max('lon')]
        lons = ds.lon[ds_sys2d.max('lat')]
        # logging.info(f"{lats=}; {lons=}")
        return ds.sel(lon=lons, lat=lats)

    ds = grab_times(s, data_dir)
    mask = spatial_ext(ds, s)
    logging.info(f"masking to system {s}")
    mask = mask.where(ds.system == s) # drop coincident clouds
    mask['lat'] = mask.lat.round(2)
    mask['lon'] = mask.lon.round(2)
    return mask

def grab_system_data(mask, variables):
    dataset = load_corresponding_data(mask, None, variables=variables)
    dataset = preprocess_for_tobac(dataset).chunk({'time': 1, 'lat': 100, 'lon': 100})
    dataset['lat'] = dataset.lat.round(2)
    dataset['lon'] = dataset.lon.round(2)
    mask['lat'] = mask.lat.round(2)
    mask['lon'] = mask.lon.round(2)
    return dataset.where(mask.system > 0)