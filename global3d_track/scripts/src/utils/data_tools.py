''' 
Mathilde Ritman 2023, mathilde.ritman@physics.ox.ac.uk
Adapted from William Jones: https://github.com/w-k-jones/tobac_icon_hackathon.git
'''

import intake
import xarray as xr
from datetime import timedelta
import numpy as np
import pandas as pd
from . import regrid


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


def load_tobac_data(variables, region, start_date, end_date):
    # load data
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    dataset = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().sel(time=slice(start_date, end_date-timedelta(minutes=1)))

    # process data
    # ensure no repeats ni the variables
    variables = list(set(list(variables) + ['zghalf','zg']))
    li = regrid.Regrid(region).perform(dataset[variables], zoom=9, resolution=0.1)
    relevant_data = xr.concat(li, dim='time')
    data = preprocess_for_tobac(relevant_data)
    if 'cli' in data.data_vars and 'clw' in data.data_vars:
        data['cl'] = data.cli + data.clw

    return data


def load_corresponding_data(mask, region, variables=['cli','clw']):
    # times
    start = pd.to_datetime(mask.time[0].values)
    end = pd.to_datetime(mask.time[-1].values)
    # load data
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    dataset = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().sel(time=slice(start, end))
    # process data
    li = regrid.Regrid(region).perform(dataset[variables], zoom=9, resolution=0.1)
    data = xr.concat(li, dim='time')
    return data