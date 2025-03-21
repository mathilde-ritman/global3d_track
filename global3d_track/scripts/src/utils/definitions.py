''' 
Mathilde Ritman 2024, mathilde.ritman@physics.ox.ac.uk

'''

import numpy as np
from my_library import multivariate_tobac 
from my_library.calculations import group_propertiesV2 as gg
import logging
# Set up the logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


#### ------------------------ mains ------------------------ ####

def define_objects(ds, data):

    if 'core' in ds.data_vars:
        ds = ds.rename({'core':'updraft'})
    if 'anvil' in ds.data_vars:
        ds = ds.rename({'anvil':'ice'})

    ds = define_core(ds, data)
    ds = define_anvil(ds, data)
    return ds

def define_core(ds, data):
    tobac_data = data
    config = '/home/b/b382635/s/global_DCC_tracking/submission_files/core_config.yaml'
    # track
    track_module = multivariate_tobac.Track(tobac_data.wa_phy, tobac_data.wa_phy, config, overwrite=True)
    cores, df = track_module.perform(track=True, merge=True, save=False)
    # get max vertical velocities
    df = get_statistic(df, cores.feature, tobac_data.wa_phy)
    # require reaches 1 m/s at some point
    r = df[['merged','max_w']].groupby('merged').max()
    cores_above_thresh = r[r>=1].dropna().index
    cores = cores.where(cores.merged.isin(cores_above_thresh)).where(ds.system>0)
    # collect
    ds['core_features'] = cores.merged
    ds['core'] = ds.system.where(ds.core_features>0)
    return ds

def define_anvil(ds, data, core_name='core'):
    # system total condensate
    data = data.sel(time=ds.time, lat=ds.lat, lon=ds.lon)
    if not 'cl' in data.data_vars:
        data['cl'] = data.cli + data.clw
    cl = data[['cl']].where(ds.system>0)
    calc_di = {'total_cl':('cl', 'sum'),}
    system_cl = gg.all_aggregate(ds.system, cl, calc_di, calc_func=gg.calc_spatial_aggregate)
    # invert generalised height axis so that up direction is TOA
    system_cl = system_cl.sel(level_full=system_cl.level_full[::-1])
    ABH = discover_abh(system_cl.total_cl)
    # apply abh height threshold (MEAN)
    abh = ds.system.copy()
    for i in ABH.system:
        sys = ABH.sel(system=i).system.item()
        abh = abh.where(ds.system != sys, ABH.sel(system=i).mean().values)
    ds['anvil'] = ds.system.where(ds.level_full <= abh)
    # remove intruding core from anvil
    ds['anvil'] = ds.anvil.where(~(ds[core_name] > 0))
    return ds


#### ------------------------ helpers ------------------------ ####


def get_statistic(df, mask, field):
    masked_field = field.where(mask>0)
    group = masked_field.groupby(mask).max()
    df.loc[:,'max_w'] = group.sel(feature=slice(1,None)).values
    return df

def derivative(da, smooth=True, interp=False, verbose=False):
    if da.level_full[0] == 90:
        # ensure values increasing
        da = da.sel(level_full = da.level_full[::-1])
    if interp:
        n,x = da.level_full[0].item(), da.level_full[-1].item()
        da_interp = da.interp(level_full=np.arange(n, x+.5, .5), method='linear')
        da = da_interp
    if smooth:
        da_smooth = da.rolling(level_full=3, center=True).mean('level_full')
        da = da_smooth
    da = da.compute().differentiate('level_full') * -1
    da = da.sel(level_full = da.level_full[::-1])
    da.attrs['positive'] = 'up'
    return da

def discover_abh(da):
    # determines the anvil base height as the minima that precedes the hightest maxima in cloud condensate
    da = da.compute()
    # derivative
    d = derivative(da, verbose=True)
    # find minima and maxima (approx)
    maxima = (d > 0) & (d.shift(level_full=-1) <= 0)
    minima = (d <= 0) & (d.shift(level_full=-1) > 0)
    # find minima that precedes the last maxima, assumming the latter represents the anvil!
    last_max = maxima.sel(level_full=maxima.level_full[::-1]).idxmax('level_full')
    minima_below = minima.where(minima.level_full >= last_max)
    abh = minima_below.sel(level_full=minima_below.level_full[::-1]).idxmax('level_full')
    return abh
