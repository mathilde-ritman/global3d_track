'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask

# to use to diagnose model relative humidity


def get_alpha(T):
    ''' T-dependant weightings for water vs ice saturation vapour pressure contributions. Taken from doi:10.21957/4whwo8jw0 P116'''

    Tice = 250.16 # K
    T0 = 273.16 # K

    # alpha = 0 for T <= Tice
    # alpha = 1 for T >= T0
    # alpha = frac for T0 < T < Tice

    frac = ((T - Tice) / (T0 - Tice)) ** 2
    alpha = xr.where(T <= Tice, 0, xr.where(T >= T0, 1, frac))
    return alpha

def calc_es(ds, phase):
    ''' calculate saturation vapour pressure for water or ice using the Tetens forumla, as described in doi:10.21957/4whwo8jw0 P116 '''

    constants = {'water': {'a': 611.2, 'b': 17.502, 'c': 32.19},
                 'ice': {'a': 611.2, 'b': 22.587, 'c': -0.7}}
    
    T = ds.ta # K
    T0 = 273.16 # K
    d = constants[phase] # choose params
    
    es = d['a'] * np.exp(d['b'] * ((T - T0) / (T - d['c']))) # Pa
    return es


def saturation_vapour_pressure(ds):

    alpha = get_alpha(ds.ta) # mixed phase weighting
    es = alpha * calc_es(ds, 'water') + (1 - alpha) * calc_es(ds, 'ice') # Pa

    return es

def vapour_pressure(ds):

    p = ds.pfull
    q = ds.hus # kg kg-1
    Rd = 287.04 # J kg-1 K-1
    Rv = 461.4 # J kg-1 K-1
    eps = Rd / Rv

    e = (p * q) / (eps * (1 + q * (1/eps - 1))) # Pa

    return e

def relative_humidity(ds):

    ### not tested for upper levels 

    e = vapour_pressure(ds) # Pa
    es = saturation_vapour_pressure(ds) # Pa

    RH = e / es # unitless

    return RH