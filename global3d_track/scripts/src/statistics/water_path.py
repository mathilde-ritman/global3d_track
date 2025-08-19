'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
import logging

# calculate for IWP, LWP and similar



def calculate_xWC(ds, v='cli'):
    ''' uses ideal gas law to get moist air density '''
    # thermodynamic variables
    p = ds.pfull # Pa (kg m-1 s-2)
    T = ds.ta # K
    Rd = 287.04 # J kg-1 K-1 (m2 s-2 K-1)
    Rv = 461.4 # J kg-1 K-1 (m2 s-2 K-1)

    # specific vapour
    q_v = ds.hus # kg kg-1

    # ICON model eqn state gives
    q_condensate = ds.cli + ds.clw + ds.qg + ds.qr + ds.qs # kg kg-1
    alpha = (Rv / Rd - 1) * q_v - q_condensate
    rho = p / (Rd * T * (1 + alpha)) # kg m-3

    # density of quantity
    q_x = ds[v] # kg kg-1
    rho_x = q_x * rho # kg m-3
    return rho_x

def calculate_xWP(ds, v='cli'):
    ''' uses ideal gas law to get moist air density '''
    # content
    rho_x = calculate_xWC(ds, v) # kg m-3

    # path
    grid_depth = ds.dzghalf # m
    xWP = (rho_x * grid_depth).sum('level_full') # kg m-2
    return xWP