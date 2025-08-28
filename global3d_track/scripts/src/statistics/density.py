'''
Mathilde Ritman 2025
'''

import dask

# diagnose model density, mass mixing ratios, and ice water path


def density(ds):
    ''' uses ideal gas law to get total (dry and moist) air density '''
    # thermodynamic variables
    p = ds.pfull # Pa (kg m-1 s-2)
    T = ds.ta # K
    Rd = 287.04 # J kg-1 K-1 (m2 s-2 K-1)
    Rv = 461.4 # J kg-1 K-1 (m2 s-2 K-1)
    
    # specific vapour
    q_v = ds.hus # kg kg-1
    
    # ICON model eqn state gives
    q_condensate = ds.cli + ds.clw + ds.qg + ds.qr + ds.qs # kg kg-1
    alpha = ((Rv / Rd) - 1) * q_v - q_condensate
    rho = p / (Rd * T * (1 + alpha)) # kg m-3
    return rho

# get condensate concentration and path

def calculate_xWC(ds, v='cli'):
    # total air density
    rho = density(ds) # kg m-3

    # specific mass fractions - mass of quantity per mass of total air
    q_x = ds[v] # kg kg-1

    # density of quantity (e.g., liquid water content)
    rho_x = q_x * rho # kg m-3
    return rho_x

def calculate_xWP(ds, v='cli'):
    # density of quantity
    rho_x = calculate_xWC(ds, v) # kg m-3

    # total mass per tropospheric column of air (e.g., liquid water path)
    grid_depth = ds.dzghalf # m
    xWP = (rho_x * grid_depth).sel(level_full=slice(23,90)).sum('level_full') # kg m-2
    return xWP

def calculate_IWP(ds, verbose=0):
    ''' calculate ice water path from frozen hydrometeors: ice, snow and graupel '''
    # densities
    q_frozen = ds['cli'] + ds['qs'] + ds['qg'] # kg kg-1
    rho_frozen = density(ds) * q_frozen # kg m-3

    # iwp
    grid_depth = ds.dzghalf # m
    IWP = (rho_frozen * grid_depth).sel(level_full=slice(23,90)).sum('level_full') # kg m-2
    if verbose:
        return IWP, rho_frozen
    return IWP

