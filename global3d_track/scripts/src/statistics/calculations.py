'''
Mathilde Ritman 2025

Contains functions to calculate the following:
- air density from the ideal gas law
- condensate density and path from specific mass fractions
- ice water path from frozen hydrometeors
- relative humidity from vapour and saturation vapour pressures

Also contains a class to calculate convective mass fluxes (CMF)

'''

import xarray as xr
import numpy as np
import dask

# to use to diagnose model relative humidity


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


class CMF:

    def __init__(self) -> None:
        self.grid_spacings = 11000 # m
        self.grid_area = self.grid_spacings**2 # m2

    def get_mass(self, d, quantity=None):
        if quantity is None:
            return 1
        if isinstance(quantity, (list, tuple)):
            return xr.concat([d[q] for q in quantity], dim='q').sum('q', skipna=True)
        return d[quantity]

    def mass_flux(self, masked_data, quantity=None, rho=1):

        # calculate
        mass = self.get_mass(masked_data, quantity) # kg kg-1
        return mass * masked_data.wa_phy * rho # kg s-1 m-2
    
    def area_mass_flux(self, masked_data, quantity=None, RHO=1):
        ''' assumes constant density. '''

        # calculate
        area = (masked_data.wa_phy>0).sum(('lat','lon'), skipna=True) * self.grid_area
        mass = self.get_mass(masked_data, quantity) # kg kg-1
        transport = mass * masked_data.wa_phy # kg m s-1 kg-1
        cmf = transport.mean(('lat','lon')) * RHO * area # kg s-1

        return cmf
    
    def area_mass_flux_vary_rho(self, masked_data, rho, quantity=None):
        ''' for varyiable density. '''

        # calculate
        area = (masked_data.wa_phy>0).sum(('lat','lon'), skipna=True) * self.grid_area # m2
        mass = self.get_mass(masked_data, quantity) # kg kg-1
        transport = mass * masked_data.wa_phy * rho # kg s-1 m-2
        cmf = transport.mean(('lat','lon')) * area # kg s-1

        return cmf