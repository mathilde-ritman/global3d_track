'''
Mathilde Ritman 2026

'''

import dask
import xarray as xr
import numpy as np
import metpy

# 1. calculate total air density (incl. hydrometeors)

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

# 2. get condensate concentrations and paths

def xWC(ds, v='cli'):
    # total air density
    rho = density(ds) # kg m-3
    # specific mass fractions - mass of quantity per mass of total air
    q_x = ds[v] # kg kg-1
    # density of quantity (e.g., liquid water content)
    rho_x = q_x * rho # kg m-3
    return rho_x

def xWP(ds, v='cli'):
    # density of quantity
    rho_x = xWC(ds, v) # kg m-3
    # total mass per tropospheric column of air (e.g., liquid water path)
    grid_depth = ds.dzghalf # m
    xWP = (rho_x * grid_depth).sel(level_full=slice(23,90)).sum('level_full') # kg m-2
    return xWP

def IWC(ds, return_density=False):
    ''' calculate ice water content from frozen hydrometeors: ice, snow and graupel '''
    # densities
    q_frozen = ds['cli'] + ds['qs'] + ds['qg'] # kg kg-1
    rho = density(ds)
    rho_frozen = rho * q_frozen # kg m-3
    if return_density:
        return rho_frozen, rho
    return rho_frozen

def IWP(ds, return_iwc=False):
    ''' calculate ice water path from frozen hydrometeors: ice, snow and graupel '''
    # densities
    rho_frozen = IWC(ds) # kg m-3
    # iwp
    grid_depth = ds.dzghalf # m
    IWP = (rho_frozen * grid_depth).sel(level_full=slice(23,90)).sum('level_full') # kg m-2
    if return_iwc:
        return IWP, rho_frozen
    return IWP

def TWP(ds, return_twc=False):
    ''' calculate ice water path from frozen hydrometeors: ice, snow and graupel '''
    # densities
    q_condensate = ds['cli'] + ds['qs'] + ds['qg'] + ds['qr'] + ds['clw'] # kg kg-1
    rho_condensate = density(ds) * q_condensate # kg m-3
    # iwp
    grid_depth = ds.dzghalf # m
    TWP = (rho_condensate * grid_depth).sel(level_full=slice(23,90)).sum('level_full') # kg m-2
    if return_twc:
        return TWP, rho_condensate
    return TWP

# 3. calculate model relative humidity

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

# 4. caclulate cloud convective mass flux

class CMF:

    def __init__(self, grid_spacing=11000):
        self.grid_spacing = grid_spacing # m
        self.grid_area = self.grid_spacing**2 # m2

    def get_mass(self, quantities=None):
        ''' Add up contributions from multiple quantities (xr.DataArrays) if needed, or return 1 '''
        if quantities is None:
            return 1
        if isinstance(quantities, (list, tuple)):
            return xr.concat([q for q in quantities], dim='q').sum('q', skipna=True)
        return quantities
    
    def mass_flux(self, velocity, density, quantities=None):
        ''' calculate mass flux (kg s-1 m-2) for a given velocity and density field, and optional quantity (e.g., liquid water content) to weight by. '''
        return self.get_mass(quantities) * velocity * density # kg s-1 m-2
    
    def mass_transport(self, velocity, density, quantities=None):
        ''' calculate total mass transport (kg s-1) for a given velocity and density field, and optional quantity (e.g., liquid water content) to weight by. '''
        area = (velocity>0).sum(('lat','lon'), skipna=True) * self.grid_area # m2
        mass_flux = self.mass_flux(velocity, density, quantities) # kg s-1 m-2
        return mass_flux.mean(('lat','lon')) * area # kg s-1

# 5. CAPE / CIN
    
def cape_cin_per_column(p, T, rh):
    # ensure units
    p = p  * metpy.units.Pa
    T = T * metpy.units.kelvin
    rh = rh * metpy.units.dimensionless
    # calculations
    Td = metpy.calc.dewpoint_from_relative_humidity(T, rh)
    prof = metpy.calc.parcel_profile(p, T[0], Td[0])
    cape, cin = metpy.calc.cape_cin(p, T, Td, prof)
    return cape.magnitude, cin.magnitude

def cape_cin(data, return_humidity=False):

    data = data.sel(level_full=data.level_full[::-1]) # force pressure increasing

    p = data.pfull # pressure
    T = data.ta # temperature
    rh = relative_humidity(data) # relative humidity

    cape, cin = xr.apply_ufunc(
        cape_cin_per_column,
        p,
        T,
        rh,
        input_core_dims=[["level_full"], ["level_full"], ["level_full"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )

    if return_humidity:
        return cape, cin, rh.metpy.dequantify()
    return cape, cin