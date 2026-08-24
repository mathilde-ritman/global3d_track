'''
Mathilde Ritman 2026

'''

import dask
import xarray as xr
import numpy as np
import metpy
from metpy.units import units
import logging

# 1. calculate total air density (incl. hydrometeors)

def dry_density(ds):
    ''' uses ideal gas law to get dry air density '''
    # thermodynamic variables
    p = ds.pfull # Pa (kg m-1 s-2)
    T = ds.ta # K
    Rd = 287.04 # J kg-1 K-1 (m2 s-2 K-1)
    # dry density
    rho = p / (Rd * T) # kg m-3
    return rho

def moist_air_density(ds, output='10km'):
    ''' total dry and moist air density '''
    p = ds.pfull # Pa (kg m-1 s-2)
    T = ds.ta # K
    Rd = 287.04 # J kg-1 K-1
    Rv = 461.4 # J kg-1 K-1
    e = vapour_pressure(ds) # Pa
    # total dry + moist air density
    rho = (e / (Rv * T)) + ((p - e) / (Rd * T)) # kg m-3
    return rho

def density(ds, output='10km'):
    ''' total air and condensate density '''
    if output == '5km':
        q_d = 1 - ds.hus - ds.qall # kg kg-1
    elif output == '10km':
        q_d = 1 - ds.hus - ds.cli - ds.clw - ds.qg - ds.qr - ds.qs # kg kg-1
    rho = dry_density(ds) / q_d # kg m-3
    return rho

# 2. get condensate concentrations and paths

def xWC(ds, v='cli', output='10km'):
    # total air density
    rho = density(ds, output) # kg m-3
    # specific mass fractions - mass of quantity per mass of total air
    q_x = ds[v] # kg kg-1
    # density of quantity (e.g., liquid water content)
    rho_x = q_x * rho # kg m-3
    return rho_x

def xWP(ds, v='cli', output='10km', zdim='level_full'):
    # density of quantity
    rho_x = xWC(ds, v, output) # kg m-3
    # total mass per tropospheric column of air (e.g., liquid water path)
    grid_depth = ds.dzghalf # m
    xWP = (rho_x * grid_depth).sum(zdim) # kg m-2
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

def IWP(ds, return_iwc=False, zdim='level_full'):
    ''' calculate ice water path from frozen hydrometeors: ice, snow and graupel '''
    # densities
    rho_frozen = IWC(ds) # kg m-3
    # iwp
    grid_depth = ds.dzghalf # m
    IWP = (rho_frozen * grid_depth).sum(zdim) # kg m-2
    if return_iwc:
        return IWP, rho_frozen
    return IWP

def TWC(ds, return_density=False):
    ''' calculate total water content from all hydrometeors: ice, snow, graupel, rain and cloud liquid '''
    q_condensate = ds['cli'] + ds['qs'] + ds['qg'] + ds['qr'] + ds['clw'] # kg kg-1
    rho = density(ds)
    rho_condensate = rho * q_condensate # kg m-3
    if return_density:
        return rho_condensate, rho
    return rho_condensate

def TWP(ds, return_twc=False):
    ''' calculate ice water path from frozen hydrometeors: ice, snow and graupel '''
    # densities
    q_condensate = ds['cli'] + ds['qs'] + ds['qg'] + ds['qr'] + ds['clw'] # kg kg-1
    rho_condensate = density(ds) * q_condensate # kg m-3
    # iwp
    grid_depth = ds.dzghalf # m
    TWP = (rho_condensate * grid_depth).sum('level_full') # kg m-2
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
        mass_flux = self.mass_flux(velocity, density, quantities) # kg s-1 m-2
        return mass_flux.sum(('lat','lon')) * self.grid_area # kg s-1

# 5. CAPE / CIN
    
def cape_cin_per_column(p, T, rh):
    if np.isfinite(p).sum() < 5:
        return np.nan, np.nan

    mask = np.isfinite(p) & np.isfinite(T) & np.isfinite(rh)
    p = p[mask] * units.Pa
    T = T[mask] * units.kelvin
    rh = rh[mask] * units.dimensionless
    Td = metpy.calc.dewpoint_from_relative_humidity(T, rh)

    try:
        prof = metpy.calc.parcel_profile(p, T[0], Td[0])
    except ValueError:
        logging.warning(
            "Parcel profile calculation failed; retrying with pressures above 100 hPa."
        )
        mask = p >= (100 * units.hPa)
        p = p[mask]
        T = T[mask]
        rh = rh[mask]
        Td = metpy.calc.dewpoint_from_relative_humidity(T, rh)
        prof = metpy.calc.parcel_profile(p, T[0], Td[0])

    try:
        cape, cin = metpy.calc.cape_cin(p, T, Td, prof)
        return cape.magnitude, cin.magnitude
    except ValueError:
        return np.nan, np.nan

def profile_quantities_per_column(p, T, Td):
    if np.isfinite(p).sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    p = p * units.Pa
    T = T * units.kelvin
    Td = Td * units.kelvin
    prof = metpy.calc.parcel_profile(p, T[0], Td[0])
    lfc, _ = metpy.calc.lfc(p, T, Td, prof)
    lcl, _ = metpy.calc.lcl(p[0], T[0], Td[0])
    cape, cin = metpy.calc.cape_cin(p, T, Td, prof)
    return cape.magnitude, cin.magnitude, lfc.magnitude, lcl.magnitude

def profile_quantities(p, T, rh, zdim='level_full'):
    ''' returns (CAPE, CIN, LFC, LCL) '''

    # reverse zdim if pressure increases with z
    if p[zdim].diff(zdim).mean() > 0:
        p = p.sel({zdim: p[zdim][::-1]})
        T = T.sel({zdim: T[zdim][::-1]})
        rh = rh.sel({zdim: rh[zdim][::-1]})

    Td = metpy.calc.dewpoint_from_relative_humidity(T * units.kelvin, rh * units.dimensionless)
    Td = xr.DataArray(Td.data.to(units.kelvin), coords=T.coords).metpy.dequantify()
    
    return xr.apply_ufunc(
            profile_quantities_per_column,
            p,
            T,
            Td,
            input_core_dims=[[zdim],[zdim],[zdim]],
            output_core_dims=[[], [], [], []],
            vectorize=True,
        )

def wind_shear(data, z0, z1, zdim='level_full'):
    idx0 = np.abs(data.zg - z0*1e3).idxmin(zdim)
    idx1 = np.abs(data.zg - z1*1e3).idxmin(zdim)
    bottom = data.sel({zdim: idx0})
    top = data.sel({zdim: idx1})
    bulk_shear = np.sqrt((top.ua - bottom.ua)**2 + (top.va - bottom.va)**2)
    return bulk_shear

def dxdy(X, Y, dim="level_full"):
    def gradient_1d(x, y):
        return np.gradient(x, y, edge_order=2)

    return xr.apply_ufunc(
        gradient_1d,
        X,
        Y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.result_type(X.dtype, Y.dtype)],
    )

def static_stability(T, p, zdim='level_full'):
    ''' calculate static stability (T/theta dtheta/dp) in K/Pa '''
    p = p * units.Pa
    T = T * units.kelvin
    theta = metpy.calc.potential_temperature(p, T)
    theta = theta.copy(data=theta.data.magnitude)
    S = (T / theta) * dxdy(theta, p.metpy.dequantify(), zdim)
    return S.metpy.dequantify()


def wmo_tropopause_height(T, zg, zdim="level_full"):

    # make sure vertical coordinate runs from low to high altitude
    zg_mean = zg.mean(dim=[d for d in zg.dims if d != zdim])
    if zg_mean.diff(zdim).mean() < 0:
        T = T.isel({zdim: slice(None, None, -1)})
        zg = zg.isel({zdim: slice(None, None, -1)})

    # levels where lapse rate <= 2 K km-1
    z_km = zg / 1000.0 # convert height to km
    lapse_rate = - dxdy(T, z_km, dim=zdim)
    x = lapse_rate <= 2

    def find_tropopause(z, x):
        candidates = np.flatnonzero(x & ~np.r_[False, x[:-1]]) # first instances
        for c in candidates:
            z0 = z[c]
            z1 = z0 + 2
            mask = np.logical_and(z>=z0, z<=z1)
            lapse_above = x[mask]
            if lapse_above.mean() == 1:
                return z0
        return np.nan

    tropopause = xr.apply_ufunc(
        find_tropopause,
        z_km,
        x,
        input_core_dims=[[zdim], [zdim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    tropopause.attrs = dict(units="km", long_name="tropopause height", definition="WMO lapse-rate tropopause")
    return tropopause