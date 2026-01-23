'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from typing import Union

# funcs to process different mass fluxes


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
