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

    # def mass_flux_at_level(self, masked_data, level, quantity=None, RHO=1, name='', drop_levels=False):
    #     # index core mask at level
    #     try:
    #         mask_at_level = masked_data.sel(level_full=level)
    #     except KeyError:
    #         # there are no cores that reach the level required at these times, so return nan shaped like expected output
    #         # logging.warning(f'level {level.values} not found in core data\ncore levels = {masked_data.level_full.values}')
    #         ds = xr.DataArray(np.nan, dims=('time',), coords={'time':masked_data.time}).to_dataset(name=name)
    #         if not drop_levels:
    #             ds[name+'_level'] = level
    #         return ds
    #     # caluclate core area at level
    #     area_at_level = (mask_at_level.wa_phy>0).sum(('lat','lon')) * self.grid_spacings**2
    #     # calculate transport in each grid cell
    #     if isinstance(quantity, Union[list, tuple]):
    #         amount_at_level = sum([mask_at_level[q] for q in quantity])
    #     elif isinstance(quantity, str):
    #         amount_at_level = mask_at_level[quantity]
    #     else:
    #         amount_at_level = 0 # no quantity
    #     # aggregate
    #     transport_at_level = (mask_at_level.wa_phy * amount_at_level).mean(('lat','lon'))
    #     # finally, calculate the mass flux
    #     CMF = transport_at_level * area_at_level * RHO
    #     # record model level at which it was calculated
    #     ds = CMF.to_dataset(name=name)
    #     ds = ds.reset_coords('level_full').rename({'level_full':name + '_level'})
    #     ds[name+'_level'].attrs = dict(long_name=f'model level at which {name} CMF was calculated')
    #     if drop_levels:
    #         ds = ds.drop_vars(name+'_level')
    #     return ds
    

    # def mass_flux(self, core_mask, anvil_mask, data, name, RHO=1, shortname=None):
    #     ''' calculate the mass flux at specific levels. '''

    #     masked_data = data.where(core_mask>0)
    #     if not shortname:
    #         shortname = name[0]

    #     # levels to calculate mass flux at
    #     di = {}
        
    #     # 1 - anvil base
    #     anvil_base_heights = anvil_mask.max(('lat','lon')).idxmin('level_full')
    #     level_below_anvil = anvil_base_heights + 1
    #     di['entry'] = (level_below_anvil, f'{name} convective mass flux at anvil base')

    #     # 2 - approx. pressure level
    #     pressure_by_level = data.pfull.mean(('lat','lon','time'))
    #     level_approx_P500 = np.abs(pressure_by_level - 500).idxmin('level_full')
    #     di['P500'] = (level_approx_P500, f'{name} convective mass flux at model level to closest 500 hPa')

    #     # 3 - top of core
    #     top_of_core = core_mask.idxmin('level_full')
    #     di['top'] = (top_of_core, f'{name} convective mass flux at core top')

    #     # compute
    #     ds = xr.Dataset()
    #     for opt, tup in di.items():
    #         level, long_name = tup
    #         cmf = self.mass_flux_at_level(masked_data, level, RHO=RHO, name=f'{name}_cmf_{opt}')
    #         cmf_cl = self.mass_flux_at_level(masked_data, level, quantity=['clw','cli'], RHO=RHO, name=f'{name}_cmf_cl_{opt}', drop_levels=True)
    #         ds = xr.merge((ds, cmf, cmf_cl))
    #         ds[f'{name}_cmf_{opt}'].attrs = dict(units='kg s-1', long_name=long_name)
    #         ds[f'{name}_cmf_cl_{opt}'].attrs = dict(units='kg s-1', long_name=long_name.replace('convective', 'condensate'))

    #     return ds