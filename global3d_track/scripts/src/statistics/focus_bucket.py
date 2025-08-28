'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
from datetime import datetime
# from .cmf import CMF
from .calculations import relative_humidity, density, CMF


'''
more cloud object statistics: this time with a focus on key results wanter to assess the properties of the anvils, and the properties of the cores.

'''

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FocusBucketAC(CMF):

    def __init__(self):

        super().__init__()

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99


    def get_iwp(self, mask, data, name, ):
        '''  Ice water path calculation '''
    
        masked_data = data[['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf']].sel(time=mask.time).where(mask>0)

        IWP = density.calculate_xWP(masked_data, 'cli')
        IWP.attrs = dict(units='kg m-2', long_name=f'{name} ice water path')

        ds = xr.Dataset({f'{name}_iwp': IWP})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    

    def get_geometric(self, mask, data, name, ):

        masked_data = data[['dzghalf']].where(mask>0).sel(time=mask.time)

        depth = masked_data.dzghalf.sum('level_full') # m
        cell_area = (mask > 0) * (self.grid_spacings**2) # m2
        cell_depth = masked_data.dzghalf
        volume = (cell_area * cell_depth).sum(('lat','lon','level_full')) # m3
        
        depth.attrs = dict(units='m', long_name=f'{name} depth')
        volume.attrs = dict(units='m3', long_name=f'{name} volume')

        ds = xr.Dataset({f'{name}_depth': depth,
                        f'{name}_volume': volume,
                        })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)


    def efficient_convection_results(self, mask, data, name):
        ''' Combo function to get results from both 'conevction_extremes' and 'get_density_cmf' functions without double-up on computations. '''

        req_vars = ['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf','wa_phy']
        masked_data = data[req_vars].sel(time=mask.time).where(mask>0)

        # density
        rho = density(masked_data) # kg m-3
        mean_rho = rho.mean(('lat','lon'))
        mean_rho.attrs = dict(units='kg m-3', long_name=f'{name} mean density')

        # pressure
        mean_pres = masked_data.pfull.mean(('lat','lon')) # kg m-3
        mean_pres.attrs = dict(units='Pa', long_name=f'{name} mean pressure')

        # velocity
        w = masked_data.wa_phy # m s-1
        # 1 - column-max
        max_w = w.max('level_full')
        max_w.attrs = dict(units='m s-1', long_name=f'{name} maximum column verticl velocity')
        # 2 - area mean
        mean_w = w.mean(('lat','lon'))
        mean_w.attrs = dict(units='m s-1', long_name=f'{name} mean vertical velocity')

        # area
        area = (mask > 0).sum(('lat','lon')) * (self.grid_spacings**2) # m2
        area.attrs = dict(units='m2', long_name=f'{name} area')

        # CMF air
        cmf_air = CMF().mass_flux(masked_data, rho=rho, quantity=None) # by pixel
        # 1 - column-wise
        cmf_air_column_agg = cmf_air.sum('level_full') # kg m-2 s-1
        cmf_air_column_agg.attrs = dict(units='kg s-1 m-2', long_name=f'{name} total column convective mass flux (air)')
        # 2 - area-wise
        area_cmf_air = cmf_air.mean(('lat','lon')) * area # kg s-1
        area_cmf_air.attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux (air)')

        # CMF condensate
        cmf_cl = CMF().mass_flux(masked_data, rho=rho, quantity=('cli','clw')) # by pixel
        # 1 - column-wise
        cmf_cl_column_agg = cmf_cl.sum('level_full') # kg m-2 s-1
        cmf_cl_column_agg.attrs = dict(units='kg s-1 m-2', long_name=f'{name} total column convective mass flux (condensate)')
        # 2 - area-wise
        area_cmf_cl = cmf_cl.mean(('lat','lon')) * area # kg s-1
        area_cmf_cl.attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux (total condensate)')

        ds = xr.Dataset({f'{name}_rho': mean_rho,
                         f'{name}_pres': mean_pres,
                        f'{name}_mean_w': mean_w,
                        f'{name}_area': area,
                        f'{name}_cmf_air': area_cmf_air,
                        f'{name}_cmf_cl': area_cmf_cl,
                        f'{name}_max_w': max_w,
                        f'{name}_column_cmf_air': cmf_air_column_agg,
                        f'{name}_column_cmf_cl': cmf_cl_column_agg,
                        })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    

    def convection_extremes(self, mask, data, name):
        ''' Spatially-resolved convection extremes for later determining locations of core centres. '''

        masked_data = data.sel(time=mask.time).where(mask>0)

        # density
        rho = density(masked_data) # kg m-3

        # CMF air
        cmf_air = CMF().mass_flux(masked_data, rho=rho, quantity=None)
        cmf_air_agg = cmf_air.sum('level_full')
        cmf_air_agg.attrs = dict(units='kg s-1 m-2', long_name=f'{name} total column convective mass flux (air)')
        
        # CMF condensate
        cmf_cl = CMF().mass_flux(masked_data, rho=rho, quantity=('cli','clw'))
        cmf_cl_agg = cmf_cl.sum('level_full')
        cmf_cl_agg.attrs = dict(units='kg s-1 m-2', long_name=f'{name} total column convective mass flux (condensate)')

        # velocity
        w = masked_data.wa_phy
        w_agg = w.max('level_full')
        w_agg.attrs = dict(units='m s-1', long_name=f'{name} maximum column verticl velocity')

        ds = xr.Dataset({f'{name}_column_cmf_air': cmf_air_agg,
                        f'{name}_column_cmf_cl': cmf_cl_agg,
                        f'{name}_max_w': w_agg,
                        })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_density_cmf(self, mask, data, name):
        ''' Differs frm previous CMF calulcations as it uses the model density, rather than a constant density. Will also retain parameters rho, w used in CMF calculations. '''

        req_vars = ['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf','wa_phy']
        masked_data = data[req_vars].sel(time=mask.time).where(mask>0)

        # density
        rho = density(masked_data) # kg m-3
        mean_rho = rho.mean(('lat','lon'))
        mean_rho.attrs = dict(units='kg m-3', long_name=f'{name} mean density')

        # velocity
        mean_w = masked_data.wa_phy.mean(('lat','lon')) # m s-1
        mean_w.attrs = dict(units='m s-1', long_name=f'{name} mean vertical velocity')

        # area
        area = (mask > 0).sum(('lat','lon')) * (self.grid_spacings**2) # m2
        area.attrs = dict(units='m2', long_name=f'{name} area')

        # CMF air
        cmf_air = CMF().area_mass_flux_vary_rho(masked_data, rho, quantity=None)
        cmf_air.attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux (air)')

        # CMF condensate
        cmf_cl = CMF().area_mass_flux_vary_rho(masked_data, rho, quantity=('cli','clw'))
        cmf_cl.attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux (total condensate)')

        ds = xr.Dataset({f'{name}_rho': mean_rho,
                        f'{name}_w': mean_w,
                        f'{name}_area': area,
                        f'{name}_cmf_air': cmf_air,
                        f'{name}_cmf_cl': cmf_cl,
                        })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    

    def get_preconditions(self, mask, data, name):
        ''' Get mean surface temperature and low level relative humidity in the footprint of the mask. '''

        # mask data to max spatial extent of mask
        mask_footprint = mask.max(('time','level_full'))
        masked_data = data[['ts','hus','pfull','ta']].where(mask_footprint>0)

        # surface temperature
        ts = masked_data.ts.mean(('lat','lon'))

        # humidty
        low_levels = masked_data.sel(level_full=slice(80,90)) # take low level atmosphere only
        rh = relative_humidity.relative_humidity(low_levels).mean(('level_full','lat','lon')) # take low level atmosphere only

        # attrs
        ts.attrs = dict(units='K', long_name=f'{name} surface temperature')
        rh.attrs = dict(units='dimensionless', long_name=f'{name} relative humidity')

        ds = xr.Dataset({f'{name}_ts': ts,
                        f'{name}_rh': rh,})
                
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    

    def winds_at_mask(self, mask, data, name):
        ''' mean u and v winds at the within the anvil area and at height.'''

        # get winds at the objects spatial extent but all vertical levels it spans over time
        anvil_levels = mask.max(('time','lat','lon'))
        anvil_footprint = mask.max('level_full')
        masked_data = data[['ua','va']].where(np.logical_and(anvil_levels>0, anvil_footprint>0))

        # winds
        mean_uwind = masked_data.ua.mean(('level_full','lat','lon')) # m s-1
        mean_vwind = masked_data.va.mean(('level_full','lat','lon')) # m s-1
        mean_uwind.attrs = dict(units='m s-1', long_name=f'{name} mean zonal wind', direction='eastward')
        mean_vwind.attrs = dict(units='m s-1', long_name=f'{name} mean meridional wind', direction='northward')

        ds = xr.Dataset({f'{name}_ua': mean_uwind,
                        f'{name}_va': mean_vwind,})
        
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def _process_single_core(self, core_mask, c, data, name):
        c_mask = core_mask.where(core_mask == c) # mask current core
        # staitistics
        core_stats = self.efficient_convection_results(c_mask, data, name)
        core_stats.update(self.get_geometric(c_mask, data, name))
        core_stats = core_stats.merge(self.get_preconditions(c_mask, data, name)) # merge to keep new times
        # collect
        return core_stats

    def core(self, core_mask, data, name):

        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset(coords=dict(core=None, time=core_mask.time)).expand_dims('core').fillna(self.NAN)
       
        # iterate cores
        # cores = np.unique(core_mask.values)
        cores = dask.array.unique(core_mask.data).compute()
        cores = cores[~np.isnan(cores)]

        # process
        tasks = []
        for c in cores:
            task = delayed(self._process_single_core)(core_mask, c, data, name)
            tasks.append(task)

        # collect
        core_stats = xr.concat(compute(*tasks), dim='core')
        core_stats = core_stats.assign_coords({'core':cores})
        return core_stats.fillna(self.NAN)
    
    def anvil(self, anvil_mask, data, name='anvil'):

        if not (anvil_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=anvil_mask.coords).fillna(self.NAN)
        
        anvil_stats = self.get_iwp(anvil_mask, data, name)
        anvil_stats.update(self.get_geometric(anvil_mask, data, name))
        anvil_stats.update(self.get_density_cmf(anvil_mask, data, name))
        anvil_stats.update(self.winds_at_mask(anvil_mask, data, name))

        return anvil_stats.fillna(self.NAN)


    def get_everything(self, mask, data, ):
            
        core_mask = mask.u_tracks
        anvil_mask = mask.anvil

        # anvil results
        stats = self.anvil(anvil_mask, data, 'anvil')
            
        # core results
        stats = stats.merge(self.core(core_mask, data, 'core'))

        return stats.fillna(self.NAN)