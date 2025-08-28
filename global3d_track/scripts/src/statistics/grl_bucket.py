'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
logging.basicConfig(level=logging.INFO)
import metpy
from datetime import datetime
from . import calculations
CMF = calculations.CMF
''' 


'''


class GRLBucket(CMF):

    def __init__(self):

        super().__init__()

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99

    def get_geometric(self, mask, data, name, dims=('level_full','lat','lon')):
        ''' Geometric calculation '''

        masked_data = data[['zg','dzghalf','ta','rlut','ta']].sel(time=mask.time).where(mask>0)

        # area (time)
        area = (mask>0).sum(('lat','lon')) * self.grid_spacings**2
        area.attrs = dict(units='m^2', long_name=f'{name} area')
        # top, bottom height (time)
        cth = masked_data.zg.max(dims)
        cbh = masked_data.zg.min(dims)
        cth.attrs = dict(units='m', long_name=f'geometric {name} top height')
        cbh.attrs = dict(units='m', long_name=f'geometric {name} base height')
        # depth, volume (time, space)
        depth = masked_data.dzghalf.sum(dims) # m
        cell_area = (mask > 0) * (self.grid_spacings**2) # m2
        cell_depth = masked_data.dzghalf
        volume = (cell_area * cell_depth).sum(('lat','lon','level_full')) # m3
        area.attrs = dict(units='m^2', long_name=f'{name} area')
        depth.attrs = dict(units='m', long_name=f'{name} depth')
        # top temperature, OLR
        top_T = masked_data.ta.min('level_full').mean(('lat','lon')) # K
        OLR = masked_data.rlut.mean(('lat','lon')) # W m-2
        top_T.attrs = dict(units='K', long_name=f'{name} mean top temperature')
        OLR.attrs = dict(units='W m-2', long_name=f'{name} mean OLR')

        ds = xr.Dataset({f'{name}_area':area,
                         f'{name}_th':cth,
                         f'{name}_bh':cbh,
                         f'{name}_depth':depth,
                         f'{name}_volume':volume,
                         f'{name}_tt':top_T,
                         f'{name}_olr':OLR,
                })
        
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    def get_iwp(self, mask, data, name, ):
        '''  Frozen & Ice water path, and related calculation '''
    
        masked_data = data[['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf']].sel(time=mask.time).where(mask>0)

        # frozen/ice water path
        FWP, rho_F = calculations.calculate_IWP(masked_data, verbose=1)
        FWP.attrs = dict(units='kg m-2', long_name=f'{name} frozen water path')
        IWP = calculations.calculate_xWP(masked_data, 'cli')
        IWP.attrs = dict(units='kg m-2', long_name=f'{name} ice water path')
        # related
        rho_F = rho_F.mean(('lat','lon','level_full'))
        rho_F.attrs = dict(units='kg m-3', long_name=f'{name} frozen water density')
        rho_I = calculations.calculate_xWC(masked_data, v='cli').mean(('lat','lon','level_full'))
        rho_I.attrs = dict(units='kg m-3', long_name=f'{name} ice water density')

        ds = xr.Dataset({f'{name}_frozenwp': FWP,
                         f'{name}_iwp': IWP,
                         f'{name}_rho_frozen': rho_F,
                         f'{name}_rho_ice': rho_I,
                         })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_divergence(self, mask, data, name):
        ''' Divergence calculation '''

        masked_data = data[['ua','va']].sel(time=mask.time).where(mask>0)

        # horizontal wind divergence
        div = metpy.calc.divergence(masked_data.ua, masked_data.va).mean(('lat','lon')) # s-1
        div = div.metpy.dequantify() # remove units for safe write to disk
        div.attrs = dict(units='s-1', long_name=f'{name} horizontal wind divergence')

        ds = xr.Dataset({f'{name}_div': div,
                        })
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_temperature(self, mask, data, name):
        ''' Temperature with height calculation '''

        masked_data = data[['ta']].sel(time=mask.time).where(mask>0)

        # temperature
        mean_t = masked_data.ta.mean(('lat','lon'))
        mean_t.attrs = dict(units='K', long_name=f'{name} mean temperature')
        ds = xr.Dataset({f'{name}_T': mean_t,
                        })
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_precipitation(self, mask, data, name):
        ''' Surface precipitation calculation '''

        masked_data = data[['pr']].sel(time=mask.time).where(mask>0)

        # precipitation
        total_pr = masked_data.pr.sum(('lat','lon')) * (self.grid_spacings**2) # kg s-1
        total_pr.attrs = dict(units='kg s-1', long_name=f'{name} total precipitation flux')
        ds = xr.Dataset({f'{name}_pr': total_pr,
                        })
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_density_cmf(self, mask, data, name):
        ''' Density, convective mass flux, and related calculation '''

        req_vars = ['ts','pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf','wa_phy']
        masked_data = data[req_vars].sel(time=mask.time).where(mask>0)

        # density
        rho = calculations.density(masked_data) # kg m-3
        mean_rho = rho.mean(('lat','lon')) # kg m-3
        rho_frozen = ((masked_data.cli + masked_data.qs + masked_data.qg) * rho).mean(('lat','lon')) # kg m-3
        mean_rho.attrs = dict(units='kg m-3', long_name=f'{name} mean density')
        rho_frozen.attrs = dict(units='kg m-3', long_name=f'{name} mean frozen water density')
        # velocity
        mean_w = masked_data.wa_phy.mean(('lat','lon')) # m s-1
        mean_w.attrs = dict(units='m s-1', long_name=f'{name} mean vertical velocity')
        # CMF air
        cmf = CMF().area_mass_flux_vary_rho(masked_data, rho, quantity=None)
        cmf.attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux (air)')
        # CMF frozen
        cmff = CMF().area_mass_flux_vary_rho(masked_data, rho, quantity=['cli','qg','qs'])
        cmff.attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux (air+frozen)')
        # area
        area = (mask > 0).sum(('lat','lon')) * (self.grid_spacings**2) # m2
        area.attrs = dict(units='m2', long_name=f'{name} area')
        # temperature, humidity
        surf_t = masked_data.ts.mean(('lat','lon')) # K
        mean_t = masked_data.ta.mean(('lat','lon')) # K
        rh = calculations.relative_humidity(masked_data).mean(('lat','lon')) # %
        mean_t.attrs = dict(units='K', long_name=f'{name} mean temperature')
        rh.attrs = dict(units='', long_name=f'{name} mean relative humidity')

        ds = xr.Dataset({f'{name}_rho': mean_rho,
                        f'{name}_rho_frozen': rho_frozen,
                        f'{name}_w': mean_w,
                        f'{name}_cmf': cmf,
                        f'{name}_cmff': cmff,
                        f'{name}_area': area,
                        f'{name}_T_surf': surf_t,
                        f'{name}_T': mean_t,
                        f'{name}_RH': rh,
                        })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    
    def _process_single_core(self, core_mask, c, data, name):
        c_mask = core_mask.where(core_mask == c) # mask current core
        core_stats = self.get_density_cmf(c_mask, data, name)
        return core_stats

    def core(self, core_mask, data, name):
        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset(coords=dict(core=None, time=core_mask.time)).expand_dims('core').fillna(self.NAN)
        # iterate cores
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
        anvil_stats = self.get_geometric(anvil_mask, data, name, dims=('level_full'))
        anvil_stats.update(self.get_iwp(anvil_mask, data, name))
        anvil_stats.update(self.get_divergence(anvil_mask, data, name))
        return anvil_stats.fillna(self.NAN)
    
    def cloud(self, cloud_mask, data, name='cloud'):
        if not (cloud_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=cloud_mask.coords).fillna(self.NAN)
        cloud_stats = self.get_geometric(cloud_mask, data, name)
        cloud_stats.update(self.get_iwp(cloud_mask, data, name))
        cloud_stats.update(self.get_temperature(cloud_mask, data, name))
        cloud_stats.update(self.get_precipitation(cloud_mask, data, name))
        return cloud_stats.fillna(self.NAN)

    def get_everything(self, mask, data, ):
        
        core_mask = mask.u_tracks
        anvil_mask = mask.anvil
        cloud_mask = mask.system

        # anvil, core, cloud results
        stats = self.anvil(anvil_mask, data, 'anvil')
        stats = stats.merge(self.core(core_mask, data, 'core'))
        stats = stats.merge(self.cloud(cloud_mask, data, 'cloud'))

        return stats.fillna(self.NAN)