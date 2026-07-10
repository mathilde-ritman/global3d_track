'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
from datetime import datetime
from . import calculations as calcs
from ..utils import definitions
import metpy


'''
Calculate the statistics used in the analyses

'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GRLStats:

    def __init__(self):

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99

        self.CMF = calcs.CMF(grid_spacing=self.grid_spacings)
    
    def define_anvil(self, mask, tw):
        abh = definitions.discover_abh(tw).mean('time') # calculate anvil base height from total condensate profile
        return mask.where(mask.level_full <= abh), abh
    
    def get_density(self, mask, data, name):

        req_vars = ['pfull','ta','hus','cli','clw','qg','qr','qs']
        masked_data = data[req_vars].sel(time=mask.time).where(mask>0)

        # total density
        rho = calcs.density(masked_data) # kg m-3
        rho_mu = rho.mean(('lat','lon'))
        rho_mu.attrs = dict(units='kg m-3', long_name='mean density')

        ds = xr.Dataset({f'{name}_rho': rho_mu})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    def get_geometric(self, mask, data, name, shortname=None):

        masked_data = data[['dzghalf','zg']].where(mask>0).sel(time=mask.time)

        if not shortname:
            shortname = name + '_'

        cell_area = (mask > 0) * (self.grid_spacings**2) # m2
        cell_depth = masked_data.dzghalf # m
        area = cell_area.sum(('lat','lon')) # m2
        depth = masked_data.dzghalf.sum('level_full') # m
        volume = (cell_area * cell_depth).sum(('lat','lon','level_full')) # m3
        cth = masked_data.zg.max(('level_full','lat','lon')) # m
        cbh = masked_data.zg.min(('level_full','lat','lon')) # m
        
        cth.attrs = dict(units='m', long_name=f'geometric {name} top height')
        cbh.attrs = dict(units='m', long_name=f'geometric {name} base height')
        area.attrs = dict(units='m^2', long_name=f'{name} area')
        depth.attrs = dict(units='m', long_name=f'{name} depth')
        volume.attrs = dict(units='m3', long_name=f'{name} volume')

        ds = xr.Dataset({f'{name}_area':area,
                        f'{shortname}th':cth,
                        f'{shortname}bh':cbh,
                        f'{name}_depth':depth,
                        f'{name}_volume': volume,
                })
        
        ds = ds.where(ds>0)

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_condensates(self, mask, data, name):

        req_vars = ['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf','wa_phy']
        masked_data = data[req_vars].sel(time=mask.time).where(mask>0)
        footprint = mask.max('level_full') > 0 # get footprint for precip calculations
        footprint_data = data[['pr']].sel(time=mask.time).where(footprint)

        # xWP(ds, v='cli')
        iwp = calcs.xWP(masked_data, v='cli')
        swp = calcs.xWP(masked_data, v='qs')
        gwp = calcs.xWP(masked_data, v='qg')
        iwp.attrs = dict(units='kg m-2', long_name=f'{name} ice water path')
        swp.attrs = dict(units='kg m-2', long_name=f'{name} snow water path')
        gwp.attrs = dict(units='kg m-2', long_name=f'{name} graupel water path')

        # frozen water path
        fwp, fwc = calcs.IWP(masked_data, return_iwc=True)
        fwc = fwc.mean(('lat','lon'))
        fwp.attrs = dict(units='kg m-2', long_name=f'{name} frozen water path')
        fwc.attrs = dict(units='kg m-3', long_name=f'{name} mean frozen water concentration')

        # total water path
        twp, twc = calcs.TWP(masked_data, return_twc=True)
        tw = twc.sum(('lat','lon')) * self.grid_spacings**2 * masked_data.dzghalf.mean(('lat','lon')) # kg
        twp.attrs = dict(units='kg m-1', long_name=f'{name} total water path')
        tw.attrs = dict(units='kg', long_name=f'{name} total water content')

        # precipitation
        pr = footprint_data.pr # kg m-2 s-1
        pr.attrs = dict(units='kg m-2 s-1', long_name=f'{name} precipitation flux')

        ds = xr.Dataset({f'{name}_fwp': fwp,
                    f'{name}_fwc': fwc,
                    f'{name}_twp': twp,
                    f'{name}_tw': tw,
                    f'{name}_pr': pr,
                    f'{name}_iwp': iwp,
                    f'{name}_swp': swp,
                    f'{name}_gwp': gwp,
            })
        
        ds = ds.where(ds>0)

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    def get_dynamic(self, mask, data, name):

        req_vars = ['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf','wa_phy', 'ua', 'va']
        masked_data = data[req_vars].sel(time=mask.time).where(mask>0)

        # density
        rho = calcs.density(masked_data)
        rho_mu = rho.mean(('lat','lon')) # kg m-3
        rho_mu.attrs = dict(units='kg m-3', long_name=f'{name} mean density')

        # velocity
        w = masked_data.wa_phy # m s-1
        # - column-max
        max_w = w.max('level_full')
        max_w.attrs = dict(units='m s-1', long_name=f'{name} maximum vertical velocity')
        # - area mean
        mean_w = w.mean(('lat','lon'))
        mean_w.attrs = dict(units='m s-1', long_name=f'{name} mean vertical velocity')

        # mass flux
        cmf_mu = self.CMF.mass_flux(w, rho).mean(('lat','lon'))
        cmf_max = self.CMF.mass_flux(w, rho).max(('lat','lon'))
        cmt = self.CMF.mass_transport(w, rho)
        cmf_mu.attrs = dict(units='kg s-1 m-2', long_name=f'{name} mean convective mass flux')
        cmf_max.attrs = dict(units='kg s-1 m-2', long_name=f'{name} maximum convective mass flux')
        cmt.attrs = dict(units='kg s-1', long_name=f'{name} convective mass transport')

        # divergence
        is_valid = (masked_data.lat.size > 2) & (masked_data.lon.size > 2) # skip anvils without enough data points
        if not is_valid:
            div = xr.full_like(masked_data.ua.any(('lat','lon')), np.nan, dtype=float)
        div = metpy.calc.divergence(masked_data.ua, masked_data.va).mean(('lat','lon')) # s-1
        div = div.metpy.dequantify() # remove metpy units for safe write to disk
        div.attrs = dict(units='s-1', long_name=f'{name} horizontal wind divergence')

        ds = xr.Dataset({f'{name}_rho': rho_mu,
                        f'{name}_max_w': max_w,
                        f'{name}_mean_w': mean_w,
                        f'{name}_cmf_mu': cmf_mu,
                        f'{name}_cmf_max': cmf_max,
                        f'{name}_cmt': cmt,
                        f'{name}_div': div,
                })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_environment(self, mask, data, name='environment'):

        # get max anvil footprint
        max_footprint = mask.max(('time','level_full')) > 0 # get max footprint over time and vertical levels
        initial_footprint = mask.isel(time=0).max('level_full') > 0 # get initial cloud mask at time of first detection

        # get inital data at footprint
        masked_data = data.isel(time=0).where(np.logical_and(max_footprint, ~initial_footprint))

        # surface temperature
        ts = masked_data.ts.mean(('lat','lon')) # K
        ts.attrs = dict(units='K', long_name=f'{name} surface temperature')

        # CAPE, CIN, RH
        cape, cin, rh = calcs.cape_cin(masked_data, return_humidity=True)
        cape = cape.mean(('lat','lon')) # J kg-1
        cin = cin.mean(('lat','lon')) # J kg-1
        rh = rh.mean(('lat','lon')) * 100 # %
        cape.attrs = dict(units='J kg-1', long_name=f'{name} CAPE')
        cin.attrs = dict(units='J kg-1', long_name=f'{name} CIN')
        rh.attrs = dict(units='%', long_name=f'{name} relative humidity')

        # winds
        uwind = masked_data.ua.mean(('lat','lon')) # m s-1
        vwind = masked_data.va.mean(('lat','lon')) # m s-1
        uwind.attrs = dict(units='m s-1', long_name=f'{name} zonal wind', direction='eastward')
        vwind.attrs = dict(units='m s-1', long_name=f'{name} meridional wind', direction='northward')

        ds = xr.Dataset({f'{name}_ua': uwind,
                        f'{name}_va': vwind,
                        f'{name}_ts': ts,
                        f'{name}_cape': cape,
                        f'{name}_cin': cin,
                        f'{name}_rh': rh,
                        })
        ds = ds.expand_dims("time")
        
        for v in ds.data_vars:
            ds[v].attrs['source'] = 'values at the time of first detection and averaged over the maximal anvil footprint'
        
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def individual_core(self, core_mask, c, data, name):
        c_mask = core_mask.where(core_mask == c) # mask current core
        core_stats = self.get_geometric(c_mask, data, name)
        core_stats.update(self.get_condensates(c_mask, data, name))
        core_stats.update(self.get_dynamic(c_mask, data, name))
        return core_stats

    def cores(self, core_mask, data, name='core'):

        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset(coords=dict(core=None, time=core_mask.time)).expand_dims('core')
       
        # iterate cores
        cores = dask.array.unique(core_mask.data).compute()
        cores = cores[~np.isnan(cores)]

        # process
        results = []
        for c in cores:
            results.append(self.individual_core(core_mask, c, data, name))

        # collect
        core_stats = xr.concat(results, dim='core')
        core_stats = core_stats.assign_coords({'core':cores})

        return core_stats
    
    def anvil(self, anvil_mask, data, name='anvil'):

        if not (anvil_mask.max() > 0):
            return xr.Dataset(coords=anvil_mask.coords) # there are no results in the mask provided
        
        anvil_stats = self.get_geometric(anvil_mask, data, name)
        anvil_stats.update(self.get_condensates(anvil_mask, data, name))
        anvil_stats.update(self.get_dynamic(anvil_mask, data, name))

        return anvil_stats

    def cloud(self, system_mask, data, name='cloud', shortname='c'):

        if not (system_mask.max() > 0):
            return xr.Dataset(coords=system_mask.coords) # there are no results in the mask provided
        
        cloud_stats = self.get_geometric(system_mask, data, name, shortname=shortname)
        cloud_stats.update(self.get_condensates(system_mask, data, name))
        cloud_stats.update(self.get_density(system_mask, data, name))
        cloud_stats.update(self.get_environment(system_mask, data))

        return cloud_stats

    def get_everything(self, mask, data, ):
            
        core_mask = mask.u_tracks
        system_mask = mask.system

        stats = self.cloud(system_mask, data) # system results
        stats = stats.merge(self.cores(core_mask, data, 'core')) # core results
        anvil_mask, ABH = self.define_anvil(mask.system, stats.cloud_tw) # define anvil
        ABH.attrs = dict(units='dimensionless', long_name='model level used to define the anvil base height')
        stats['ABH'] = ABH # add anvil base height to stats
        stats = stats.merge(self.anvil(anvil_mask, data, 'anvil')) # anvil results

        return stats.fillna(self.NAN)