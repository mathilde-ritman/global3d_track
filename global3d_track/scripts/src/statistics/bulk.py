'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
from datetime import datetime
from . import density


'''
Calculate the statistics used in the analyses

'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BulkStats:

    def __init__(self):

        super().__init__()

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99    

    def get_geometric(self, mask, data, name, shortname=None, keep_zdim=False, dims=('level_full','lat','lon')):

        masked_data = data[['dzghalf','zg']].where(mask>0).sel(time=mask.time)

        if not shortname:
            shortname = name[0] + '_'

        cell_area = (mask > 0) * (self.grid_spacings**2) # m2
        cell_depth = masked_data.dzghalf # m
        area = cell_area.sum(('lat','lon')) # m2
        depth = masked_data.dzghalf.sum('level_full') # m
        volume = (cell_area * cell_depth).sum(('lat','lon','level_full')) # m3
        cth = masked_data.zg.max(dims) # m
        cbh = masked_data.zg.min(dims) # m

        if not keep_zdim:
            area = area.max('level_full') # get extrema
        
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

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    def efficient_convection_results(self, mask, data, name):

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

        ds = xr.Dataset({f'{name}_rho': mean_rho,
                         f'{name}_pres': mean_pres,
                        f'{name}_mean_w': mean_w,
                        f'{name}_area': area,
                        f'{name}_max_w': max_w,
                        })

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def precipitation(self, mask, data, name, dims=('level_full','lat','lon'), shortname=None):

        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        total_precip = masked_data.pr.sum((x for x in dims if x in masked_data.dims))
        total_precip.attrs = dict(units='kg m-2 s-1', long_name=f'{name} total precip flux')

        ds = xr.Dataset({f'{name}_pr': total_precip})

        var_short = ('qs','qg','qr')
        for v in var_short:
            total_precip_type = masked_data[v].sum(dims)
            total_precip_type.attrs = dict(units=masked_data[v].attrs['units'], long_name=f'{name} total {masked_data[v].attrs["long_name"]}')
            ds[f'{name}_{v}'] = total_precip_type

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def cloud_top(self, mask, data, name, dims=('level_full','lat','lon'), shortname=None):

        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        ctt = masked_data.ta.min(dims)
        ctt.attrs = dict(units='K', long_name=f'{name} minimum temperature')

        ds = xr.Dataset({f'{shortname}tt': ctt,})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def _process_single_core(self, core_mask, c, data, name):
        c_mask = core_mask.where(core_mask == c) # mask current core
        # staitistics
        core_stats = self.efficient_convection_results(c_mask, data, name)
        core_stats.update(self.get_geometric(c_mask, data, name))
        core_stats.update(self.precipitation(c_mask, data, name, dims=('lat','lon',)))
        # collect
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

        shortname = 'a'

        if not (anvil_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=anvil_mask.coords).fillna(self.NAN)
        
        anvil_stats = self.efficient_convection_results(anvil_mask, data, name)
        anvil_stats.update(self.get_geometric(anvil_mask, data, name))
        anvil_stats.update(self.cloud_top(anvil_mask, data, name, dims=('level_full',), shortname=shortname))

        return anvil_stats.fillna(self.NAN)

    def cloud(self, system_mask, data, dims=('level_full','lat','lon'), keep_z=False):

        name = 'cloud'
        shortname = None

        if not (system_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=system_mask.coords).fillna(self.NAN)
        
        cloud_stats = self.get_geometric(system_mask, data, name, dims=dims, keep_z=keep_z, shortname=shortname)
        cloud_stats.update(self.precipitation(system_mask, data, name, dims=dims, shortname=shortname))
        cloud_stats.update(self.cloud_top(system_mask, data, name, dims=dims, shortname=shortname))

        return cloud_stats.fillna(self.NAN)

    def get_everything(self, mask, data, ):
            
        core_mask = mask.u_tracks
        anvil_mask = mask.anvil
        system_mask = mask.system

        # system  results
        stats = self.cloud(system_mask, data)
        stats = stats.merge(self.anvil(anvil_mask, data, 'anvil'))
        stats = stats.merge(self.core(core_mask, data, 'core'))

        return stats.fillna(self.NAN)