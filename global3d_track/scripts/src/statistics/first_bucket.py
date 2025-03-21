'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
import logging
from typing import Union

# general cloud object statistics



class FirstBucket:

    def __init__(self):

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s


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
    
    def condensate(self, mask, data, name, dims=('level_full','lat','lon'), shortname=None):

        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        cli = masked_data.cli.sum(dims)
        clw = masked_data.clw.sum(dims)
        cli.attrs = dict(units='kg kg-1', long_name=f'specific {name} ice content')
        clw.attrs = dict(units='kg kg-1', long_name=f'specific {name} water content')

        ds = xr.Dataset({f'{name}_clw': clw,
                        f'{name}_cli': cli})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def velocity(self, mask, data, name, dims=('level_full','lat','lon'), keep_z=False, RHO=1, shortname=None):
            
        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        # velocity
        w = masked_data.wa_phy.max(dims)
        w.attrs = dict(units='m s-1', long_name=f'{name} maximim vertical velocity')
        ds = xr.Dataset({f'{name}_w': w,})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def geometric(self, mask, data, name, dims=('level_full','lat','lon'), keep_z=False, shortname=None):

        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]
        
        # geometric
        area = (mask>0).sum(('lat','lon')) * self.grid_spacings**2
        if not keep_z:
            area = area.max('level_full')
        cth = masked_data.zg.max(dims)
        cbh = masked_data.zg.min(dims)
        depth = cth - cbh
        cth.attrs = dict(units='m', long_name=f'geometric {name} top height')
        cbh.attrs = dict(units='m', long_name=f'geometric {name} base height')
        area.attrs = dict(units='m^2', long_name=f'{name} area')
        depth.attrs = dict(units='m', long_name=f'{name} depth')

        ds = xr.Dataset({f'{name}_area':area,
                        f'{shortname}th':cth,
                        f'{shortname}bh':cbh,
                        f'{name}_depth':depth,
                })
        
        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def overall(self, mask):

        # number of cores
        n_cores = np.unique(mask.u_tracks).size
        n_cores = xr.DataArray(n_cores)
        n_cores.attrs = dict(units='count', long_name='number of cores')

        # lifetime
        lifetime = mask.time[-1] - mask.time[0]
        lifetime = xr.DataArray(lifetime.dt.seconds.values / 60)
        lifetime.attrs = dict(units='minutes', long_name='lifetime')

        # initiation time
        initiation_time = mask.time[0].values ## to local time ??
        initiation_time = xr.DataArray(initiation_time)
        initiation_time.attrs = dict(units='UTC', long_name='initiation time')

        # central lon/lat
        central_lat = mask.lat.median()
        central_lat.attrs = mask.lat.attrs
        central_lon = mask.lon.median()
        central_lon.attrs = mask.lon.attrs

        ds = xr.Dataset({'central_lon': central_lon,
                         'central_lat': central_lat,
                         'initiation_time':initiation_time,
                         'lifetime':lifetime,
                         'n_cores':n_cores,
                        })
        
        return ds

    def cloud_top(self, mask, data, name, dims=('level_full','lat','lon'), shortname=None):

        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        ctt = masked_data.ta.min(dims)
        ctt.attrs = dict(units='K', long_name=f'{name} minimum temperature')
        olr = masked_data.rlut.min(dims)
        olr.attrs = dict(units='K', long_name=f'{name} OLR')

        ds = xr.Dataset({f'{shortname}tt': ctt,
                        f'{name}_olr': olr})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def mass_flux_at_level(self, masked_data, level, quantity=None, RHO=1, name='', drop_levels=False):
        # index core mask at level
        try:
            mask_at_level = masked_data.sel(level_full=level)
        except KeyError:
            # there are no cores that reach the level required at these times, so return nan shaped like expected output
            # logging.warning(f'level {level.values} not found in core data\ncore levels = {masked_data.level_full.values}')
            ds = xr.DataArray(np.nan, dims=('time',), coords={'time':masked_data.time}).to_dataset(name=name)
            if not drop_levels:
                ds[name+'_level'] = level
            return ds
        # caluclate core area at level
        area_at_level = (mask_at_level.wa_phy>0).sum(('lat','lon')) * self.grid_spacings**2
        # calculate transport in each grid cell
        if isinstance(quantity, Union[list, tuple]):
            amount_at_level = sum([mask_at_level[q] for q in quantity])
        elif isinstance(quantity, str):
            amount_at_level = mask_at_level[quantity]
        else:
            amount_at_level = 0 # no quantity
        # aggregate
        transport_at_level = (mask_at_level.wa_phy * amount_at_level).mean(('lat','lon'))
        # finally, calculate the mass flux
        CMF = transport_at_level * area_at_level * RHO
        # record model level at which it was calculated
        ds = CMF.to_dataset(name=name)
        ds = ds.reset_coords('level_full').rename({'level_full':name + '_level'})
        ds[name+'_level'].attrs = dict(long_name=f'model level at which {name} CMF was calculated')
        if drop_levels:
            ds = ds.drop_vars(name+'_level')
        return ds
     
    def mass_flux(self, core_mask, anvil_mask, data, name, RHO=1, shortname=None):
        ''' calculate the mass flux at specific levels. '''

        masked_data = data.where(core_mask>0)
        if not shortname:
            shortname = name[0]

        # levels to calculate mass flux at
        di = {}
        
        # 1 - anvil base
        anvil_base_heights = anvil_mask.max(('lat','lon')).idxmin('level_full')
        level_below_anvil = anvil_base_heights + 1
        di['entry'] = (level_below_anvil, f'{name} convective mass flux at anvil base')

        # 2 - approx. pressure level
        pressure_by_level = data.pfull.mean(('lat','lon','time'))
        level_approx_P500 = np.abs(pressure_by_level - 500).idxmin('level_full')
        di['P500'] = (level_approx_P500, f'{name} convective mass flux at model level to closest 500 hPa')

        # 3 - top of core
        top_of_core = core_mask.idxmin('level_full')
        di['top'] = (top_of_core, f'{name} convective mass flux at core top')

        # 4 - CMF at all levels 
        

        # compute
        ds = xr.Dataset()
        for opt, tup in di.items():
            level, long_name = tup
            cmf = self.mass_flux_at_level(masked_data, level, RHO=RHO, name=f'{name}_cmf_{opt}')
            cmf_cl = self.mass_flux_at_level(masked_data, level, quantity=['clw','cli'], RHO=RHO, name=f'{name}_cmf_cl_{opt}', drop_levels=True)
            ds = xr.merge((ds, cmf, cmf_cl))
            ds[f'{name}_cmf_{opt}'].attrs = dict(units='kg s-1', long_name=long_name)
            ds[f'{name}_cmf_cl_{opt}'].attrs = dict(units='kg s-1', long_name=long_name.replace('convective', 'condensate'))

        return ds

    def core(self, core_mask, data, anvil_mask, dims=('level_full','lat','lon'), keep_z=False, RHO=1, overall=False):
        ''' mask: xr.DataArray
        '''

        name = 'core'
        shortname = 'core_'

        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset()
        
        # core stats
        cores = np.unique(core_mask.values)
        cores = cores[~np.isnan(cores)]
        li = []
        for c in cores:
            c_mask = core_mask.where(core_mask == c) # mask out one core

            # staitistics
            core_stats = self.geometric(c_mask, data, name, dims=dims, keep_z=False, shortname=shortname)
            core_stats.update(self.condensate(c_mask, data, name, dims=dims, shortname=shortname))
            core_stats.update(self.precipitation(c_mask, data, name, dims=dims, shortname=shortname))
            core_stats.update(self.velocity(c_mask, data, name, dims=dims, keep_z=keep_z, RHO=RHO, shortname=shortname))
            core_stats.update(self.mass_flux(anvil_mask, c_mask, data, name, RHO=RHO, shortname=shortname))

            # additions
            core_stats[f'{name}_label'] = c
            if overall:
                times = c_mask.time[c_mask.max(('lat','lon','level_full')).values > 0]
                core_stats[f'{name}_lifetime'] = (times[-1] - times[0]).dt.seconds.values / 60
                core_stats.core_lifetime.attrs = dict(units='minutes', long_name='core lifetime')
            li.append(core_stats)

        core_stats = xr.concat(li, dim='core')
        core_stats = core_stats.assign_coords({'core':cores})
        return core_stats
    
    def anvil(self, anvil_mask, data, dims=('level_full'), keep_z=True):

        name = 'anvil'
        shortname = None

        if not (anvil_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset()
        
        anvil_stats = self.geometric(anvil_mask, data, name, dims=dims, keep_z=keep_z, shortname=shortname)
        anvil_stats.update(self.condensate(anvil_mask, data, name, dims=dims, shortname=shortname))
        anvil_stats.update(self.precipitation(anvil_mask, data, name, dims=dims, shortname=shortname))
        anvil_stats.update(self.cloud_top(anvil_mask, data, name, dims=dims, shortname=shortname))

        return anvil_stats
    
    def cloud(self, system_mask, data, dims=('level_full','lat','lon'), keep_z=False):

        name = 'cloud'
        shortname = None

        if not (system_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset()
        
        cloud_stats = self.geometric(system_mask, data, name, dims=dims, keep_z=keep_z, shortname=shortname)
        cloud_stats.update(self.condensate(system_mask, data, name, dims=dims, shortname=shortname))
        cloud_stats.update(self.precipitation(system_mask, data, name, dims=dims, shortname=shortname))
        cloud_stats.update(self.cloud_top(system_mask, data, name, dims=dims, shortname=shortname))

        return cloud_stats
    
    def get_everything(self, mask, data, overall=False):
            
        core_mask = mask.u_tracks
        anvil_mask = mask.anvil
        system_mask = mask.system

        stats = self.cloud(system_mask, data)
        stats.update(self.anvil(anvil_mask, data))
        stats.update(self.core(core_mask, data, anvil_mask, overall=overall))

        if overall:
            stats.update(self.overall(mask))

        return stats
