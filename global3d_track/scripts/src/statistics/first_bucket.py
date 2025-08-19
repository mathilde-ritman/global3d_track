'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from datetime import datetime
import logging
from .cmf import CMF

# general cloud object statistics



class FirstBucket(CMF):

    def __init__(self):

        super().__init__()

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99


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
        cli.attrs = dict(units='kg kg-1', long_name=f'total specific {name} ice content')
        clw.attrs = dict(units='kg kg-1', long_name=f'total specific {name} water content')

        ds = xr.Dataset({f'{name}_clw': clw,
                        f'{name}_cli': cli})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def velocity(self, mask, data, name, dims=('level_full','lat','lon'), shortname=None, func='max'):
            
        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        # velocity
        w = getattr(masked_data.wa_phy, func)(dim=dims)
        w.attrs = dict(units='m s-1', long_name=f'{name} {func} vertical velocity')
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
                        #  'initiation_time':initiation_time,
                        #  'lifetime':lifetime,
                        #  'n_cores':n_cores,
                        })
        
        return ds

    def cloud_top(self, mask, data, name, dims=('level_full','lat','lon'), shortname=None):

        masked_data = data.where(mask>0)
        if not shortname:
            shortname = name[0]

        ctt = masked_data.ta.min(dims)
        ctt.attrs = dict(units='K', long_name=f'{name} minimum temperature')
        olr = masked_data.rlut.min(dims)
        olr.attrs = dict(units='W m-2', long_name=f'{name} OLR')

        ds = xr.Dataset({f'{shortname}tt': ctt,
                        f'{name}_olr': olr})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    
    def mass_flux(self, core_mask, data, name, RHO=1):

        masked_data = data.where(core_mask>0)

        # calculate       
        cmf_air = self.area_mass_flux(masked_data, quantity=None, RHO=RHO) # air flux
        cmf_cl = self.area_mass_flux(masked_data, quantity=('clw','cli'), RHO=RHO) # condensate

        # collect
        ds = xr.Dataset({f'{name}_cmf': cmf_air,
                        f'{name}_cmf_cl': cmf_cl})
        ds[f'{name}_cmf'].attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux')
        ds[f'{name}_cmf_cl'].attrs = dict(units='kg s-1', long_name=f'{name} convective mass flux of condensate')

        return ds

    def core(self, core_mask, data, dims=('level_full','lat','lon'), keep_z=False, RHO=1, overall=False):
        ''' mask: xr.DataArray
        '''

        name = 'core'
        shortname = 'core_'

        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset(coords=dict(core=None, time=core_mask.time)).expand_dims('core').fillna(self.NAN)
        
        # core stats
        # cores = dask.array.unique(core_mask.data).compute()
        cores = np.unique(core_mask.values)
        cores = cores[~np.isnan(cores)]

        # calculate
        cores_at_time = xr.DataArray(0, dims=('time',), coords={'time':core_mask.time}) # init
        li = []
        for c in cores:
            c_mask = core_mask.where(core_mask == c) # mask out one core

            # staitistics
            core_stats = self.geometric(c_mask, data, name, dims=dims, keep_z=False, shortname=shortname)
            # core_stats.update(self.mass_flux(c_mask, data, name, RHO=RHO))

            # horizontally-resolved
            core_stats.update(self.velocity(c_mask, data, name, dims=('level_full',), shortname=shortname, func='max'))

            # vertically-resolved
            core_stats.update(self.velocity(c_mask, data, name, dims=('lat','lon',), shortname=shortname, func='mean'))
            core_stats.update(self.precipitation(c_mask, data, name, dims=('lat','lon',), shortname=shortname))
            core_stats.update(self.condensate(c_mask, data, name, dims=('lat','lon',), shortname=shortname))

            # additions
            # cores_at_time += (c_mask.max(('lat','lon','level_full')) > 0).compute() # count coures at each time
            # core_stats['n_cores'] = cores_at_time
            # core_stats.n_cores.attrs = dict(units='count', long_name='number of cores')
            # core_label = int(c) # label of core
            # core_stats['core_label'] = core_label
            if overall:
                times = c_mask.time[c_mask.max(('lat','lon','level_full')).values > 0]
                core_stats[f'{name}_lifetime'] = (times[-1] - times[0]).dt.seconds.values / 60
                core_stats.core_lifetime.attrs = dict(units='minutes', long_name='core lifetime')
            li.append(core_stats)

        core_stats = xr.concat(li, dim='core')
        core_stats = core_stats.assign_coords({'core':cores})
        return core_stats.fillna(self.NAN)
    
    def anvil(self, anvil_mask, data, dims=('level_full',), keep_z=True):

        name = 'anvil'
        shortname = None

        if not (anvil_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=anvil_mask.coords).fillna(self.NAN)
        
        # staistics
        anvil_stats = self.geometric(anvil_mask, data, name, dims=dims, keep_z=keep_z, shortname=shortname)

        # horizontally-resolved
        anvil_stats.update(self.condensate(anvil_mask, data, name, dims=('level_full',), shortname=shortname))
        anvil_stats.update(self.precipitation(anvil_mask, data, name, dims=('level_full',), shortname=shortname))
        anvil_stats.update(self.cloud_top(anvil_mask, data, name, dims=('level_full',), shortname=shortname))

        # vertically-resolved
        anvil_stats.update(self.condensate(anvil_mask, data, f"{name}_vert", dims=('lat','lon',), shortname=shortname))

        return anvil_stats.fillna(self.NAN)
    
    def cloud(self, system_mask, data, dims=('level_full','lat','lon'), keep_z=False):

        name = 'cloud'
        shortname = None

        if not (system_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=system_mask.coords).fillna(self.NAN)
        
        cloud_stats = self.geometric(system_mask, data, name, dims=dims, keep_z=keep_z, shortname=shortname)
        cloud_stats.update(self.condensate(system_mask, data, name, dims=dims, shortname=shortname))
        cloud_stats.update(self.precipitation(system_mask, data, name, dims=dims, shortname=shortname))
        cloud_stats.update(self.cloud_top(system_mask, data, name, dims=dims, shortname=shortname))

        return cloud_stats.fillna(self.NAN)
    
    def get_everything(self, mask, data, overall=False):
            
        core_mask = mask.u_tracks
        anvil_mask = mask.anvil
        system_mask = mask.system

        stats = self.cloud(system_mask, data)
        stats.update(self.anvil(anvil_mask, data))
        stats.update(self.core(core_mask, data, overall=overall))

        if overall:
            stats.update(self.overall(mask))

        return stats.fillna(self.NAN)
