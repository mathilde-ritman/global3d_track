'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import logging
from datetime import datetime
from . import calculations as calcs
from ..utils import definitions
from . import environment
import metpy


'''
Calculate the statistics used in the analyses

'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GRLStats:

    def __init__(self):

        self.dxy = 11000 # m
        self.dz = 300 # m
        self.dt = 900 # s
        self.NAN = -999.99

        self.CMF = calcs.CMF(grid_spacing=self.dxy)
    
    def derive_fields(self, data):
        data.attrs = {}
        for var in data.variables:
            data[var].attrs = {}

        # calculate data fields
        ds = xr.Dataset({})

        # geometric
        da = data.ta
        ds['area'] = (self.dxy**2 * xr.DataArray(np.ones_like(da), dims=da.dims, coords=da.coords)).assign_attrs(units='m2', long_name='area')
        ds['depth'] = data.dzghalf.assign_attrs(units='m', long_name='depth')
        ds['volume'] = (ds.area * ds.depth).assign_attrs(units='m3', long_name='volume')
        ds['zg'] = data.zg.assign_attrs(units='m', long_name='height')

        # environmental
        ds['T'] = data.ta
        ds['p'] = data.pfull

        # condensate
        ds['pr'] = data.pr
        ds['fwc'] = calcs.IWC(data)
        ds['twc'] = calcs.TWC(data)
        
        # dynamic
        ds['rho'] = calcs.density(data)
        ds['w'] = data.wa_phy
        ds['cmf'] = self.CMF.mass_flux(ds.w, ds.rho)
        ds['div'] = metpy.calc.divergence(data.ua, data.va).metpy.dequantify()

        return ds

    def derive_environmental_fields(self, data):
        data.attrs = {}
        for var in data.variables:
            data[var].attrs = {}

        ds = xr.Dataset({})
        ds['zg'] = data.zg
        ds['T'] = data.ta
        ds['p'] = data.pfull
        ds['rh'] = calcs.relative_humidity(data)
        ds['tpwv'] = calcs.xWC(data, 'hus') * self.dxy**2 * data.dzghalf
        ds['low_shear'] = calcs.wind_shear(data, 0, 2, zdim='level_full')
        ds['deep_shear'] = calcs.wind_shear(data, 0, 6, zdim='level_full')
        ds['S'] = calcs.static_stability(ds.T, ds.p, zdim='level_full')
        ds['tropo'] = calcs.wmo_tropopause_height(ds.T, ds.zg, zdim='level_full')
        
        return ds

    def define_anvil(self, mask, ds, level=None, shift=None):

        if level is None:
            # t_max_depth = ds.depth.sum('level_full', min_count=1).max(('lat','lon')).idxmax('time')
            # mature_mask = mask.sel(time=slice(t_max_depth, None))
            # level = mature_mask.level_full.where(mature_mask).max('level_full').mean() # mean CBH
            level = definitions.discover_abh_old(ds.cloud_tw).mean('time')

        elif shift is None:
            pass

        else:
            zg_profile = ds.zg.mean(dim=[d for d in ds.zg.dims if d != 'level_full'])
            target_height = float(zg_profile.interp(level_full=level)) + shift * 1000 # m
            order = np.argsort(zg_profile.values)
            level = xr.DataArray(float(np.interp(target_height, zg_profile.values[order], zg_profile['level_full'].values[order])))

        anvil = mask & (mask.level_full <= level)
        return anvil, level
    
    def environment_calcs(self, ds, name):

        results = xr.Dataset({})
        results[f'{name}_zg'] = ds.zg.mean(('lat','lon')).assign_attrs(units='m', long_name='height')
        results[f'{name}_T'] = ds.T.mean(('lat','lon')).assign_attrs(units='K', long_name='temperature')
        results[f'{name}_p'] = ds.p.mean(('lat','lon')).assign_attrs(units='Pa', long_name='air pressure')
        results[f'{name}_rh'] = ds.rh.mean(('lat','lon')).assign_attrs(units='', long_name='relative humidity')
        results[f'{name}_tpwv'] = ds.tpwv.sum(min_count=1).assign_attrs(units='kg', long_name='total precipitable water vapour')
        results[f'{name}_low_shear'] = ds.low_shear.mean().assign_attrs(units='m s-1', long_name='low-level wind shear (0-2 km)')
        results[f'{name}_deep_shear'] = ds.deep_shear.mean().assign_attrs(units='m s-1', long_name='mid-level wind shear (2-6 km)')
        results[f'{name}_S'] = ds.S.mean(('lat','lon')).assign_attrs(units='K Pa-1', long_name='static stability')
        results[f'{name}_tropo'] = calcs.wmo_tropopause_height(ds.T, ds.zg, zdim='level_full').assign_attrs(units='km', long_name='tropopause height')

        for var in results.data_vars:
            results[var].attrs['description'] = "calculated at the DCC's first time step and over the region corresponding to the maximal cloud extent reached"

        return results

    def geometric_calcs(self, ds, name):

        results = xr.Dataset({})
        results[f"{name}_area"] = ds.area.sum(('lat','lon'), keep_attrs=True, min_count=1).assign_attrs(long_name=f'{name} area')
        results[f"{name}_depth"] = ds.depth.sum('level_full', keep_attrs=True, min_count=1).assign_attrs(long_name=f'{name} depth')
        results[f"{name}_volume"] = ds.volume.sum(('level_full','lat','lon'), keep_attrs=True, min_count=1).assign_attrs(long_name=f'{name} volume')
        results[f"{name}_cth"] = ds.zg.max(('level_full','lat','lon')).assign_attrs(units='m', long_name=f'{name} top height')
        results[f"{name}_cbh"] = ds.zg.min(('level_full','lat','lon')).assign_attrs(units='m', long_name=f'{name} base height')

        return results

    def watery_calcs(self, ds, name):

        results = xr.Dataset({})
        results[f"{name}_fw"] = (ds.fwc * ds.volume).sum(('lat','lon'), min_count=1).assign_attrs(units='kg', long_name=f'{name} total frozen water')
        results[f"{name}_tw"] = (ds.twc * ds.volume).sum(('lat','lon'), min_count=1).assign_attrs(units='kg', long_name=f'{name} total water')
        results[f"{name}_fwp"] = (ds.fwc * ds.depth).sum('level_full', min_count=1).assign_attrs(units='kg m-2', long_name=f'{name} frozen water path')
        results[f"{name}_twp"] = (ds.twc * ds.depth).sum('level_full', min_count=1).assign_attrs(units='kg m-2', long_name=f'{name} total water path')
        results[f"{name}_fwc"] = ds.fwc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean frozen water concentration')
        results[f"{name}_twc"] = ds.twc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean total water concentration')
        results[f"{name}_rho_mu"] = ds.rho.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean density')

        return results

    def cloud_calcs(self, ds, name='cloud'):

        results = xr.Dataset({})
        results = results.merge(self.geometric_calcs(ds, name))
        results = results.merge(self.watery_calcs(ds, name))
        results[f"{name}_cbh_mu"] = ds.zg.min('level_full').mean(('lat','lon')).assign_attrs(units='m', long_name=f'{name} median base height')
        pr = ds.pr.max('level_full', skipna=True) if 'level_full' in ds.pr.dims else ds.pr
        results[f"{name}_pr"] = pr.sum(('lat','lon'), min_count=1).assign_attrs(units='kg m-2 s-1', long_name=f'{name} total precipitation flux')
        results[f"{name}_div_mu"] = ds.div.mean(('lat','lon')).assign_attrs(units='s-1', long_name=f'{name} mean horizontal wind divergence')
        results[f'{name}_pressure'] = ds.p.mean(('lat','lon')).assign_attrs(units='Pa', long_name=f'{name} mean pressure')
        results[f'{name}_tt'] = ds.T.min('level_full').assign_attrs(units='K', long_name=f'{name} minimum temperature')

        return results

    def anvil_calcs(self, ds, name='anvil'):

        results = xr.Dataset({})
        results = results.merge(self.geometric_calcs(ds, name))
        results = results.merge(self.watery_calcs(ds, name))

        # dynamic
        results[f"{name}_div_mu"] = ds.div.mean(('lat','lon')).assign_attrs(units='s-1', long_name=f'{name} mean horizontal wind divergence')
        results[f"{name}_div_max"] = ds.div.max(('lat','lon')).assign_attrs(units='s-1', long_name=f'{name} maximum horizontal wind divergence')

        return results

    def anvil_sensitvity_calcs(self, ds, level, name):

        results = xr.Dataset({})
        results[f"{name}_ABH"] = level.assign_attrs(units='level_full', long_name=f'model level used to define {name} ABH')
        results[f"{name}_area"] = ds.area.sum(('lat','lon'), keep_attrs=True, min_count=1).assign_attrs(long_name=f'{name} area')
        results[f"{name}_fwp"] = (ds.fwc * ds.depth).sum('level_full', min_count=1).assign_attrs(units='kg m-2', long_name=f'{name} frozen water path')
        results[f"{name}_twp"] = (ds.twc * ds.depth).sum('level_full', min_count=1).assign_attrs(units='kg m-2', long_name=f'{name} total water path')
    
        return results
    
    def core_calcs(self, ds, name='core'):

        ds = ds.unstack(ds.dims)
        results = xr.Dataset({})
        results = results.merge(self.geometric_calcs(ds, name))

        # watery
        results[f"{name}_fw"] = (ds.fwc * ds.volume).sum(('lat','lon'), min_count=1).assign_attrs(units='kg', long_name=f'{name} total frozen water')
        results[f"{name}_tw"] = (ds.twc * ds.volume).sum(('lat','lon'), min_count=1).assign_attrs(units='kg', long_name=f'{name} total water')
        results[f"{name}_fwc"] = ds.fwc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean frozen water concentration')
        results[f"{name}_twc"] = ds.twc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean total water concentration')
        results[f"{name}_rho_mu"] = ds.rho.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean density')

        # rainy
        pr = ds.pr.max('level_full', skipna=True) if 'level_full' in ds.pr.dims else ds.pr
        results[f"{name}_pr"] = pr.sum(('lat','lon'), min_count=1).assign_attrs(units='kg m-2 s-1',
        long_name=f'{name} total precipitation flux')
        results[f"{name}_pr_max"] = pr.max(('lat','lon')).assign_attrs(units='kg m-2 s-1', long_name=f'{name} maximum precipitation flux')

        # dynamic
        results[f"{name}_max_w"] = ds.w.max('level_full').assign_attrs(units='m s-1', long_name=f'{name} column maximum vertical velocity')
        results[f"{name}_mean_w"] = ds.w.mean(('lat','lon')).assign_attrs(units='m s-1', long_name=f'{name} mean vertical velocity')
        results[f"{name}_amax_w"] = ds.w.max(('lat','lon')).assign_attrs(units='m s-1', long_name=f'{name} maximum vertical velocity')
        results[f"{name}_div_mu"] = ds.div.mean(('lat','lon')).assign_attrs(units='s-1', long_name=f'{name} mean horizontal wind divergence')

        # convective fluxes
        results[f"{name}_cmf_max"] = ds.cmf.max(('lat','lon')).assign_attrs(units='kg s-1 m-2', long_name=f'{name} maximum convective mass flux')
        results[f"{name}_cmf_mu"] = ds.cmf.mean(('lat','lon')).assign_attrs(units='kg s-1 m-2', long_name=f'{name} mean convective mass flux')
        results[f"{name}_cmt"] = (ds.cmf * ds.area).sum(('lat','lon'), min_count=1).assign_attrs(units='kg s-1', long_name=f'{name} convective mass transport')

        return results

    def get_everything(self, mask, data, cpoint, radius):
        # derive properties
        ds = self.derive_fields(data.sel(time=mask.time, level_full=mask.level_full, lat=mask.lat, lon=mask.lon))
        results = xr.Dataset({})

        # enironment calcs
        logging.info(f"{datetime.now()} Environment...")
        ds_env = self.derive_environmental_fields(data.sel(time=cpoint.time.isoformat())).where(environment.interacting(data, cpoint, radius))
        results = results.merge(self.environment_calcs(ds_env, name='env'))

        # cloud calcs
        logging.info(f"{datetime.now()} Cloud...")
        cloud_mask = mask.system > 0
        cloud_ds = ds.where(cloud_mask)
        results = results.merge(self.cloud_calcs(cloud_ds, name='cloud'))

        # anvil calcs
        logging.info(f"{datetime.now()} Anvil...")
        anvil_mask, ABH = self.define_anvil(cloud_mask, results, level=None, shift=None)
        results['ABH'] = ABH.assign_attrs(units='level_full', long_name='level used to define the anvil base height')
        results = results.merge(self.anvil_calcs(ds.where(anvil_mask), name='anvil'))

        # anvil base height sensitivity calcs
        l400 = np.abs(ds_env.p.mean(('lat','lon')) - 40000).idxmin('level_full') # 400 hPa level
        anvil_mask, ABH_x = self.define_anvil(cloud_mask, results, level=l400, shift=None)
        results = results.merge(self.anvil_sensitvity_calcs(ds.where(anvil_mask), level=ABH_x, name=f'anvil_l400'))

        # core calcs
        logging.info(f"{datetime.now()} Cores...")
        core_mask = mask.u_tracks
        core_id = core_mask.where(core_mask > 0).rename('core')
        core_results = ds.groupby(core_id).apply(self.core_calcs)
        
        return results.merge(core_results).fillna(self.NAN)

 