'''
Mathilde Ritman 2026
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
from datetime import datetime
from . import calculations as calcs
from . import environment
from ..utils import definitions


'''
Calculate the statistics used in the analyses

'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSUStats:

    def __init__(self, dxy, dz, zdim, pname):

        self.dxy = dxy # m
        self.dz = dz # m
        self.dt = 900 # s
        self.zdim = zdim # eg. "level_full"
        self.pname = pname # eg. "pressure"
        self.NAN = -999.99
        self.CMF = calcs.CMF(grid_spacing=self.dxy)

    def get_cloudy_fields(self, data):
        data.attrs = {}
        for var in data.variables:
            data[var].attrs = {}

        # calculate all cloudy data fields
        ds = xr.Dataset({})

        # geometric
        da = data.ta
        ds['area'] = (self.dxy**2 * xr.DataArray(np.ones_like(da), dims=da.dims, coords=da.coords)).assign_attrs(units='m2', long_name='area')
        ds['depth'] = data.dzghalf.assign_attrs(units='m', long_name='depth')
        ds['volume'] = (ds.area * ds.depth).assign_attrs(units='m3', long_name='volume')
        ds['zg'] = data.zg.assign_attrs(units='m', long_name='height')

        # condensate
        ds['p'] = data.pfull.assign_attrs(units='Pa', long_name='pressure')
        ds['pr'] = data.pr.assign_attrs(units='kg m-2 s-1', long_name='precipitation flux')
        if 'cli' in data.data_vars:
            ds['fwc'] = calcs.IWC(data).assign_attrs(units='kg m-3', long_name='frozen water concentration')
            ds['twc'] = calcs.TWC(data).assign_attrs(units='kg m-3', long_name='total water concentration')
        elif 'qfrozen' in data.data_vars:
            ds['fwc'] = calcs.xWC(data, 'qfrozen', '5km').assign_attrs(units='kg m-3', long_name='frozen water concentration')
            ds['twc'] = calcs.xWC(data, 'qall', '5km').assign_attrs(units='kg m-3', long_name='total water concentration')
        
        # dynamic
        ds['rho'] = calcs.moist_air_density(data).assign_attrs(units='kg m-3', long_name='moist air density')
        ds['w'] = data.wa_phy.assign_attrs(units='m s-1', long_name='vertical velocity')
        ds['cmf'] = self.CMF.mass_flux(ds.w, ds.rho).assign_attrs(units='kg s-1 m-2', long_name='convective mass flux')

        return ds

    def get_environmnetal_fields(self, data):
        version = '10km' if 'cli' in data.data_vars else '5km'

        data.attrs = {}
        for var in data.variables:
            data[var].attrs = {}

        ds = xr.Dataset({})

        ds['p'] = data.pfull.assign_attrs(units='Pa', long_name='pressure')
        ds['T'] = data.ta.assign_attrs(units='K', long_name='temperature')
        ds['rh'] = (data.hur/100 if 'hur' in data.data_vars else calcs.relative_humidity(data)).assign_attrs(units='', long_name='relative humidity')
        ds['tpwv'] = calcs.xWC(data, 'hus', version) * self.dxy**2 * data.dzghalf
        ds['ua'] = data.ua.assign_attrs(units='m s-1', long_name='zonal wind')
        ds['va'] = data.va.assign_attrs(units='m s-1', long_name='meridional wind')
        ds['low_shear'] = calcs.wind_shear(data, 0, 2, zdim=self.zdim).assign_attrs(units='m s-1', long_name='low-level wind shear (0-2 km)')
        ds['mid_shear'] = calcs.wind_shear(data, 2, 6, zdim=self.zdim).assign_attrs(units='m s-1', long_name='mid-level wind shear (2-6 km)')
        ds['high_shear'] = calcs.wind_shear(data, 6, 10, zdim=self.zdim).assign_attrs(units='m s-1', long_name='upper-level wind shear (6-10 km)')
        ds['S'] = calcs.static_stability(ds.T, ds.p, zdim=self.zdim).assign_attrs(units='K Pa-1', long_name='static stability')
        ds['tropo'] = calcs.wmo_tropopause_height(ds.T, data.zg, zdim=self.zdim).assign_attrs(units='km', long_name='tropopause height')
        cape, cin, lfc, lcl = calcs.profile_quantities(ds.p, ds.T, ds.rh, zdim=self.zdim)
        ds['cape'] = cape.assign_attrs(units='J kg-1', long_name='convective available potential energy')
        ds['cin'] = cin.assign_attrs(units='J kg-1', long_name='convective inhibition')
        ds['lfc'] = lfc.assign_attrs(units='Pa', long_name='level of free convection')
        ds['lcl'] = lcl.assign_attrs(units='Pa', long_name='lifting condensation level')

        return ds

    def cloud_calcs(self, ds, name='cloud'):

        ds[f"{name}_area"] = ds.area.sum(('lat','lon'), keep_attrs=True)
        ds[f"{name}_depth"] = ds.depth.sum(self.zdim, keep_attrs=True)
        ds[f"{name}_volume"] = ds.volume.sum(keep_attrs=True)
        ds[f"{name}_cth"] = ds.zg.max((self.zdim,'lat','lon')).assign_attrs(units='m', long_name=f'{name} top height')
        ds[f"{name}_cbh"] = ds.zg.min((self.zdim,'lat','lon')).assign_attrs(units='m', long_name=f'{name} base height')
        ds[f"{name}_pr"] = ds.pr.sum(('lat','lon'), keep_attrs=True)
        ds[f"{name}_fw"] = (ds.fwc * ds.area * ds.depth).sum(('lat','lon')).assign_attrs(units='kg', long_name=f'{name} total frozen water')
        ds[f"{name}_tw"] = (ds.twc * ds.area * ds.depth).sum(('lat','lon')).assign_attrs(units='kg', long_name=f'{name} total water')
        ds[f"{name}_fwp"] = (ds.fwc * ds.depth).sum(self.zdim).assign_attrs(units='kg m-2', long_name=f'{name} frozen water path')
        ds[f"{name}_twp"] = (ds.twc * ds.depth).sum(self.zdim).assign_attrs(units='kg m-2', long_name=f'{name} total water path')
        ds[f"{name}_fwc"] = ds.fwc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean frozen water concentration')
        ds[f"{name}_twc"] = ds.twc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean total water concentration')

        return ds
    
    def core_calcs(self, ds, abh, name='core'):

        ds = ds.unstack(ds.dims)

        ds[f"{name}_area"] = ds.area.sum(('lat','lon'), keep_attrs=True)
        ds[f"{name}_depth"] = ds.depth.sum(self.zdim, keep_attrs=True)
        ds[f"{name}_volume"] = ds.volume.sum(keep_attrs=True)
        ds[f"{name}_cth"] = ds.zg.max((self.zdim,'lat','lon')).assign_attrs(units='m', long_name=f'{name} top height')
        ds[f"{name}_cbh"] = ds.zg.min((self.zdim,'lat','lon')).assign_attrs(units='m', long_name=f'{name} base height')
        ds[f"{name}_pr"] = ds.pr.sum(('lat','lon'), keep_attrs=True)
        ds[f"{name}_fwc"] = ds.fwc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean frozen water concentration')
        ds[f"{name}_twc"] = ds.twc.mean(('lat','lon')).assign_attrs(units='kg m-3', long_name=f'{name} mean total water concentration')
        ds[f"{name}_max_w"] = ds.w.max(self.zdim).assign_attrs(units='m s-1', long_name=f'{name} maximum vertical velocity')
        ds[f"{name}_w_max"] = ds.w.max(('lat','lon')).assign_attrs(units='m s-1', long_name=f'{name} maximum vertical velocity')
        ds[f"{name}_cmf_max"] = ds.cmf.max(('lat','lon')).assign_attrs(units='kg s-1 m-2', long_name=f'{name} maximum convective mass flux')
        ds[f"{name}_cmf_mu"] = ds.cmf.mean(('lat','lon')).assign_attrs(units='kg s-1 m-2', long_name=f'{name} mean convective mass flux')
        ds[f"{name}_cmt"] = (ds.cmf * ds.area).sum(('lat','lon')).assign_attrs(units='kg s-1', long_name=f'{name} convective mass transport')
        ds[f"{name}_top_w"] = ds.w.sel({self.zdim: ds[self.zdim].max()}).assign_attrs(units='m s-1', long_name=f'{name} vertical velocity at core top height')
        ds[f"{name}_top_cmf"] = ds.cmf.sel({self.zdim: ds[self.zdim].max()}).assign_attrs(units='kg s-1 m-2', long_name=f'{name} convective mass flux at core top height')
        if abh.values in ds[self.zdim].values:
            abh_w = ds.w.sel({self.zdim: abh})
            abh_cmf = ds.cmf.sel({self.zdim: abh})
        else:
            da = ds[f"{name}_top_w"]
            abh_w = xr.DataArray(np.nan, dims=da.dims, coords=da.coords)
            abh_cmf = xr.DataArray(np.nan, dims=da.dims, coords=da.coords)
        ds[f"{name}_abh_w"] = abh_w.assign_attrs(units='m s-1', long_name=f'{name} vertical velocity at anvil base height')
        ds[f"{name}_abh_cmf"] = abh_cmf.assign_attrs(units='kg s-1 m-2', long_name=f'{name} convective mass flux at anvil base height')

        return ds

    def aggregate_cloudy(self, mask, data):
        # derive properties
        ds = self.get_cloudy_fields(data)

        # cloud calcs
        logging.info(f"{datetime.now()} Cloud...")
        cloud_mask = mask.system > 0
        results = self.cloud_calcs(ds.where(cloud_mask), name='cloud')
        results['cloud_pressure'] = ds.p.where(cloud_mask).mean(('lat','lon')).assign_attrs(units='Pa', long_name='mean pressure')

        # anvil mask
        logging.info(f"{datetime.now()} Anvil...")
        anvil_mask, ABH = definitions.define_anvil(cloud_mask, data.zg, results.cloud_tw.reindex({self.zdim: mask[self.zdim]}), zdim=self.zdim)
        results['abh'] = ABH.assign_attrs(units=self.zdim, long_name='level used to define the anvil base height')
        results = results.merge(self.cloud_calcs(ds.where(anvil_mask), name='anvil'))

        # core calcs
        logging.info(f"{datetime.now()} Core...")
        core_mask = mask.u_tracks
        core_id = core_mask.where(core_mask > 0).rename('core')
        core_results = ds.groupby(core_id).apply(self.core_calcs, args=(results.abh, 'core'))
        
        return results.merge(core_results)
    
    def ingested_calcs(self, ds, mask, name='in'):
        ds = ds[['T', 'rh', 'tpwv', 'low_shear', 'mid_shear', 'cape', 'cin', 'lfc', 'lcl']]

        for v in ds.data_vars:
            if self.zdim in ds[v].dims:
                for lvl in (75000, 90000):
                    v_new = f"{v}_{lvl/100}"
                    ds[v_new] = ds[v].sel({self.zdim: np.abs(ds.pfull - lvl).idxmin(self.zdim)})
        ds = ds.where(mask)

        for v in ds.data_vars:
            func = 'sum' if v in ('tpwv',) else 'mean'
            ds[f"{v}_{name}"] = getattr(ds[v], func)(keep_attrs=1)
        
        return ds
    
    def interacting_calcs(self, ds, mask, name='around'):
        ds = ds[['T', 'rh', 'ua', 'va', 'high_shear', 'S', 'tropo']]

        for v in ds.data_vars:
            if self.zdim in ds[v].dims:
                lvl0, lvl1 = 60000, 10000
                mask_lvl = np.logical_and(ds.pfull<=lvl0, ds.pfull>=lvl1)
                ds[v] = ds[v].where(mask_lvl)
        ds = ds.where(mask)

        for v in ds.data_vars:
            func = 'mean'
            ds[f"{v}_{name}"] = getattr(ds[v], func)(keep_attrs=1)

        return ds

    def aggregate_environment(self, data, cpoint, radii):
        # derive properties
        ds = self.get_environmnetal_fields(data)

        results = xr.Dataset({})
        for radius in radii:
            logging.info(f"{datetime.now()} Define environments...")
            ingested = environment.ingested(data, cpoint, 75000, radius, zdim=self.zdim)
            interacting = environment.interacting(data, cpoint, radius*2)
            results.merge(self.ingested_calcs(ds, ingested, name=f"in{radius}"))
            results.merge(self.interacting_calcs(ds, interacting, radius, name=f"around{radius}"))

        return results

    def get_everything(self, mask, data, cpoint, radii, derive_properties=['cloud', 'environmental']):

        # confirm variable names
        rename = {self.pname:'pfull', 'wa':'wa_phy',}
        existing = set(data.variables) | set(data.dims)
        data = data.rename({k: v for k, v in rename.items() if k in existing})

        if 'pfull' in data.dims:
            mask = mask.rename({self.zdim:'pfull'})
            self.zdim = 'pfull'
            mask = mask.sel(pfull=slice(1000,100000))
            data = data.sel(pfull=slice(1000,100000))

        if 'dzghalf' not in data.data_vars:
            data['dzghalf'] = np.abs(data.zg.diff(self.zdim, label='lower')) # calculate dzghalf

        # get cloud and core properties
        if 'cloud' in derive_properties:
            logging.info(f"{datetime.now()} Cloudy properties...")
            mdata = data.sel({'time': mask.time, self.zdim: mask[self.zdim], 'lat': mask.lat, 'lon': mask.lon})
            cloudy_results = self.aggregate_cloudy(mask, mdata)

            if not 'environmental' in derive_properties:
                return cloudy_results.fillna(self.NAN)

        # get environment properties
        if 'environmental' in derive_properties:
            logging.info(f"{datetime.now()} Environmental properties...")
            around_results = self.aggregate_environment(data, cpoint, radii)

            if not 'cloud' in derive_properties:
                return around_results.fillna(self.NAN)

        return cloudy_results.merge(around_results).fillna(self.NAN)
