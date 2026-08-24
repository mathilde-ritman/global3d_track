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

    def get_geometric(self, mask, data, name, shortname=None):

        masked_data = data[['dzghalf','zg']].where(mask>0).sel(time=mask.time)

        if not shortname:
            shortname = name + '_'

        cell_area = (mask > 0) * (self.dxy**2) # m2
        cell_depth = masked_data.dzghalf # m
        area = cell_area.sum(('lat','lon')) # m2
        depth = masked_data.dzghalf.sum(self.zdim) # m
        volume = (cell_area * cell_depth).sum(('lat','lon',self.zdim)) # m3
        cth = masked_data.zg.max((self.zdim,'lat','lon')) # m
        cbh = masked_data.zg.min((self.zdim,'lat','lon')) # m
        
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

        dims = (x for x in ('time',self.zdim,'lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_condensates(self, mask, data, name):

        masked_data = data.sel(time=mask.time).where(mask>0)
        footprint = mask.max(self.zdim) > 0 # get footprint for precip calculations
        footprint_data = data[['pr']].sel(time=mask.time).where(footprint)

        # hard code versions
        if 'cli' in masked_data.data_vars:
            fwp, fwc = calcs.IWP(masked_data, return_iwc=True)
            twp, twc = calcs.TWP(masked_data, return_twc=True)
        else:
            fwc = calcs.xWC(masked_data, 'qfrozen', '5km')
            fwp = calcs.xWP(masked_data, 'qfrozen', '5km', zdim=self.zdim)
            twc = calcs.xWC(masked_data, 'qall', '5km')
            twp = calcs.xWP(masked_data, 'qall', '5km', zdim=self.zdim)

        # aggreagate
        cell_depth = masked_data.dzghalf.mean(('lat','lon')) # m
        fw = fwc.sum(('lat','lon')) * self.dxy**2 * cell_depth # kg
        tw = twc.sum(('lat','lon')) * self.dxy**2 * cell_depth # kg
        fwc = fwc.mean(('lat','lon'))
        twc = twc.mean(('lat','lon'))

        # attrs
        fwp.attrs = dict(units='kg m-2', long_name=f'{name} frozen water path')
        fwc.attrs = dict(units='kg m-3', long_name=f'{name} mean frozen water concentration')
        fw.attrs = dict(units='kg', long_name=f'{name} total frozen water')

        twp.attrs = dict(units='kg m-1', long_name=f'{name} total water path')
        twc.attrs = dict(units='kg m-3', long_name=f'{name} mean total water concentration')
        tw.attrs = dict(units='kg', long_name=f'{name} total water')

        # precipitation
        pr = footprint_data.pr # kg m-2 s-1
        pr.attrs = dict(units='kg m-2 s-1', long_name=f'{name} precipitation flux')

        ds = xr.Dataset({f'{name}_fwp': fwp,
                        f'{name}_fwc': fwc,
                        f'{name}_fw': fw,
                        f'{name}_twp': twp,
                        f'{name}_twc': twc,
                        f'{name}_tw': tw,
                        f'{name}_pr': pr,
            })
        
        dims = (x for x in ('time',self.zdim,'lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    def get_dynamic(self, mask, data, name, ABH):

        masked_data = data.sel(time=mask.time).where(np.logical_and(mask>0, data.wa_phy>1))

        # density
        rho = calcs.moist_air_density(masked_data)
        rho_mu = rho.mean(('lat','lon')) # kg m-3
        rho_mu.attrs = dict(units='kg m-3', long_name=f'{name} mean dry air density')

        # velocity
        w = masked_data.wa_phy # m s-1
        # - column-max
        max_w = w.max(self.zdim)
        max_w.attrs = dict(units='m s-1', long_name=f'{name} maximum vertical velocity')

        # - anvil base height
        abh_w = w.sel({self.zdim: ABH}) # m s-1
        abh_w.attrs = dict(units='m s-1', long_name=f'{name} vertical velocity at anvil base height')

        # mass flux
        cmf = self.CMF.mass_flux(w, rho)
        cmt = self.CMF.mass_transport(w, rho)
        abh_cmf = cmf.sel({self.zdim: ABH})
        abh_cmf.attrs = dict(units='kg s-1 m-2', long_name=f'{name} convective mass flux at anvil base height')
        cmt.attrs = dict(units='kg s-1', long_name=f'{name} convective mass transport')

        # aggregated
        cmf_mu = cmf.mean(('lat','lon'))
        cmf_max = cmf.max(('lat','lon'))
        cmf_mu.attrs = dict(units='kg s-1 m-2', long_name=f'{name} mean convective mass flux')
        cmf_max.attrs = dict(units='kg s-1 m-2', long_name=f'{name} maximum convective mass flux')

        ds = xr.Dataset({f'{name}_rho': rho_mu,
                        f'{name}_max_w': max_w,
                        f'{name}_abh_w': abh_w,
                        f'{name}_abh_cmf': abh_cmf,
                        f'{name}_cmf_mu': cmf_mu,
                        f'{name}_cmf_max': cmf_max,
                        f'{name}_cmt': cmt,
                })

        dims = (x for x in ('time',self.zdim,'lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def get_ingested_environment(self, mask, data, name='ingested', radius=''):

        # full results
        logging.info(f"{datetime.now()} pressure, temperature, humidity, tpwv...")
        p, T = data.pfull, data.ta
        rh = data.hur/100 if 'hur' in data.data_vars else calcs.relative_humidity(data)
        version = '5km' if 'hur' in data.data_vars else '10km'
        tpwv = calcs.xWC(data, 'hus', version) * self.dz * self.dxy**2
        logging.info(f"{datetime.now()} CAPE...")
        cape, cin, lfc, lcl = calcs.profile_quantities(p, T, rh, zdim=self.zdim)
        logging.info(f"{datetime.now()} wind shear...")
        low_shear = calcs.wind_shear(data, 0, 2, zdim=self.zdim)
        mid_shear = calcs.wind_shear(data, 2, 6, zdim=self.zdim)

        ds = xr.Dataset({
            f'{name}_rh': rh.assign_attrs(units='', long_name=f'{name} relative humidity'),
            f'{name}_T': T.assign_attrs(units='K', long_name=f'{name} temperature'),
            f'{name}_tpwv': tpwv.assign_attrs(units='kg', long_name=f'{name} total precipitable water vapor'),
            f'{name}_cape': cape.assign_attrs(units='J kg-1', long_name=f'{name} convective available potential energy'),
            f'{name}_cin': cin.assign_attrs(units='J kg-1', long_name=f'{name} convective inhibition'),
            f'{name}_lfc': lfc.assign_attrs(units='Pa', long_name=f'{name} level of free convection'),
            f'{name}_lcl': lcl.assign_attrs(units='Pa', long_name=f'{name} lifting condensation level'),
            f'{name}_low_shear': low_shear.assign_attrs(units='m s-1', long_name=f'{name} low-level wind shear (0-2 km)'),
            f'{name}_mid_shear': mid_shear.assign_attrs(units='m s-1', long_name=f'{name} mid-level wind shear (2-6 km)'),
            })

        # levels
        for v in ds.data_vars:
            if self.zdim in ds[v].dims:
                for lvl in (75000, 90000):
                    v_new = f"{v}_{lvl/100}"
                    ds[v_new] = ds[v].sel({self.zdim: np.abs(data.pfull - lvl).idxmin(self.zdim)})

        # mask
        ds = ds.where(mask)

        # aggregate
        for v in ds.data_vars:
            func = 'sum' if v in ('tpwv',) else 'mean'
            ds[v] = getattr(ds[v], func)(keep_attrs=1)
        
        for v in ds.data_vars:
            ds[v].attrs['description'] = f'upwind environment defined using a radius of {radius} deg'
        
        dims = (x for x in ('time',self.zdim,'lat','lon') if x in ds.dims)
        return ds.transpose(*dims)

    def get_interacting_environment(self, mask, data, name='interacting', radius=''):

        # full results
        logging.info(f"{datetime.now()} pressure, temperature, humidity...")
        p, T = data.pfull, data.ta
        rh = data.hur/100 if 'hur' in data.data_vars else calcs.relative_humidity(data)
        logging.info(f"{datetime.now()} horizontal winds...")
        ua, va = data.ua, data.va
        logging.info(f"{datetime.now()} static stability...")
        S = calcs.static_stability(T, p, zdim=self.zdim)
        logging.info(f"{datetime.now()} tropopause height...")
        tropo = calcs.wmo_tropopause_height(T, data.zg, zdim=self.zdim)
        logging.info(f"{datetime.now()} wind shear...")
        upper_shear = calcs.wind_shear(data, 6, 10, zdim=self.zdim)

        logging.info(f"{datetime.now()} masking...")
        ds = xr.Dataset({
            f'{name}_rh': rh.assign_attrs(units='', long_name=f'{name} relative humidity'),
            f'{name}_ua': ua.assign_attrs(units='m s-1', long_name=f'{name} zonal wind'),
            f'{name}_va': va.assign_attrs(units='m s-1', long_name=f'{name} meridional wind'),
            f'{name}_S': S.assign_attrs(units='K Pa-1', long_name=f'{name} static stability'),
            f'{name}_th': tropo.assign_attrs(units='km', long_name=f'{name} tropopause height'),
            f'{name}_upper_shear': upper_shear.assign_attrs(units='m s-1', long_name=f'{name} upper-level wind shear (6-10 km)'),
            })

        # levels
        for v in ds.data_vars:
            if self.zdim in ds[v].dims:
                lvl0, lvl1 = 60000, 10000
                mask = np.logical_and(data.pfull<=lvl0, data.pfull>=lvl1)
                ds[v] = ds[v].where(mask)

        ds = ds.where(mask)

        # aggregate
        logging.info(f"{datetime.now()} aggregating...")
        for v in ds.data_vars:
            func = 'mean'
            ds[v] = getattr(ds[v], func)(keep_attrs=1)
        
        for v in ds.data_vars:
            ds[v].attrs['description'] = f'environment defined using a radius of {radius} deg'

        logging.info(f"{datetime.now()} returning...")
        
        dims = (x for x in ('time',self.zdim,'lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    
    def individual_core(self, core_mask, c, data, name, ABH):
        c_mask = core_mask.where(core_mask == c) # mask current core
        core_stats = self.get_geometric(c_mask, data, name)
        core_stats.update(self.get_condensates(c_mask, data, name))
        core_stats.update(self.get_dynamic(c_mask, data, name, ABH))
        return core_stats

    def cores(self, core_mask, data, name='core', ABH=None):

        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset(coords=dict(core=None, time=core_mask.time)).expand_dims('core')
       
        # iterate cores
        cores = dask.array.unique(core_mask.data).compute()
        cores = cores[~np.isnan(cores)]

        # process
        results = []
        for c in cores:
            results.append(self.individual_core(core_mask, c, data, name, ABH))

        # collect
        core_stats = xr.concat(results, dim='core')
        core_stats = core_stats.assign_coords({'core':cores})

        return core_stats
    
    def anvil(self, anvil_mask, data, name='anvil'):

        if not (anvil_mask.max() > 0):
            return xr.Dataset(coords=anvil_mask.coords) # there are no results in the mask provided
        
        anvil_stats = self.get_geometric(anvil_mask, data, name)
        anvil_stats.update(self.get_condensates(anvil_mask, data, name))

        return anvil_stats

    def cloud(self, system_mask, data, name='cloud', shortname='c'):

        if not (system_mask.max() > 0):
            return xr.Dataset(coords=system_mask.coords) # there are no results in the mask provided
        
        cloud_stats = self.get_geometric(system_mask, data, name, shortname=shortname)
        cloud_stats.update(self.get_condensates(system_mask, data, name))

        return cloud_stats

    def get_everything(self, mask, data, cpoint, radii, derive_properties=['cloud', 'environmental']):

        # confirm variable names
        rename = {self.pname:'pfull', 'wa':'wa_phy',}
        existing = set(data.variables) | set(data.dims)
        data = data.rename({k: v for k, v in rename.items() if k in existing})

        if 'pfull' in data.dims:
            mask = mask.rename({self.zdim:'pfull'})
            self.zdim = 'pfull'
            # mask = mask.sel(pfull=slice(1000,100000))
            # data = data.sel(pfull=slice(1000,100000))

        if 'dzghalf' not in data.data_vars:
            data['dzghalf'] = np.abs(data.zg.diff(self.zdim, label='lower')) # calculate dzghalf

        # get cloud and core properties
        if 'cloud' in derive_properties:

            logging.info(f"{datetime.now()} Cloud properties...")
            mdata = data.sel(time=mask.time, lat=mask.lat, lon=mask.lon).sel({self.zdim: mask[self.zdim]})
            stats = self.cloud(mask.system, mdata) # system

            # define anvil & get anvil properties
            logging.info(f"{datetime.now()} Anvil properties...")
            anvil_mask, ABH = definitions.define_anvil(mask.system, mdata.zg, stats.cloud_tw.reindex({self.zdim: mask[self.zdim]}), zdim=self.zdim)
            stats['ABH'] = ABH # add anvil base height to stats
            stats['ABH'].attrs = dict(units=self.zdim, long_name='level used to define the anvil base height')
            stats = stats.merge(self.anvil(anvil_mask, mdata, 'anvil')) # anvil results

            # # core properties
            logging.info(f"{datetime.now()} Core properties...")
            stats = stats.merge(self.cores(mask.u_tracks, mdata, 'core', stats.ABH)) # core
            results = stats

        # get environment properties
        if 'environmental' in derive_properties:
            for radius in radii:
                logging.info(f"{datetime.now()} Ingested definition...")
                ingested = environment.ingested(data, cpoint, 75000, radius, zdim=self.zdim)
                logging.info(f"{datetime.now()} Interacting definition...")
                interacting = environment.interacting(data, cpoint, radius)
                logging.info(f"{datetime.now()} Ingested properties...")
                estats = self.get_ingested_environment(ingested, data, name='ingested', radius=radius)
                logging.info(f"{datetime.now()} Interacting properties...")
                estats = estats.merge(self.get_interacting_environment(interacting, data, name='interacting', radius=radius*2))
                results = estats

        logging.info(f"{datetime.now()} leaving calcs...")

        if 'cloud' in derive_properties and 'environmental' in derive_properties:
            results = stats.merge(estats)

        return results.fillna(self.NAN)