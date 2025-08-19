'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
from datetime import datetime
import metpy

'''
calculate the cross section of a given variable over the cloud centre in the direction of the mean wind.

'''


class CrossSection:

    def __init__(self):

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99

    def find_max_coords_heavy(self, da):
        ''' Return the lat/lon coordinates of the maximum value in a multidimensional array. '''

        # stack to determine index of maxima
        stacked = da.stack(points=("lat", "lon"))
        flat_idx = stacked.argmax("points")

        # unravel_index via apply_ufunc (Dask-safe, returns a (2, ...) array)
        lat_len, lon_len = da.sizes["lat"], da.sizes["lon"]
        lat_idx, lon_idx = xr.apply_ufunc(
            lambda idx: np.unravel_index(idx, (lat_len, lon_len)),
            flat_idx,
            input_core_dims=[[]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64, np.int64],
        )

        # Convert indices into coordinate values
        lat_vals = xr.apply_ufunc(
            np.take,
            da.lat,
            lat_idx,
            input_core_dims=[["lat"], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[da.lat.dtype],
        )

        lon_vals = xr.apply_ufunc(
            np.take,
            da.lon,
            lon_idx,
            input_core_dims=[["lon"], []],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[da.lon.dtype],
        )

        return xr.Dataset({"lat": lat_vals, "lon": lon_vals})
    
    def find_max_coords(self, da):
        ''' As above, but for smaller arrays. '''
        stacked = da.stack(points=("lat", "lon"))
        flat_idx = stacked.argmax("points")
        lat_idx, lon_idx = np.unravel_index(flat_idx, (da.sizes["lat"], da.sizes["lon"]))
        result = xr.Dataset(coords=dict(time=da.time))
        result["lat"] = (('time',), da.lat[lat_idx].data)
        result["lon"] = (('time',), da.lon[lon_idx].data)
        return result
    
    def determine_cloud_centre(self, ds):

        # IWP maxima
        iwp_max = self.find_max_coords(ds.anvil_iwp.fillna(0))
        iwp_max = iwp_max.where(ds.anvil_volume>0)

        # CMF maxima
        c_data = ds.core_column_cmf_cl.max('core')
        cmf_max = self.find_max_coords(c_data.fillna(0))
        cmf_max = cmf_max.where(ds.core_volume.max('core')>0)

        # take CMF maxima when available, otherwise take IWP maxima
        return iwp_max.where(cmf_max.isnull(), cmf_max)
    
    def grid_walk_uv(self, mask_2d, u, v, lat, lon, center_lat, center_lon, max_steps=100, step_dir=1):
        ''' Walk from the centre coords to the edge of the mask in the direction of the mean wind. Returns intersect coordinate. '''

        
        if np.any(np.isnan(center_lat)) or np.any(np.isnan(center_lon)):
            # skip times with NaN values
            return np.nan, np.nan
        
        # Convert center point to nearest index
        iy0 = np.abs(lat - center_lat).argmin()
        ix0 = np.abs(lon - center_lon).argmin()

        # determine number of steps in each dir
        ratio = np.abs(u / v)
        dx = step_dir * ratio * (u / np.abs(u))
        dy = step_dir * (v / np.abs(v))

        # norm = np.hypot(u, v)
        # dx = step_dir *  u / norm
        # dy = step_dir * v / norm

        for step in np.arange(0, max_steps, 1):
            x = ix0 + step * dx
            y = iy0 + step * dy
            ix, iy = int(x), int(y)
            if 0 <= iy < lat.shape[0] and 0 <= ix < lon.shape[0]:
                if not mask_2d[iy, ix]:  # exited the mask
                    return lat[iy], lon[ix]
            else:
                # exited the grid, take preceding coordinate
                ix, iy = int(ix0 + (step - 1) * dx), int(iy0 + (step - 1) * dy)
                if 0 <= iy < mask_2d.shape[0] and 0 <= ix < mask_2d.shape[1]:
                    return lat[iy], lon[ix]

    def find_boundary_intersections(self, mask, u, v, lat, lon, center_lat, center_lon, step_dir=1):
        ''' Find the intersection of the mean wind direction with the cloud boundary. '''
        out_lat, out_lon = xr.apply_ufunc(
            self.grid_walk_uv,
            mask,
            u, v,
            lat,
            lon,
            center_lat,
            center_lon,
            input_core_dims=[["lat", "lon"], [], [], ["lat"], ["lon"], [], []],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
            kwargs={"max_steps": 100, "step_dir": step_dir},
        )
        return xr.Dataset(dict(lat=out_lat, lon=out_lon))

    def metpy_cross_section(self, data, start, end):
        start = (start.lat, start.lon)
        end = (end.lat, end.lon)
        return metpy.interpolate.cross_section(data, start, end)
    
    def compute_cross_sections(self, ds, centre, windward, leeward):

        # provide CRS information
        projection_crs = dict(grid_mapping_name = 'latitude_longitude')
        metpy_ds = ds.metpy.assign_crs(projection_crs)
        metpy_ds = metpy_ds.metpy.parse_cf().squeeze()

        if np.any(np.isnan(metpy_ds)):
            logging.warning("NaN values present in data fields, using default fill value = 0.")
            metpy_ds = metpy_ds.fillna(0)

        all_results = []
        for t in metpy_ds.time:
            if np.any(np.isnan(centre.sel(time=t).lat)) or np.any(np.isnan(centre.sel(time=t).lon)):
                # skip times with NaN values
                continue
            # compute
            in_fwrd = [metpy_ds, centre, windward]
            in_bkwd = [metpy_ds, centre, leeward]
            wward_cross = self.metpy_cross_section(*tuple([x.sel(time=t) for x in in_fwrd]))
            lee_cross = self.metpy_cross_section(*tuple([x.sel(time=t) for x in in_bkwd]))
            # shape
            lee_cross = lee_cross.sel(index=slice(1,None)) # drop duplicate 0 point
            lee_cross['index'] = -1*lee_cross.index # reverse index
            cross_sec = xr.concat((wward_cross, lee_cross), dim='index').sortby('index')
            cross_sec = cross_sec.rename({'index': 'normalised_distance'}).expand_dims(time=[t.values])
            # make lat/lon coords of transect defined along time
            cross_sec['lat'] = cross_sec.lat.expand_dims(time=cross_sec.time)
            cross_sec['lon'] = cross_sec.lon.expand_dims(time=cross_sec.time)
            all_results.append(cross_sec)
        
        output = xr.concat(all_results, dim='time').reset_coords(["lat", "lon"], drop=False)
        output = output.reindex(time=ds.time, ) # fill missing timesteps
        return output
    
    def get_everything(self, ds, variables=['anvil_iwp','anvil_depth']):

        # input
        ds = ds.where(ds != self.NAN)

        # check NaNs
        if ds.anvil_ua.isnull().all() or ds.anvil_va.isnull().all():
            logging.warning("No wind data available, skipping cross section calculation.")
            return None

        # 1. centre points (lagrangian)
        cloud_centre = self.determine_cloud_centre(ds)

        # 2. coordinates of intersection with cloud boundary in mean wind direction
        ds = ds.chunk(dict(lat=-1, lon=-1, time=1, ))
        mask = (ds.anvil_depth>0) | (ds.core_depth.max('core')>0)
        inputs = (mask, ds.anvil_ua.mean('time'), ds.anvil_va.mean('time'), ds.lat, ds.lon, cloud_centre.lat, cloud_centre.lon)
        windward = self.find_boundary_intersections(*inputs) # from centre to edge
        leeward = self.find_boundary_intersections(*inputs, step_dir=-1) # from edge to centre

        # 3. calculate cross section
        cross_section = self.compute_cross_sections(ds[variables], cloud_centre, windward, leeward)
        cross_section.normalised_distance.attrs = dict(long_name='normalised distrance', description='normalised distance from cloud centre in the direction of mean wind (positive)')

        # 4. collect output
        coords = xr.merge((cloud_centre.rename({'lat':'centre_lat', 'lon':'centre_lon'}), 
                           windward.rename({'lat':'wward_lat', 'lon':'wward_lon'}), 
                           leeward.rename({'lat':'lee_lat', 'lon':'lee_lon'})))
        result = xr.merge((cross_section.rename({'lat':'cross_section_lat', 'lon':'cross_section_lon'}), coords))

        return result.drop_vars('metpy_crs').fillna(self.NAN)
        
