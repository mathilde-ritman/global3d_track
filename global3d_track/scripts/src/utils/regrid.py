''' 
Mathilde Ritman 2023, mathilde.ritman@physics.ox.ac.uk

'''

import numpy as np
import xarray as xr
import healpy
import dask

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


class Regrid:

    ''' Regrid ICON model to the chosen region output using nearest neighbour interpolation '''

    def __init__(self, region):
        self.named_regions = {"amazon": (-83, -43, -15, 15),
                        "congo": (1, 41, -15, 15),
                        "tropics": (-180, 180, -15, 15),
                        "global": (-180, 180, -90, 90),
                        "anatomy_example": (-78, -70, 9, 15), # Example from hackathon March 2024
                        "anatomy_example_large": (-80, -68, 7, 17), # Example from hackathon March 2024
                        "tcr_amazon": (280, 325, -15, 10),
                        "tcr_congo": (0, 35, -12, 24),
                        "tcr_warm pool": (90, 160, -15, 15),
                        "atlantic": (-50, 0, -10, 15)}  
        if isinstance(region, str) and region in self.named_regions:
            self.bbox = self.named_regions[region]
        else:
            self.bbox = region

    def perform(self, data, zoom, resolution=0.1):
        self.pix = self._get_pix(
            2**zoom, *self._get_latlon(self.bbox, resolution=resolution)
        )
        # r = lambda x: ((x / resolution).round() * resolution).round(3)
        # data['lon'], data['lat'] = r(data.lon), r(data.lat)
        # return data.isel(cell=self.pix)
        return data.drop_vars(["lat", "lon"], errors="ignore").sel(cell=self.pix).drop_vars(["cell", "crs",], errors="ignore")
    
    def _get_pix(self, nside, lon, lat):
        return xr.DataArray(
            healpy.ang2pix(nside, *np.meshgrid(lon, lat), nest=True, lonlat=True), coords=(lat, lon),
            )

    # Fixed, bbox-independent anchor for the output grid (matches the "tropics"/"global"
    # named regions). Grid points are always `origin + k*resolution` for integer k, computed
    # the same way no matter which bbox asked for them - this is what makes a small, local
    # regrid (per DCC) bit-for-bit identical to slicing a regrid of the full (-180,180,-15,15)
    # (or (0,360,-15,15), equivalently) region.
    _ORIGIN_LON = -180.0
    _ORIGIN_LAT = -90.0

    def _get_latlon(self, bbox, resolution):
        # Building each call's grid from np.arange(bbox[0], bbox[1], resolution) makes the
        # sampled float64 coordinates depend on bbox[0] - two calls covering the "same"
        # nominal point can therefore differ by ~1e-13, which is enough to flip which native
        # HEALPix cell gets matched by the nearest-neighbour lookup below (the ~0.115 deg
        # native cell size at zoom 9 is close to this 0.1 deg output resolution, so lookups
        # are already right at the aliasing limit). Anchoring to a fixed origin and indexing
        # by integer offset removes that dependence entirely.
        lon = self._ORIGIN_LON + self._grid_indices(bbox[0], bbox[1], self._ORIGIN_LON, resolution) * resolution
        lat = self._ORIGIN_LAT + self._grid_indices(bbox[2], bbox[3], self._ORIGIN_LAT, resolution) * resolution
        lon = xr.DataArray(lon, dims=("lon",), name="lon", attrs=dict(units="degrees", standard_name="longitude"))
        lat = xr.DataArray(lat, dims=("lat",), name="lat", attrs=dict(units="degrees", standard_name="latitude"))
        return lon, lat

    @staticmethod
    def _grid_indices(lo, hi, origin, resolution):
        ''' integer offsets from `origin` covering [lo, hi) on a fixed resolution grid '''
        i0 = int(np.floor(round((lo - origin) / resolution, 6)))
        i1 = int(np.ceil(round((hi - origin) / resolution, 6)))
        return np.arange(i0, i1)
    

