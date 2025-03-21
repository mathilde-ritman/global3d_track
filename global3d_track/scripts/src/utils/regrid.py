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
        
        if region == 'amazon':
            self.bbox = (-83,-43,-15,15) # Amazon
        elif region == 'congo':
            self.bbox = (1, 41,-10,10) # Congo basin
        elif region == 'tropics':
            self.bbox = (-180, 180, -30, 30) # Tropics
        elif region == 'global':
            self.bbox = (-180, 180, -90, 90) # Global
        elif region == 'anatomy_example':
            self.bbox = (-78,-70,9,15) # Example from hackathon March 2024
        elif region == 'anatomy_example_large':
            self.bbox = (-80,-68,7,17) # Example from hackathon March 2024
        else:
            self.bbox = region


    def perform(self, data, zoom, resolution=0.1):

        # define nearest neighbour pix
        self.pix = self._get_pix(2**zoom, *self._get_latlon(self.bbox, resolution=resolution))
        
        # select data
        if 'time' in data.dims:
            li = []
            for t in data.time: 
                data_i = data.sel(time=t)
                li.append(self._regrid(data_i))

            return li

        else:
            data = self._regrid(data)

            return data


    def _regrid(self, data):
        # select
        return data.isel(cell=self.pix)
    

    def _get_pix(self, nside, lon, lat):
        return xr.DataArray(
            healpy.ang2pix(nside, *np.meshgrid(lon, lat), nest=True, lonlat=True), coords=(lat, lon),
            )

    def _get_latlon(self, bbox, resolution):
        lon = xr.DataArray(np.arange(bbox[0], bbox[1], step=resolution), dims=("lon",), name="lon", attrs=dict(units="degrees", standard_name="longitude"))
        lat = xr.DataArray(np.arange(bbox[2], bbox[3], step=resolution), dims=("lat",), name="lat", attrs=dict(units="degrees", standard_name="latitude"))
        return lon, lat
    

