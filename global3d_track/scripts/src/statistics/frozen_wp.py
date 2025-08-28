'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
import logging
from datetime import datetime
from .calculations import density

'''
Frozen water path.
'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FrozenWP:

    def __init__(self):

        super().__init__()

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99

    def get_iwp(self, mask, data, name, ):
        '''  Frozen/Ice water path calculation '''
    
        masked_data = data[['pfull','ta','hus','cli','clw','qg','qr','qs','dzghalf']].sel(time=mask.time).where(mask>0)

        IWP = density.calculate_IWP(masked_data)
        IWP.attrs = dict(units='kg m-2', long_name=f'{name} frozen water path')

        ds = xr.Dataset({f'{name}_frozenwp': IWP})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def anvil(self, anvil_mask, data, name='anvil'):
        if not (anvil_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=anvil_mask.coords).fillna(self.NAN)
        anvil_stats = self.get_iwp(anvil_mask, data, name)
        return anvil_stats.fillna(self.NAN)

    def get_everything(self, mask, data, ):
        anvil_mask = mask.anvil
        stats = self.anvil(anvil_mask, data, 'anvil')
        return stats.fillna(self.NAN)