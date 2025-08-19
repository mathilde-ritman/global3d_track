'''
Mathilde Ritman 2025
'''

import xarray as xr
import numpy as np
import dask
from dask import delayed, compute
import logging
from datetime import datetime
from .cmf import CMF
from .water_path import calculate_xWP
from .density import density
from . import relative_humidty

import time
import tracemalloc

'''
more cloud object statistics: this time with a focus on key results wanter to assess the properties of the anvils, and the properties of the cores.

'''

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Decorator for timing
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"[{func.__name__}] Elapsed time: {elapsed:.2f} seconds")
        return result
    return wrapper

# Decorator for memory usage using tracemalloc
def memory_tracked(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        result = func(*args, **kwargs)
        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        logging.info(f"[{func.__name__}] Memory usage (top 10 lines):")
        for stat in stats[:10]:
            logging.info(stat)
        tracemalloc.stop()
        return result
    return wrapper


class EvaluationBucket(CMF):

    def __init__(self):

        super().__init__()

        self.grid_spacings = 11000 # m
        self.vert_spacings = 300 # m
        self.time_spacings = 900 # s
        self.NAN = -999.99

    # @timed
    # @memory_tracked
    def get_condensate_profiles(self, mask, data, name, ):
        '''  Ice water path calculation '''
    
        masked_data = data[['cli','clw']].sel(time=mask.time).where(mask>0)

        cli = masked_data.cli.sum(('lat','lon'))
        cli.attrs = dict(units='kg kg-1', long_name=f'{name} total cloud ice')
        clw = masked_data.clw.sum(('lat','lon'))
        clw.attrs = dict(units='kg kg-1', long_name=f'{name} total cloud water')

        ds = xr.Dataset({f'{name}_cli_prof': cli,
                         f'{name}_clw_prof': clw})

        dims = (x for x in ('time','level_full','lat','lon') if x in ds.dims)
        return ds.transpose(*dims)
    
    def cloudy_vs_dry_updraft(self, mask, data, name, ):
        '''  Ice water path calculation '''
    
        # masked_data = data[['cli','clw']].sel(time=mask.time).where(mask>0)
        mask
    
    
    
    def _process_single_core(self, core_mask, c, data, name):
        # c_mask = core_mask.where(core_mask == c) # mask current core
        # staitistics
        # collect
        pass

    def core(self, core_mask, data, name):

        if not (core_mask.max() > 0):
            # there are no cores in the mask provided
            return xr.Dataset(coords=dict(core=None, time=core_mask.time)).expand_dims('core').fillna(self.NAN)
       
        # # iterate cores
        # cores = dask.array.unique(core_mask.data).compute()
        # cores = cores[~np.isnan(cores)]

        # # process
        # tasks = []
        # for c in cores:
        #     task = delayed(self._process_single_core)(core_mask, c, data, name)
        #     tasks.append(task)

        # # collect
        # core_stats = xr.concat(compute(*tasks), dim='core')
        # core_stats = core_stats.assign_coords({'core':cores})
        # return core_stats.fillna(self.NAN)
    
    def anvil(self, anvil_mask, data, name='anvil'):

        if not (anvil_mask.max() > 0):
            # there are no results in the mask provided
            return xr.Dataset(coords=anvil_mask.coords).fillna(self.NAN)
        
        # stats

        # return anvil_stats.fillna(self.NAN)

    # @timed
    # @memory_tracked
    def get_everything(self, mask, data, ):
            
        # core_mask = mask.u_tracks
        # anvil_mask = mask.anvil

        # all results
        stats = self.get_condensate_profiles(mask.system, data, 'all')

        return stats.fillna(self.NAN)