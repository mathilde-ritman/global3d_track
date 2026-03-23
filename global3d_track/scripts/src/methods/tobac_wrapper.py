'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import xarray as xr
import os
import pandas as pd
import numpy as np
import iris
import yaml
from pathlib import Path
from typing import Any, Union, Callable
import logging
import tobac
import datetime
from .connect_contiguous import Connect
from ..utils import tools

class Helpers:

    def __init__(self):
        pass

    def _amend_options(self, options: Union[str, dict], option_mods: dict={}):

        if isinstance(options, str):
            di = self._load_yaml(options)
        else:
            di = options
            
        # update dict with passed params
        if option_mods:
            di = {k: self._update_di(di[k], option_mods[k]) if k in option_mods.keys() else di[k] for k in di.keys()}
            for k in option_mods.keys():
                if k not in di.keys():
                    di[k] = option_mods[k]
        return di
    
    def _load_yaml(self, file):
        
        with open(file, 'r') as f:
            di = yaml.safe_load(f)

        return di
    
    def _update_di(self, old, new):
        if isinstance(old, dict):
            return {**old, **new}
        else:
            return new
        
    def _rename_zcoord(self, ds):
        ds = ds.rename({k:'altitude' for k in ['height','level_full','level_half'] if k in ds.dims})
        if 'altitude' in ds.dims:
            ds.altitude.attrs['standard_name'] = 'altitude'
        return ds
    
    def _table_to_dataset(self, table, col, dataset, base_col='feature'):
        di = table.set_index(base_col)[col].to_dict()
        feature_array = dataset[base_col].values
        dataarray = xr.DataArray(np.vectorize(lambda x: di.get(x, 0))(feature_array), dims=dataset[base_col].dims, coords=dataset[base_col].coords)
        return dataarray


class Track(Connect, Helpers):

    '''
    This class runs a tobac tracking for the provided xr.DataArrays, with additional (optional) postprocessing to correct current (July 2024) issues with feature linking merges.

    Example useage:
        from my_library.multivariate_tobac import Track
        module = Track(select_data, segment_data, options,)
        result = module.perform(save=True)

    '''

    def __init__(self, select_data, segment_data, options: Union[str, dict], overwrite: bool=False, overwrite_tracks: bool=False, track_params: dict={}, save: bool=True):

        Helpers()

        if select_data is not None and not isinstance(select_data, iris.cube.Cube):
            # self.select_data = self._rename_zcoord(select_data).to_iris()
            select_data = select_data.to_iris()
        if segment_data is not None and not isinstance(segment_data, iris.cube.Cube):
            # self.segment_data = self._rename_zcoord(segment_data).to_iris()
            segment_data = segment_data.to_iris()
        self.segment_data = segment_data
        self.select_data = select_data
        self.options = self._amend_options(options, track_params)
        self.overwrite = overwrite
        self.overwrite_tracks = overwrite_tracks
        self.save = save
        self.savedir_v = f"{self.options['savedir']}/{self.options['version_name']}"
        os.makedirs(self.savedir_v, exist_ok=True)


    def perform(self, detect=True, segment=True, track=False, merge=False, connect=False, save=None):

        if isinstance(save, bool):
            self.save = save

        di = self.options

        dxy, dt = di['grid_spacing'], di['time_spacing']

        logging.info(f"{datetime.datetime.now()} Output will save to: {self.savedir_v}")

        # -- load or compute features
        if Path(self.savedir_v, 'features.h5').is_file() and not self.overwrite:
            features = pd.read_hdf(Path(self.savedir_v, 'features.h5'), 'table')

        elif detect:
            features = tobac.feature_detection_multithreshold(self.select_data, dxy, **di['params_features'])

            if self.save:
                savepath = Path(self.savedir_v, 'features.h5')
                features.to_hdf(savepath, 'table')
                logging.info('feature selection results saved to ' + str(savepath))

        # -- load or compute segmentation
        if Path(self.savedir_v, 'segmented_mask.nc').is_file() and not self.overwrite:
            mask_dataset = xr.open_dataset(Path(self.savedir_v, 'segmented_mask.nc'))

        elif segment:
            segmented_mask, segmented_features = tobac.segmentation.segmentation(features, self.segment_data, dxy, **di['params_segmentation'])
            # -- transform mask dataset to xarray
            mask_dataset = xr.DataArray.from_iris(segmented_mask).to_dataset(name='feature')
            mask_dataset['feature'].attrs['description'] = 'tobac features after segmentation'
            
            if self.save:
                tools.save_xarray(mask_dataset, Path(self.savedir_v, 'segmented_mask.nc'))
                logging.info('feature segmentation results saved to ' + self.savedir_v)
            
        # -- load or compute tracking
        if Path(self.savedir_v, 'tracked_features.h5').is_file() and not self.overwrite_tracks:
            tracks = pd.read_hdf(Path(self.savedir_v, 'tracked_features.h5'), 'table')
            mask_dataset = xr.open_dataset(Path(self.savedir_v, 'tracked_mask.nc'))

        elif track:
            tracks = tobac.linking_trackpy(features, None, dt=dt, dxy=dxy, **di['params_linking'])
            tracks['cell'] = tracks.cell.where(tracks.cell!=-1, 0) # untracked features are reassigned from -1 to 0
            # -- add to mask dataset
            mask_dataset['cell'] = self._table_to_dataset(table=tracks, col='cell', dataset=mask_dataset)
            mask_dataset['cell'] = mask_dataset.cell
            mask_dataset['cell'].attrs['description'] = 'tobac features after tracking'

            if self.save:
                tools.save_xarray(mask_dataset, Path(self.savedir_v, 'tracked_mask.nc'))
                tracks.to_hdf(Path(self.savedir_v, 'tracked_features.h5'), 'table')
                logging.info('tracking results saved to ' + str(Path(self.savedir_v, 'tracks.h5')))
            
        # -- output
        if connect or merge or track:
            return mask_dataset, tracks
        elif segment:
            return mask_dataset, features
        elif detect:
            return features
        else:
            raise ValueError("You haven't selected any steps to perform")

