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
from .connect_contiguous import Connect


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
        dataarray = xr.DataArray(np.vectorize(lambda x: di.get(x, -9))(feature_array), dims=dataset[base_col].dims, coords=dataset[base_col].coords)
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
        self.features = None
        self.mask_dataset = None


    def perform(self, detect=True, segment=True, track=False, merge=False, connect=False, save=None, merge_method='ndimage'):

        NAN = -9

        if isinstance(save, bool):
            self.save = save

        di = self.options

        dxy, dt = di['grid_spacing'], di['time_spacing']

        # -- load or compute features
        if Path(self.savedir_v, 'features.h5').is_file() and not self.overwrite:
            features = pd.read_hdf(Path(self.savedir_v, 'features.h5'), 'table')
            self.features = features

        elif detect:
            features = tobac.feature_detection_multithreshold(self.select_data, dxy, **di['params_features'])
            self.features = features

            if self.save:
                savepath = Path(self.savedir_v, 'features.h5')
                features.to_hdf(savepath, 'table')
                logging.info('feature selection results saved to ' + str(savepath))

        # -- load or compute segmentation
        if Path(self.savedir_v, 'segmented_mask.nc').is_file() and not self.overwrite:
            mask_dataset = xr.open_mfdataset(Path(self.savedir_v, 'segmented_mask.nc'))
            # segmented_features = pd.read_hdf(Path(self.savedir_v, 'segmented_features.h5'), 'table')

        elif segment:
            segmented_mask, segmented_features = tobac.segmentation.segmentation(features, self.segment_data, dxy, **di['params_segmentation'])
            # -- transform mask dataset to xarray
            mask_dataset = xr.DataArray.from_iris(segmented_mask).to_dataset(name='feature')
            mask_dataset['feature'].attrs['description'] = 'tobac features after segmentation'
            
            if self.save:
                mask_dataset.where(mask_dataset > 0, NAN).astype(np.int32).to_netcdf(Path(self.savedir_v, 'segmented_mask.nc'))
                # segmented_features.to_hdf(Path(self.savedir_v, 'segmented_features.h5'), 'table')
                logging.info('feature segmentation results saved to ' + self.savedir_v)
            
        # -- load or compute tracking
        if Path(self.savedir_v, 'tracked_features.h5').is_file() and not self.overwrite_tracks:
            tracks = pd.read_hdf(Path(self.savedir_v, 'tracked_features.h5'), 'table')

        elif track:
            tracks = tobac.linking_trackpy(self.features, None, dt=dt, dxy=dxy, **di['params_linking'])
            # -- add to mask dataset
            if self.mask_dataset is not None:
                mask_dataset = self.mask_dataset
            mask_dataset['cell'] = self._table_to_dataset(table=tracks, col='cell', dataset=mask_dataset)
            mask_dataset['cell'] = mask_dataset.cell
            mask_dataset['cell'].attrs['description'] = 'tobac features after tracking'
            mask_dataset = mask_dataset.where(mask_dataset > 0)

            if self.save:
                mask_dataset.astype(np.int32).to_netcdf(Path(self.savedir_v, 'tracked_mask.nc'))
                tracks.to_hdf(Path(self.savedir_v, 'tracked_features.h5'), 'table')
                logging.info('tracking results saved to ' + str(Path(self.savedir_v, 'tracks.h5')))
        
        # -- load or compute tobac merges and splits
        if Path(self.savedir_v, 'merged_split_mask.nc').is_file() and not self.overwrite_tracks:
            mask_dataset = xr.open_mfdataset(Path(self.savedir_v, 'merged_split_mask.nc'))

        elif merge:
            merges = tobac.merge_split.merge_split_MEST(tracks, dxy=dxy)
            # -- add to mask dataset
            tracks["merged"] = (merges.feature_parent_track_id.data+1).astype(np.int32)
            mask_dataset['merged'] = self._table_to_dataset(table=tracks, col='merged', dataset=mask_dataset)
            mask_dataset['merged'] = mask_dataset.merged
            mask_dataset['merged'].attrs['description'] = 'tobac features after track merging and splitting'

            if self.save:
                mask_dataset.to_netcdf(Path(self.savedir_v, 'merged_split_mask.nc'))
                tracks.to_hdf(Path(self.savedir_v, 'tracked_features.h5'), 'table')
                logging.info('tobac merge/split results saved to ' + str(Path(self.savedir_v, 'merged_split_mask.nc')))

        # -- load or compute custom merge
        if Path(self.savedir_v, 'connected_merge_mask.nc').is_file() and not self.overwrite_tracks:
            mask_dataset = xr.open_mfdataset(Path(self.savedir_v, 'connected_merge_mask.nc'))

        elif connect:
            result = Connect(mask_dataset['cell'].data > 0, method=merge_method).get_components()
            # -- add to mask dataset
            mask_dataset['connected_merge'] = (mask_dataset.dims, result)
            mask_dataset['connected_merge'] = mask_dataset['connected_merge'].where(mask_dataset['connected_merge'] > 0)
            mask_dataset['merged'].attrs['description'] = 'tobac features after cumstom merge'
            if self.save:
                mask_dataset.to_netcdf(Path(self.savedir_v, 'connected_merge_mask.nc'))
                logging.info('custom merging results saved to ' + str(Path(self.savedir_v, 'connected_merge_mask.nc')))
            
        # -- output
        if connect or merge or track:
            return mask_dataset, tracks
        elif segment:
            return mask_dataset, features
        elif detect:
            return features
        else:
            raise ValueError("You haven't selected any steps to perform")

