'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import xarray as xr
import os
import numpy as np
from pathlib import Path
from typing import Any, Union, Callable
import datetime
import logging
import scipy.ndimage  as ndi
import concurrent.futures
from . import (Connect, Erode, Track, Helpers)

'''

Collection of methods to combine tracks of two cloud features into one tracked system with constituent components. 

'''


class Methods(Erode, Connect):

    def __init__(self) -> None:
        pass

    def mutlitrack_eroded(self, parent_mask, child_mask, parent_eroded=None, child_eroded=None, verbose=True, PBC_flag=None):
        ''' Here, the 'parent' is the feature required to meet the definition for a system (in our case, this is the convective cores). All parent features will be retained. The 'child' is the feature that we add to the parent system. Only child features that at some point in time and space overlap with a parent feature will be kept. '''

        if not isinstance(parent_eroded, xr.DataArray):
            parent_eroded = parent_mask
        if not isinstance(child_eroded, xr.DataArray):
            child_eroded = child_mask
        
        # (1) get the subset of features in child that overlap with parent
        child_eroded_linked = self.subset_that_overlaps(parent_eroded, child_eroded)
        child_linked = self.subset_that_overlaps(parent_mask, child_mask)

        # (2) create new labels using the union of all data
        combined_boolean = (parent_eroded > 0) + (child_eroded_linked > 0)
        arr = Connect(combined_boolean>0).get_components(PBC_flag=PBC_flag)
        new_labels = xr.DataArray(data=arr, dims=parent_eroded.dims, coords=parent_eroded.coords)

        # (3) share resulting labels with uneroded system
        parent_eroded_renumbered = new_labels.where(parent_eroded>0)
        child_eroded_renumbered = new_labels.where(child_eroded>0)
        logging.info(f"{datetime.datetime.now()} sharing parent features")
        parent_new = self.share_labels_memory(parent_mask, parent_eroded_renumbered)
        logging.info(f"{datetime.datetime.now()} sharing child features")
        child_new = self.share_labels_memory(child_linked, child_eroded_renumbered)
        logging.info(f"{datetime.datetime.now()} sharing complete")
        
        # (4) collect results
        ds = parent_new.to_dataset(name='parent') # out data
        ds['child'] = child_new
        ds['system'] = self.union(ds.parent, ds.child)
        ds = ds.where(ds>0)
        # retain non-overlapping input masks
        if verbose:
            ds['parent_in'] = parent_mask
            ds['child_in'] = child_mask
            # add attributes
            ds['parent_in'].attrs['description'] = 'parent tracks before multivariate label assignment'
            ds['child_in'].attrs['description'] = 'child tracks before multivariate label assignment'
        ds['parent'].attrs['description'] = 'parent tracks after multivariate labels assigned'
        ds['child'].attrs['description'] = 'child tracks after multivariate labels assigned'
        return ds

    def mutlitrack_additive(self, parent_mask, child_mask, verbose=True, PBC_flag=None):
        ''' Here, the 'parent' is the feature required to meet the definition for a system (in our case, this is the convective cores). All parent features will be retained. The 'child' is the feature that we add to the parent system. Only child features that at some point in time and space overlap with a parent feature will be kept. '''

        # (1) get the subset of features in child that overlap with parent
        child_linked = self.subset_that_overlaps(parent_mask, child_mask)
        # (2) create new labels using the union of all data
        combined_boolean = (parent_mask > 0) + (child_linked > 0) # where one or both features, create labels
        arr = Connect(combined_boolean>0).get_components(PBC_flag=PBC_flag)
        new_labels = xr.DataArray(data=arr, dims=parent_mask.dims, coords=parent_mask.coords)
        # (3) assign combined labels to both datasets
        parent_renumbered = new_labels.where(parent_mask>0)
        child_renumbered = new_labels.where(child_linked>0)
        # (4) results
        ds = parent_renumbered.to_dataset(name='parent') # out data
        ds['child'] = child_renumbered
        ds = ds.where(ds>0)
        ds['system'] = self.union(ds.parent, ds.child)
        # retain non-overlapping input masks
        if verbose:
            ds['parent_in'] = parent_mask
            ds['child_in'] = child_mask
            # add attributes
            ds['parent_in'].attrs['description'] = 'parent tracks before multivariate label assignment'
            ds['child_in'].attrs['description'] = 'child tracks before multivariate label assignment'
        ds['parent'].attrs['description'] = 'parent tracks after multivariate labels assigned'
        ds['child'].attrs['description'] = 'child tracks after multivariate labels assigned'
        return ds
    
    def mutlitrack_subset(self, parent_mask, child_mask, verbose=True, PBC_flag=None):

        # assign parent values to overlapping data of child
        child_overlap = parent_mask*(child_mask>0)
        # check which features from parent were in child
        shared_features = np.unique(child_overlap.values)
        # drop unshared features from parent
        parent_overlap = parent_mask.where(parent_mask.isin(shared_features))
        # renumber features
        y1 = Connect(parent_overlap>0).get_components(PBC_flag=PBC_flag)
        ds = parent_overlap.to_dataset(name='parent')
        ds['parent'] = (tuple(ds.dims), y1)
        ds['child'] = ds.parent * (child_overlap>0)
        # mask 0s
        ds = ds.where(ds>0)
        # retain non-overlapping input masks
        if verbose:
            ds['parent_in'] = parent_mask
            ds['child_in'] = child_mask
            # add attributes
            ds['parent_in'].attrs['description'] = 'parent tracks before multivariate label assignment'
            ds['child_in'].attrs['description'] = 'child tracks before multivariate label assignment'
        ds['parent'].attrs['description'] = 'parent tracks after multivariate labels assigned'
        ds['child'].attrs['description'] = 'child tracks after multivariate labels assigned'
        return ds
    

    ''' ------- helpers -------- '''
    
    def share_labels_memory(self, current, update):
        ''' Features with the same label in current won't be split up by update '''
        flipped_labels = self.share_labels(update, current, nan_val=0)
        return self.share_labels(current, flipped_labels, nan_val=1e10) # using 1e10 (not 0) as NaN avoids the need to mask 0s in np.min, much faster

    def share_labels_slow(self, current, update, nan_val=0):
        dims, coords, shape = update.dims, update.coords, update.shape
        current = current.fillna(nan_val).values.astype(int)
        update = update.fillna(nan_val).values.astype(int)
        # find values to replace feature labels of current with feature labels from update
        # where multiple in update for given feature in current, take min
        min_label = ndi.labeled_comprehension(input=update.ravel(), 
                                  labels=current.ravel(), 
                                  index=current.ravel(), 
                                  func=np.min, 
                                  out_dtype=np.int64, 
                                  default=nan_val,
                                  pass_positions=False
                                  )
        min_label = min_label.reshape(update.shape)
        # replace
        label_map = np.zeros(int(np.nanmax(current)+1), dtype=int) # Create empty array of length labels.max + 1
        label_map[current] = min_label # For each value in labels, assign the new value from the list comprehension
        new_labels = label_map[current] # Index to create new labels
        # ax xarray
        new_labels = np.where(new_labels == nan_val, 0, new_labels)
        _, consecutive = np.unique(new_labels.ravel(), return_inverse=1) # force consecutive numbers
        result = xr.DataArray(data=consecutive.reshape(shape), dims=dims, coords=coords)
        return result.where(result != nan_val, np.nan)

    def share_labels(self, current, update, nan_val=0):
        dims, coords, shape = update.dims, update.coords, update.shape
        current = current.fillna(nan_val).values.astype(int)
        update = update.fillna(nan_val).values.astype(int)
        current_flat = current.ravel()
        update_flat = update.ravel()
        # processor function
        def process_chunk(chunk):
            return ndi.labeled_comprehension(input=chunk['update'], 
                                             labels=chunk['current'], 
                                             index=chunk['current'], 
                                             func=np.min,
                                             out_dtype=np.int64, 
                                             default=nan_val,
                                             pass_positions=False)
        # chunk data
        chunk_size = len(current_flat) // 4  # Adjust the number of chunks as needed
        chunks = [{'current': current_flat[i:i + chunk_size], 'update': update_flat[i:i + chunk_size]} 
                  for i in range(0, len(current_flat), chunk_size)]
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_chunk, chunks))
        min_label = np.concatenate(results).reshape(shape) # combine results
        # replace labels
        label_map = np.zeros(int(np.nanmax(current) + 1), dtype=int)
        label_map[current] = min_label
        new_labels = label_map[current]
        new_labels = np.where(new_labels == nan_val, 0, new_labels)
        _, consecutive = np.unique(new_labels.ravel(), return_inverse=1) # force consecutive numbers
        result = xr.DataArray(data=consecutive.reshape(shape), dims=dims, coords=coords)
        return result.where(result != nan_val, np.nan)

    def find_matches(self, target, cond, k):
        ''' Find all values in target that overlap with k in cond. '''
        out = np.unique(target.where(cond == k).values)
        return tuple(out[~np.isnan(out)])
    
    def union(self, da1, da2):
        ''' Compute the union of the two labelled arrays. '''
        ds_sum = da1.fillna(0) + da2.fillna(0)
        non_null_count = (da1.notnull() * 1 + da2.notnull() * 1)
        da_union = ds_sum / non_null_count
        da_union = da_union.where(da_union>0)
        return da_union
    
    def subset_that_overlaps(self, parent, child):
        ''' Compute the subset of features in child that overlap with parent. '''
        child_overlap = child.where(parent>0) # child features coinciding with the parent
        overlapping_features = np.unique(child_overlap.values) # values of these
        return child.where(child.isin(overlapping_features))
        
        

class CustomTrack(Methods, Track, Connect, Helpers):

    '''
    This class runs a multivariate tobac tracking algorithm for the provided xr.DataArrays. This performs standard tobac tracking seperately on each (parent, child) dataarray, then uses the resulting masks to create a new mask.
     
    Method: a 'child' that overlaps at any point in space/time with 'parent' are assigned the labels of 'parent', does not require every 'parent' to have a 'child'

    Example useage:
        from my_library.multivariate_tobac import CustomTrack
        module = CustomTrack(parent_data, child_data, parent_option, child_options, savedir)
        result = module.perform(save=True)

    '''

    def __init__(self, parent_data, child_data, parent_options: Union[str, dict], child_options: Union[str, dict], savedir=Union[str,None], overwrite: bool=True, track_params: dict={}, PBC_flag=None):

        Helpers()
        Methods()

        self.parent_data = parent_data
        self.child_data = child_data
        self.parent_options = parent_options
        self.child_options = child_options
        self.overwrite = overwrite
        self.savedir = savedir
        self.track_params = track_params
        self.PBC_flag = PBC_flag

        if savedir:
            os.makedirs(self.savedir, exist_ok=True)
        else:
            self.savedir = Path('.')


    def perform(self, save: bool=True):

        # perform tracking on each variable
        self.parent_mask, _ = self.track_dataset(track='parent')
        self.child_mask, _ = self.track_dataset(track='child')

        # determine the multivariate tracks
        self.result = self.mutlitrack_additive(self.parent_mask.merged, self.child_mask.merged, verbose=False, PBC_flag=self.PBC_flag)

        # add all results to dataset
        for result in ['feature', 'cell', 'merged']:
            self.result[f'parent_{result}'] = self.parent_mask[result].where(self.parent_mask[result]>0)
            self.result[f'child_{result}'] = self.child_mask[result].where(self.child_mask[result]>0)

        # save results
        if save:
            self.mutlitrack_dir = f'{self.savedir}/multitrack.nc'
            self.result.to_netcdf(self.mutlitrack_dir)
            logging.info('multivariate results saved to ' + str(self.mutlitrack_dir))
    
        return self.result

    def track_dataset(self, track: str):

        if track == 'parent':
            data = self.parent_data
            options = self.parent_options
            self.track_params['savedir'] = Path(self.savedir, 'parent')

        elif track == 'child':
            data = self.child_data
            options = self.child_options
            self.track_params['savedir'] = Path(self.savedir, 'child')

        else:
            raise ValueError('track must be either "parent" or "child"')
        
        di = self._load_yaml(options)
        sel, seg = data[di['select_data']], data[di['segment_data']]
        
        module = Track(sel, seg, options, self.overwrite, self.track_params, save=False)
        mask = module.perform(merge=True)

        return mask
    


    
class MultiTrack(Methods, Track, Connect):

    '''
    This class runs a multivariate tobac tracking algorithm for the provided xr.DataArrays. This performs standard tobac tracking seperately on each (parent, child) dataarray, then uses the resulting masks to create a new mask in that assumes that 'child' is a contained within 'parent', requires every 'parent' to have a 'child', and assigns the labels of 'parent' to the contained 'child'

    Example useage:
        from my_library.multivariate_tobac import MultiTrack
        module = MultiTrack(parent_data, child_data, parent_option, child_options, savedir)
        result = module.perform(save=True)

    '''

    def __init__(self, parent_data, child_data, parent_options: Union[str, dict], child_options: Union[str, dict], savedir=Union[str,None], overwrite: bool=True, track_params: dict={}, PBC_flag=None):

        Methods()

        self.parent_data = parent_data
        self.child_data = child_data
        self.parent_options = parent_options
        self.child_options = child_options
        self.overwrite = overwrite
        self.savedir = savedir
        self.track_params = track_params
        self.PBC_flag = PBC_flag

        if savedir:
            os.makedirs(self.savedir, exist_ok=True)

    def perform(self, save: bool=True):

        # perform tracking on each variable
        self.parent_mask = self.track_dataset(track='parent')
        self.child_mask = self.track_dataset(track='child')

        # determine the multivariate tracks
        self.result = self.mutlitrack_subset(self.parent_mask.merged, self.child_mask.merged, verbose=False, PBC_flag=self.PBC_flag)

        # add all results to dataset
        for result in ['feature', 'cell', 'merged']:
            self.result[f'parent_{result}'] = self.parent_mask[result].where(self.result['parent']>0)
            self.result[f'child_{result}'] = self.child_mask[result].where(self.result['child']>0)

        # save results
        if save:
            self.mutlitrack_dir = f'{self.savedir}/multitrack.nc'
            self.result.to_netcdf(self.mutlitrack_dir)
            logging.info('multivariate results saved to ' + str(self.mutlitrack_dir))
    
        return self.result

    def track_dataset(self, track: str):

        if track == 'parent':
            data = self.parent_data
            options = self.parent_options
            self.track_params['savedir'] = Path(self.savedir, 'parent')

        elif track == 'child':
            data = self.child_data
            options = self.child_options
            self.track_params['savedir'] = Path(self.savedir, 'child')

        else:
            raise ValueError('track must be either "parent" or "child"')
        
        module = Track(data, data, options, self.overwrite, self.track_params, save=False)
        mask, _ = module.perform(merge=True)

        return mask