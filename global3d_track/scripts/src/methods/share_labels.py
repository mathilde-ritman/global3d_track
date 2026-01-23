'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import xarray as xr
import numpy as np
from scipy import ndimage as ndi
import datetime
import os
import logging
from datetime import datetime
import joblib
import time
import pandas as pd
import pathlib

class ShareLabels:

    '''
    This class provides large-dataset-friendly methods to share labels from one field to another.

    '''

    def __init__(self, nan_val=1e10, n_jobs=-1, checkpoint=None, checkpoint_name=None):
        '''
        nan_val: needs to be larger than any label in the input dataarrays
        n_jobs: numper of jobs to processs in parrallel when searching for matching features between the two input arrays
        don't bother with the checkpoint options unless you are using the class I've built, it will break.
        '''
        self.nan_val=nan_val
        self.n_jobs=n_jobs
        self.checkpoint=checkpoint
        self.checkpoint_name=checkpoint_name

    #### ----------------------- main functions to call, two options ----------------------- ####

    def dataarrays(self, current: xr.DataArray, update: xr.DataArray):
        '''
        Share the coincident labels of 'update' to 'current' and return the resulting dataarray.
        current: integer dataarray
        update: integer dataarray
        '''
        
        # 1. find mappings, or load them from the checkpoint
        checkdir = f'{self.checkpoint_name}label_mappings/'
        if self.checkpoint is not None and self.checkpoint.checkpoint_reached(f'{checkdir}min_k_in_data'):  
            # load them
            index = self.checkpoint.load_array(f'{checkdir}all_index_vals').tolist()
            new = self.checkpoint.load_array(f'{checkdir}all_results').tolist()

        else:
            # find them
            index, new = self.find_labels_parallel(current, update)
            # and checkpoint them if you want
            if self.checkpoint is not None:
                self.checkpoint.checkpoint_array(np.array(index), f'{checkdir}index_vals')
                self.checkpoint.checkpoint_array(np.array(new), f'{checkdir}new_vals')

        # 2. collect the mappings in a table
        mapping = dict(zip(index, new)) # as dict
        df = pd.DataFrame({'current': np.unique(current.values)})
        df['update'] = df['current'].map(mapping).fillna(0) # record as new column

        # 3. apply the mappings to the mask dataset
        logging.info(f"{datetime.now()} Applying label mappings to dataset...")
        dataset = current.to_dataset(name='current')
        result = self.table_to_dataset(df, 'update', dataset, 'current')
        logging.info(f"{datetime.now()} done.")

        return result.where(result>0)

    def tobac_like(self, current: xr.DataArray, update: xr.DataArray, table_path: str, current_col: str, update_col: str, new_tobac_table=True):
        '''
        Share the coincident labels of 'update' to 'current' but with record keeping. Returns the resulting dataarray and saves a new pandas table at the same directory as 'table_path' with the changes recorded.
        current: integer dataarray
        update: integer dataarray
        table_path: path to the tobac-like feature table
        current_col: column name in the tobac-like feature table corresponding to the current feature labels in 'current'
        update_col: name to call the new column that will record the mappings applied by this function
        new_tobac_table: whether to save a new table instead of adding a column to the existing one
        '''
        
        # 1. find mappings, or load them from the checkpoint
        checkdir = f'{self.checkpoint_name}label_mappings/'
        if self.checkpoint is not None and self.checkpoint.checkpoint_reached(f'{checkdir}min_k_in_data'):  
            # load them
            index = self.checkpoint.load_array(f'{checkdir}all_index_vals').tolist()
            new = self.checkpoint.load_array(f'{checkdir}all_results').tolist()

        else:
            # find them
            index, new = self.find_labels_parallel(current, update)
            # and checkpoint them if you want
            if self.checkpoint is not None:
                self.checkpoint.checkpoint_array(np.array(index), f'{checkdir}index_vals')
                self.checkpoint.checkpoint_array(np.array(new), f'{checkdir}new_vals')

        # 2. record the mappings in the tobac feature table
        logging.info(f"{datetime.now()} Applying label mappings to dataframe...")
        table_path = pathlib.Path(table_path)
        if os.path.exists(table_path):
            df = pd.read_hdf(table_path, 'table') # load feature table
        else:
            df = pd.DataFrame({current_col: np.unique(current.values)})
        mapping = dict(zip(index, new)) # as dict
        # apply
        df[update_col] = df[current_col].map(mapping).fillna(0) # record as new column
        # save the table
        if new_tobac_table:
            outpath = table_path.with_name(f"{table_path.stem}_{update_col}{table_path.suffix}")
        else:
            outpath = table_path
        df.to_hdf(outpath, 'table')
        logging.info(f"{datetime.now()} saved table to {outpath}.")

        # 3. apply the mappings to the mask dataset
        logging.info(f"{datetime.now()} Applying label mappings to dataset...")
        dataset = current.to_dataset(name=current_col)
        result = self.table_to_dataset(df, update_col, dataset, current_col)
        logging.info(f"{datetime.now()} done.")

        return result.where(result>0)

    #### --------------------- helpers ----------------------- ####

    def table_to_dataset(self, table, col, dataset, base_col='feature'):
        di = table.set_index(base_col)[col].to_dict()
        feature_array = dataset[base_col].values
        dataarray = xr.DataArray(np.vectorize(lambda x: di.get(x, -9))(feature_array), dims=dataset[base_col].dims, coords=dataset[base_col].coords)
        return dataarray

    def compute_share_labels(self, chunk, nan_val=1e10):
        label_map = np.zeros_like(chunk['update'], dtype=int)
        result =  ndi.labeled_comprehension(input=chunk['update'], 
                                        labels=chunk['current'], 
                                        index=chunk['index'], 
                                        func=np.min,
                                        out_dtype=np.int64,
                                        default=nan_val,
                                        pass_positions=False)
        label_map[chunk['current'] == chunk['index']] = result
        return label_map
    
    def find_labels(self, chunk, nan_val=1e10):
        result =  ndi.labeled_comprehension(input=chunk['update'], 
                                        labels=chunk['current'], 
                                        index=chunk['index'], 
                                        func=np.min,
                                        out_dtype=np.int64,
                                        default=nan_val,
                                        pass_positions=False)
        return result

    def update_labels(self, chunk):
        label_map = np.zeros_like(chunk['current'], dtype=int)
        label_map[chunk['current'] == chunk['index']] = chunk['result'] # the 'result' is one integer label
        return label_map
    
    def proc_optimized(self, list_A, list_B):
        min_values = {}
        for i, j in zip(list_A, list_B):
            if i in min_values:
                min_values[i] = min(min_values[i], j)  # Update to min j
            else:
                min_values[i] = j  # First occurrence of i
        unique_A = list(min_values.keys())
        unique_B = list(min_values.values())
        return unique_A, unique_B
        
    #### ------------------------- operators ------------------- ####
    
    def find_labels_parallel(self, current, update):
        current = current.fillna(self.nan_val).astype(int)
        update = update.fillna(self.nan_val).astype(int)
        ntimes = len(current.time) if 'time' in current.dims else 1

        def index_data(t, index_vals=None):
            current_arr = current.isel(time=t).values.reshape(-1) if ntimes > 1 else current.values.reshape(-1)
            update_arr = update.isel(time=t).values.reshape(-1) if ntimes > 1 else update.values.reshape(-1)
            if index_vals is None:
                index_vals = [x for x in np.unique(current_arr) if x > 0 and x != self.nan_val]
            di = {'current':current_arr,
                  'update':update_arr,}
            return index_vals, di

        # find mappings
        logging.info(f"{datetime.now()} Finding label maps: {ntimes} iterations")
        durations, all_results, all_index_vals = [], [], [] # store results of each time step
        for t in range(ntimes):
            if t % 10 == 0:
                logging.info(f"{datetime.now()} ({t}/{ntimes})...")
            start_time = time.time()
            # slices[0] = t if ntimes > 1 else slice(None)
            index_vals, di = index_data(t, index_vals=None)
            di = {k: {**di, 
                      'index':k,
                      } for k in index_vals}
            results = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")(joblib.delayed(self.find_labels)(di[k], self.nan_val) for k in index_vals) # process
            all_index_vals.extend(index_vals) # collect mappings
            all_results.extend(results)
            durations.append(time.time() - start_time)
        logging.info(f"{datetime.now()} Avg duration: {sum(durations)/len(durations):.4f} seconds")

        # clean mappings
        n_init = len(all_results)
        all_index_vals, all_results = self.proc_optimized(all_index_vals, all_results)
        logging.info(f"{datetime.now()} {n_init} label maps reduced to {len(all_results)}")
        return all_index_vals, all_results
