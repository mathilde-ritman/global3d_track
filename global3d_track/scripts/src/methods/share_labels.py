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
import dask
import pathlib
from ..utils import tools

class ShareLabels:

    '''
    This class provides large-dataset-friendly methods to share labels from one field to another.

    '''

    def __init__(self, current=None, update=None, current_col=None, update_col=None, nan_val=1e10, n_jobs=-1):
        '''
        nan_val: needs to be larger than any label in the input dataarrays
        n_jobs: numper of jobs to processs in parrallel when searching for matching features between the two input arrays
        don't bother with the checkpoint options unless you are using the class I've built, it will break.
        '''
        self.nan_val=nan_val
        self.n_jobs=n_jobs
        self.current = current
        self.update = update
        self.current_col = current_col
        self.update_col = update_col


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

        return result

    def tobac_like(self, table_path=None, save_table=True, return_extra_mappings=False):
        '''
        Share the coincident labels of 'update' to 'current' but with record keeping. Returns the resulting dataarray and saves a new pandas table at the same directory as 'table_path' with the changes recorded.
        current: integer dataarray
        update: integer dataarray
        table_path: path to the tobac-like feature table
        current_col: column name in the tobac-like feature table corresponding to the current feature labels in 'current'
        update_col: name to call the new column that will record the mappings applied by this function
        '''

        # 1a. initialise arrays
        current = self.current.where(self.current > 0)
        update = self.update.where(self.update > 0)

        # 1b. initialise table record
        if table_path is not None and os.path.exists(table_path):
            table_path = pathlib.Path(table_path)
            df = pd.read_hdf(table_path, 'table') # load feature table
        else:
            current_vals = np.unique(current.values)
            current_vals = current_vals[~np.isnan(current_vals)] # only valid labels
            df = pd.DataFrame({self.current_col: current_vals})

        # 2. find mappings, but first check if they exist already
        if self.update_col in df.columns and not return_extra_mappings:
            logging.info(f"{datetime.now()} Loaded mappings from table at {table_path}")

        else:
            logging.info(f"{datetime.now()} Finding label mappings...")
            mappings, extra_mappings = self.find_labels_parallel() # derive
            df = self.mappings_to_dataframe(df, mappings)
            # collect extra mappings
            update_vals = np.unique(update.values)
            extra_df = pd.DataFrame({self.update_col: update_vals[~np.isnan(update_vals)]})
            extra_df = self.mappings_to_dataframe(extra_df, extra_mappings, self.update_col, f"{self.update_col}_new")
            # 3. save the results as a pandas dataframe
            if save_table:
                df.to_hdf(table_path, 'table')
                logging.info(f"{datetime.now()} saved table to {table_path}.")

        # 4. apply the mappings to the mask dataset
        logging.info(f"{datetime.now()} Applying label mappings to dataset...")
        dataset = current.fillna(0).to_dataset(name=self.current_col)
        result = self.apply_mapping(dataset, df, self.current_col, self.update_col)
        logging.info(f"{datetime.now()} done.")

        output = (result.rename(self.update_col), )
        if not save_table:
            output += (df, )
        if return_extra_mappings:
            output += (extra_df, )
        if len(output) == 1:
            output = output[0]
        return output

    #### --------------------- helpers ----------------------- ####

    def mappings_to_dataframe(self, df, mappings, current_col=None, update_col=None):
        if not current_col:
            current_col = self.current_col
        if not update_col:
            update_col = self.update_col
        df[update_col] = df[current_col].map(mappings).fillna(0) # record as new column
        df[update_col] = df[update_col].astype(int) # make sure it's int
        df[current_col] = df[current_col].astype(int)
        return df

    def table_to_dataset(self, table, col, dataset, base_col='feature'):
        di = table.set_index(base_col)[col].to_dict()
        feature_array = dataset[base_col].values
        dataarray = xr.DataArray(np.vectorize(lambda x: di.get(x, 0))(feature_array), dims=dataset[base_col].dims, coords=dataset[base_col].coords)
        return dataarray
    
    def apply_mapping(self, ds, df, old, new):
        mapping = df.set_index(old)[new].to_dict() # map dict
        da = ds[old] # should be a dask array
        if not isinstance(da.data, dask.array.Array):
            logging.info(f"{datetime.now()} Converting to dask array for mapping...")
            da = dask.array.from_array(da.data, chunks=tools.get_chunks(ds))
        else:
            logging.info(f"{datetime.now()} Already a dask array, proceeding with mapping...")
            da = da.chunk(tools.get_chunks(ds)).data
        # create 1d mapper array
        mapper = np.full(df[old].max()+1, 0, dtype=np.int64)
        for k, v in mapping.items():
            mapper[k] = v
        # lookup func
        def _map_block(vals):
            return mapper[vals.astype(int)]
        # apply mapping to dask chunks
        result = da.map_blocks(_map_block, dtype=np.int64)
        return xr.DataArray(result, dims=ds[old].dims, coords=ds[old].coords)

    def find_labels(self, chunk, nan_val=1e10):
        result =  ndi.labeled_comprehension(input=chunk['update'], 
                                        labels=chunk['current'], 
                                        index=chunk['index'], 
                                        func=np.min,
                                        out_dtype=np.int64,
                                        default=nan_val,
                                        pass_positions=False)
        return result

    def proc_optimized(self, list_A, list_B):
        ''' Optimise mappings, by sending repeated mappings from A to B to the minimum value of B. In doing so, record the values of B that coincided with A but were not chosen for the final mapping, this is in case you later want to ensure B is consistent '''
        a_mappings = {} # i: min(j); for mapping i->min(j)
        j_values = {} # i: all js;  for mapping j->min(j)
        for i, j in zip(list_A, list_B):
            if i in a_mappings:
                a_mappings[i] = min(a_mappings[i], j)  # Update to min j
                j_values[i] += [j]  # Collect all j values for i 
            else:
                a_mappings[i] = j  # First occurrence of i
                j_values[i] = [j]  # Start collecting j values for i
        # collect map values to format j: min(j)
        b_mappings = {}
        for i, js in j_values.items():
            min_j = a_mappings[i]
            for j in js:
                b_mappings[j] = min_j
        # ensure validity
        def valid_maps(di):
            keys = np.array(list(di.keys())).astype(int)
            vals = np.array(list(di.values()))
            vals = np.where(vals != self.nan_val, vals, 0).astype(int) # only valid mappings
            return dict(zip(keys, vals))
        return valid_maps(a_mappings), valid_maps(b_mappings)
        
    #### ------------------------- operators ------------------- ####
    
    def find_labels_parallel(self):
        current = self.current.where(self.current > 0)
        update = self.update.where(self.update > 0)
        current = current.fillna(self.nan_val).astype(int)
        update = update.fillna(self.nan_val).astype(int)
        ntimes = current.time.size if 'time' in current.dims else 1

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
        mappings, extra_mappings = self.proc_optimized(all_index_vals, all_results)
        logging.info(f"{datetime.now()} {n_init} label maps reduced to {len(mappings.keys())}")
        return mappings, extra_mappings
