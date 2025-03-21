'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import xarray as xr
import numpy as np
from scipy import ndimage as ndi
import datetime
import logging
from datetime import datetime
import joblib
import time


class ShareLabels:

    '''
    This class provides large-dataset-friendly methods to share labels from one field to another.

    '''

    def __init__(self):
        pass

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
    
    # main    
    def share_labels_parallel(self, current, update, nan_val=1e10, n_jobs=-1, n_k=250):
        '''
        Identifies the labels of features in the current field (L={...}) that overlap with features in the update field (L_update=integer). Then assigns L_update to each l in L. Uses multiprocessing and iteration to do each time step independently, but retains memory of the changes so that results are not impacted by iterating time.
        Parameters:
            current: xr.DataArray
                The current field of labels to be updated
            update: xr.DataArray
                The field of labels to use to update the current field
            nan_val: int
                The value to use for missing values is ndi.labelled_comprehension
            n_jobs: int
                The number of jobs to
            n_k: int
                The batch size to use when applying the label mappings in parallel. A larger batch size will use more memory but may be faster.
        '''
        # initalise
        current = current.fillna(nan_val).astype(int)
        update = update.fillna(nan_val).astype(int)
        new_labels = np.zeros(update.shape, dtype=int)
        ntimes = len(current.time) if 'time' in current.dims else 1
        update_tshape = current.isel(time=0).shape if ntimes > 1 else update.shape
        durations, all_results, all_index_vals = [], [], [] # store results of each time step
        max_k_in_data, min_k_in_data = {}, {}
        slices = [slice(None)] * len(current.dims)

        def index_data(t, index_vals=None):
            current_arr = current.isel(time=t).values.reshape(-1) if ntimes > 1 else current.values.reshape(-1)
            update_arr = update.isel(time=t).values.reshape(-1) if ntimes > 1 else update.values.reshape(-1)
            if index_vals is None:
                index_vals = [x for x in np.unique(current_arr) if x > 0 and x != nan_val]
            di = {'current':current_arr,
                  'update':update_arr,}
            return index_vals, di

        # iterate times to collect label mappings
        logging.info(f"{datetime.now()} Finding label maps: {ntimes} iterations")
        for t in range(ntimes):
            if t % 10 == 0:
                logging.info(f"{datetime.now()} ({t}/{ntimes})...")
            start_time = time.time()
            index_vals, di = index_data(t, index_vals=None)
            di = {k: {**di, 
                      'index':k,
                      } for k in index_vals}
            results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(joblib.delayed(self.find_labels)(di[k], nan_val) for k in index_vals) # process
            max_k_in_data[t] = np.max(index_vals) # record max k for each time
            min_k_in_data[t] = np.min(index_vals)
            all_index_vals.extend(index_vals) # collect mappings
            all_results.extend(results)
            durations.append(time.time() - start_time)
        logging.info(f"{datetime.now()} Avg duration: {sum(durations)/len(durations):.4f} seconds")

        # clean mappings
        n_values = len(all_results)
        all_index_vals, all_results = self.proc_optimized(all_index_vals, all_results)
        logging.info(f"{datetime.now()} {n_values} label maps reduced to {len(all_results)}")

        # iterate times again to apply all mappings
        durations = []
        logging.info(f"{datetime.now()} Applying all label maps: {ntimes} iterations")
        logging.info(f"{datetime.now()} {len(all_index_vals)} values, batch size = {n_k}")
        for t in range(ntimes):
            if t % 10 == 0:
                logging.info(f"{datetime.now()} ({t}/{ntimes})...")
            start_time = time.time()
            slices[0] = t if ntimes > 1 else slice(None)
            all_index_vals, di = index_data(t, index_vals=all_index_vals)
            di = {k: {**di, 
                      'index':k,
                      'result':all_results[i],
                      } for i,k in enumerate(all_index_vals)}
            # logging.info(f"{datetime.now()} dataset time {t} has value range ({min_k_in_data[t]}, {max_k_in_data[t]})")
            for batch in range(0, len(all_index_vals), n_k):
                batch_index_vals = all_index_vals[batch:batch+n_k]
                # skip if there are no k in the current dataset for this batch
                min_k = np.min(batch_index_vals)
                max_k = np.max(batch_index_vals)
                # if not t:
                    # logging.info(f"{datetime.now()} batch ({batch}:{batch+n_k}) has value range ({min_k}, {max_k})")
                if min_k > max_k_in_data[t] or max_k < min_k_in_data[t]:
                    # logging.info(f"{datetime.now()} skipping batch ({batch}:{batch+n_k}) as batch vals ({min_k}, {max_k}) not in dataset vals ({min_k_in_data[t]}, {max_k_in_data[t]})")
                    continue
                results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(joblib.delayed(self.update_labels)(di[k]) for k in batch_index_vals)
                for label_map in results:
                    new_labels[tuple(slices)] += label_map.reshape(update_tshape) # apply all mappings
            durations.append(time.time() - start_time)
        logging.info(f"{datetime.now()} Avg duration: {sum(durations)/len(durations):.4f} seconds")

        da = xr.DataArray(data=new_labels, dims=update.dims, coords=update.coords)
        da = da.where(da != nan_val)
        return da.where(da>0)
        
    def find_labels_parallel(self, current, update, nan_val=1e10, n_jobs=-1):
        current = current.fillna(nan_val).astype(int)
        update = update.fillna(nan_val).astype(int)
        ntimes = len(current.time) if 'time' in current.dims else 1
        slices = [slice(None)] * len(current.dims)

        def index_data(t, index_vals=None):
            current_arr = current.isel(time=t).values.reshape(-1) if ntimes > 1 else current.values.reshape(-1)
            update_arr = update.isel(time=t).values.reshape(-1) if ntimes > 1 else update.values.reshape(-1)
            if index_vals is None:
                index_vals = [x for x in np.unique(current_arr) if x > 0 and x != nan_val]
            di = {'current':current_arr,
                  'update':update_arr,}
            return index_vals, di

        # find mappings
        logging.info(f"{datetime.now()} Finding label maps: {ntimes} iterations")
        all_results, all_index_vals = [], [] # store results of each time step
        for t in range(ntimes):
            slices[0] = t if ntimes > 1 else slice(None)
            index_vals, di = index_data(t, index_vals=None)
            di = {k: {**di, 
                      'index':k,
                      } for k in index_vals}
            results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(joblib.delayed(self.find_labels)(di[k], nan_val) for k in index_vals) # process
            all_index_vals.extend(index_vals) # collect mappings
            all_results.extend(results)

        # clean mappings
        all_index_vals, all_results = self.proc_optimized(all_index_vals, all_results)
        logging.info(f"{datetime.now()} Found {len(all_index_vals)} label maps")
        return all_index_vals, all_results


    def update_labels_parallel(self, current, all_index_vals, all_results, nan_val=1e10, n_jobs=-1, n_k=500):
        current = current.fillna(nan_val).astype(int)
        new_labels = np.zeros(current.shape, dtype=int)
        ntimes = len(current.time) if 'time' in current.dims else 1
        current_tshape = current.isel(time=0).shape if ntimes > 1 else current.shape
        slices = [slice(None)] * len(current.dims)

        def index_data(t, index_vals=None):
            current_arr = current.isel(time=t).values.reshape(-1) if ntimes > 1 else current.values.reshape(-1)
            if index_vals is None:
                index_vals = [x for x in np.unique(current_arr) if x > 0 and x != nan_val]
            di = {'current':current_arr,}
            return index_vals, di

        # clean mappings
        all_index_vals, all_results = self.proc_optimized(all_index_vals, all_results)

        # apply mappings
        logging.info(f"{datetime.now()} Applying all label maps: {ntimes} iterations")
        logging.info(f"{datetime.now()} {len(all_index_vals)} values, batch size = {n_k}")
        durations = []
        for t in range(ntimes):
            if t % 10 == 0:
                logging.info(f"{datetime.now()} ({t}/{ntimes})...")
            start_time = time.time()
            slices[0] = t if ntimes > 1 else slice(None)
            all_index_vals, di = index_data(t, index_vals=all_index_vals)
            di = {k: {**di, 
                      'index':k,
                      'result':all_results[i],
                      } for i,k in enumerate(all_index_vals)}
            for batch in range(0, len(all_index_vals), n_k):
                batch_index_vals = all_index_vals[batch:batch+n_k]
                results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(joblib.delayed(self.update_labels)(di[k]) for k in batch_index_vals)
                for label_map in results:
                    new_labels[tuple(slices)] += label_map.reshape(current_tshape) # apply all mappings
            durations.append(time.time() - start_time)
            if t % 10 == 0:
                logging.info(f"{datetime.now()} Avg duration: {sum(durations)/len(durations):.4f} seconds")

        da = xr.DataArray(data=new_labels, dims=current.dims, coords=current.coords)
        da = da.where(da != nan_val)
        return da.where(da>0)
    
