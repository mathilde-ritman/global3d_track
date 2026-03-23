'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import numpy as np
from scipy import ndimage as ndi
import datetime
import logging
from datetime import datetime
import joblib
import time


class Erode:
    '''
    Erodes the edges of input data array by a specified value in the chosen dimensions.
    
    '''

    def __init__(self, checkpoint=None, checkpoint_name=None, check_n=False):
        self.checkpoint=checkpoint
        self.checkpoint_name=checkpoint_name
        self.check_n = check_n # checkpoint every n time steps

    def erode(self, da, value, dims=('lat','lon')):
        da_bool = da > 0
        structure = np.ones((3, 3))  # Define the structure for erosion
        eroded = ndi.binary_erosion(da_bool, structure=structure, iterations=value)
        return da.where(eroded)
    
    def weighted_erode(self, da, value, dims=('lat','lon'), PBC_flag=None, parallel=True):
        ''' da: xarray.DataArray '''
        logging.info(f"{datetime.now()} weighted eroding with value {value}")
        da = da.where(da>0)
        if parallel:
            weighted_topog = self.compute_topography_parallel(da, normalise=True, n_jobs=-1)
        else:
            weighted_topog = self.compute_topography(da, normalise=True)
        return da.where(weighted_topog > value, 0)

    def compute_topography(self, labeled_array, normalise=True):
        """
        Compute the normalised topography of a 2D labeled array.
        """
        # init
        topography = np.zeros_like(labeled_array, dtype=float)
        # unique labels, excluding the background (assumed to be 0)
        unique_labels = np.unique(labeled_array)
        unique_labels = unique_labels[unique_labels != 0]
        # process each
        logging.info(f"{datetime.now()} computing topography for {len(unique_labels)} labels")
        durations = []

        for label_value in unique_labels:
            # logging.info(f"{datetime.now()} computing topography for label {label_value}")
            start_time = time.time()
            binary_mask = labeled_array == label_value
            # Compute the distance transform for the current region
            distance = ndi.distance_transform_edt(binary_mask) # distance transform
            max_distance = distance.max() # normalise result
            if normalise and max_distance > 0:
                topography[binary_mask] = distance[binary_mask] / max_distance
            else:
                topography[binary_mask] = distance[binary_mask]

            end_time = time.time()  # End timing the iteration
            duration = end_time - start_time  # Calculate the duration
            durations.append(duration)  # Append the duration to the list

            # Calculate and print the average duration
            average_duration = sum(durations) / len(durations)
        logging.info(f"{datetime.now()} Iteration average duration: {average_duration:.4f} seconds")

        return topography
    
    ## parallel version

    def compute_label_topography(self, label_value, arr, normalise):
        binary_mask = arr == label_value
        distance = ndi.distance_transform_edt(binary_mask)  # Compute distance transform
        max_distance = distance.max()
        if normalise and max_distance > 0:
            return label_value, distance / max_distance
        return label_value, distance

    def compute_topography_parallel(self, da, normalise=True, n_jobs=-1):
        logging.info(f"{datetime.now()} computing topography at each time and height level...")
        topography = np.zeros_like(da, dtype=float)

        # iterate times and levels
        ntimes = len(da.time) if 'time' in da.dims else 1    
        slices = [slice(None)] * len(da.dims)
        durations = []

        # check if checkpoint exists
        itr_times = range(ntimes)
        if self.checkpoint is not None:
            latest_checkpoint = self.checkpoint.get_last_checkpoint(r'topography/t(\d+)')
            if latest_checkpoint is not None:
                logging.info(f"{datetime.now()} loading topography from checkpoint {latest_checkpoint}")
                tidx = int(latest_checkpoint.split('topography/t')[-1])
                logging.info(f"{datetime.now()} resuming from time index {tidx}")
                topography = self.checkpoint.load_array(latest_checkpoint) # load previous
                itr_times = range(tidx+1, ntimes) # start from next time step

        for tidx in itr_times:
            start_time = time.time()
            slices[0] = tidx if ntimes > 1 else slice(None)
            da_t = da.isel(time=tidx) if ntimes > 1 else da
            for level in range(len(da_t.level_full)):
                slices[-3] = level
                arr = da_t.isel(level_full=level).values.astype(np.int16)
                unique_labels = np.unique(arr)
                unique_labels = unique_labels[unique_labels != 0]
                results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.compute_label_topography)(label, arr, normalise) for label in unique_labels)
                for label_value, label_topography in results:
                    topography[tuple(slices)] += label_topography

            # checkpoint
            if self.check_n and (tidx % self.check_n == 0) and (self.checkpoint is not None):
                self.checkpoint.checkpoint_array(topography, f'{self.checkpoint_name}topography/t{tidx}')
                # remove previous checkpoint to save space
                if tidx > 0:
                    prev_checkpoint = f'{self.checkpoint_name}topography/t{tidx-self.check_n}'
                    self.checkpoint.remove_old(prev_checkpoint)

            # Calculate and print the average duration
            end_time = time.time()  # End timing the iteration
            duration = end_time - start_time  # Calculate the duration
            durations.append(duration)  # Append the duration to the list
            average_duration = sum(durations) / len(durations)
            if self.check_n and (tidx % self.check_n == 0):
                logging.info(f"Average duration for one tstep: {average_duration:.4f} seconds")
        logging.info(f"Average duration for one tstep: {average_duration:.4f} seconds")
        
        return topography
    
    