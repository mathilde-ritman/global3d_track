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

    def __init__(self):
        pass

    def erode(self, da, value, dims=('lat','lon')):
        da_bool = da > 0
        structure = np.ones((3, 3))  # Define the structure for erosion
        eroded = ndi.binary_erosion(da_bool, structure=structure, iterations=value)
        return da.where(eroded)
    
    def weighted_erode(self, da, value, dims=('lat','lon'), PBC_flag=None, parallel=True):
        ''' da: xarray.DataArray '''
        logging.info(f"{datetime.now()} weighted eroding with value {value}")
        if parallel:
            weighted_topog = self.compute_topography_parallel(da, normalise=True, n_jobs=-1)
        else:
            weighted_topog = self.compute_topography(da, normalise=True)
        return da.where(weighted_topog > value)
        # return weighted_topog

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
        for t in range(ntimes):
            slices[0] = t if ntimes > 1 else slice(None)
            da_t = da.isel(time=t) if ntimes > 1 else da
            for level in range(len(da_t.level_full)):
                slices[-3] = level
                arr = da_t.isel(level_full=level).values.astype(np.int16)
                unique_labels = np.unique(arr)
                unique_labels = unique_labels[unique_labels != 0]
                
                start_time = time.time()

                results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.compute_label_topography)(label, arr, normalise) for label in unique_labels)
                for label_value, label_topography in results:
                    topography[tuple(slices)] += label_topography

                end_time = time.time()  # End timing the iteration
                duration = end_time - start_time  # Calculate the duration
                durations.append(duration)  # Append the duration to the list

                # Calculate and print the average duration
                average_duration = sum(durations) / len(durations)
        logging.info(f"Average duration: {average_duration:.4f} seconds")

        return topography
    
    