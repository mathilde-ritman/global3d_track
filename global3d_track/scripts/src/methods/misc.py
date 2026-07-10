'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import os
import pathlib
import logging
import dask
from .. import methods, utils

def track_connected_components(d, dims_to_skip=(), PBC_flag=None):
    # track using connected components
    arr = methods.connect_contiguous.Connect(d>0).get_components(dims_to_skip=dims_to_skip, PBC_flag=PBC_flag)
    labels = xr.DataArray(data=arr, dims=d.dims, coords=d.coords)
    return labels.fillna(0)

def child_that_overlaps(parent, child):
    child_overlap = child.where(np.logical_and(child>0, parent>0)) # child features coinciding with the parent
    child_overlap = child_overlap.astype(float)
    overlapping_features = np.unique(child_overlap.values)
    overlapping_features = overlapping_features[np.isnan(overlapping_features)]
    output = child.where(child.isin(overlapping_features)).fillna(0)
    return output

def child_that_overlaps_array(parent, child):
    # find positions where both > 0
    mask = (child > 0) & (parent > 0)
    # unique values of child where overlap occurs
    overlapping = np.unique(child[mask])
    overlapping = overlapping[~np.isnan(overlapping)]
    if overlapping.size == 0:
        # no overlapping features -> return zeros with same shape/dtype
        return np.zeros_like(child)
    # keep child values that are in overlapping, else 0
    return np.where(np.isin(child, overlapping), child, 0)

def wrap_map_blocks(func, datasets=(), params=(), chunks=None):
    dtype, dims, coords = datasets[0].dtype, datasets[0].dims, datasets[0].coords
    # chunk data  as dask array
    if chunks is None:
        chunks = {'time': 1} | {dim: -1 for dim in datasets[0].dims if dim != 'time'}
    datasets = list(datasets)
    for i, da in enumerate(datasets):
        if not isinstance(da.data, dask.array.Array):
            logging.info(f"{datetime.now()} Converting to dask array for mapping...")
            da = dask.array.from_array(da.data, chunks=chunks)
        else:
            logging.info(f"{datetime.now()} Already a dask array, proceeding with mapping...")
            da = da.chunk(chunks).data
        datasets[i] = da
    # apply mapping to dask chunks
    result = dask.array.map_blocks(func, *tuple(datasets), *params, dtype=dtype)
    return xr.DataArray(result, dims=dims, coords=coords).rename('data')

def union_all(datasets):
    # collect value at all points covered
    da_union = datasets[0]
    for i in range(1,len(datasets)):
        da_union = np.maximum(da_union, datasets[i])
    return da_union.fillna(0)

def force_consecutive_labels(da, table_path=None, current_col=None, update_col=None, new_tobac_table=True):
    logging.info(f'{datetime.now()} forcing labels to be consecutive...')
    values = da.values.ravel()
    _, consecutive = np.unique(values, return_inverse=1) # force consecutive numbers
    result = xr.DataArray(data=consecutive.reshape(da.shape), dims=da.dims, coords=da.coords)
    # record mapping, optional
    if table_path is not None:
        table_path = pathlib.Path(table_path)
        if os.path.exists(table_path):
            df = pd.read_hdf(table_path, 'table') # load feature table
        else:
            df = pd.DataFrame({current_col: np.unique(values)}) # create
        mapping = dict(zip(values, consecutive)) # as dict
        # save
        df[update_col] = df[current_col].map(mapping).fillna(0) # record as new column
        if new_tobac_table:
            outpath = table_path.with_name(f"{table_path.stem}_{update_col}{table_path.suffix}")
        else:
            outpath = table_path
        df.to_hdf(outpath, 'table')
    return result

class Link:

    def __init__(self):
        pass

    def link_files(self, di, files, fname_suffix, last_completed_file=None):
        logging.info(f"{datetime.now()} Linking tracks across time...")

        overwrite = di['post_processing'].get('overwrite_linking', False)

        if len(files) < 2:
            logging.warning(f"Not enough files to link, skipping.")
            return
        
        # load current
        current_file = files.pop(0)
        current_mask = xr.open_dataset(current_file)
        files_remaining = len(files)

        # if not last_completed_file is None:
        #     current_file = last_completed_file
        #     current_mask = xr.open_dataset(current_file)
        #     existing_records = list(pathlib.Path(current_file).glob(f"{pathlib.Path(current_file).stem}_linked_*_maps.h5"))
        #     if not existing_records:
        #         logging.warning(f"No existing mapping records found for {current_file}, exiting...")
        #         return
        #     # continue
        #     records = {}
        #     for rec in existing_records:
        #         lab = rec.stem.split('_linked_')[1].split('_maps')[0]
        #         if lab in di['post_processing']['link_variables']:
        #             records[lab] = pd.read_hdf(rec, 'table')
        #     files_remaining = len(files) + 1

        # variables and methods
        temp = di['post_processing']['link_variables']
        methods = {}
        for lab, method in temp.items():
            if lab not in current_mask.data_vars:
                lab = di['objects'][lab]['shortname'] + '_tracks'
                if lab in current_mask.data_vars:
                    methods[lab] = method # add using updated name
            else:
                methods[lab] = method
            if isinstance(method, float):
                methods[lab] = {'method':'erode', 'erode_by': method}

        # record keeping
        records = {lab: (None, None) for lab in methods.keys()} # of the mappings performed for each variable

        while files_remaining:
            # load next mask
            next_file = files.pop(0)
            next_mask = xr.open_dataset(next_file)
            # link
            for lab, method in methods.items():
                logging.info(f"{datetime.now()} linking {current_file} for variable: {lab}, using method:{method}...")
                params = {}
                if isinstance(method, dict):
                    params = {k:v for k,v in method.items() if k != 'method'}
                    method = method['method']
                current_record, next_record = records[lab]
                current_mask, next_mask, current_record, next_record = self.link_tracks(current_mask, next_mask, variable=lab, method=method, params=params, current_record=current_record)
                records[lab] = current_record, next_record
                logging.info(f"{datetime.now()} done for variable: {lab}")
            # save
            logging.info(f"{datetime.now()} saving...")
            fpath = pathlib.Path(current_file)
            data_path = fpath.with_name(f"{fpath.stem}{fname_suffix}{fpath.suffix}")
            if os.path.exists(data_path) and not overwrite:
                logging.warning(f"{datetime.now()} {data_path} already exists, skipping saving...")
            else:
                for lab in records.keys():
                    current_record, _ = records[lab]
                    record_path = fpath.with_name(f"{fpath.stem}_linked_{lab}_maps.h5")
                    current_record.to_hdf(record_path, 'table')
                utils.tools.save_xarray(current_mask, data_path)
                logging.info(f"{datetime.now()} saved to {data_path}")
            # iterate
            current_file = next_file
            current_mask = next_mask
            records = {lab: (records[lab][1], None) for lab in records.keys()}
            files_remaining = len(files)

        # save final
        logging.info(f"{datetime.now()} saving...")
        fpath = pathlib.Path(next_file)
        for lab in records.keys():
                current_record, _ = records[lab]
                record_path = fpath.with_name(f"{fpath.stem}_linked_{lab}_maps.h5")
                current_record.to_hdf(record_path, 'table')
        data_path = fpath.with_name(f"{fpath.stem}{fname_suffix}{fpath.suffix}")
        utils.tools.save_xarray(next_mask, data_path)
        logging.info(f"{datetime.now()} saved to {data_path}")

    def link_tracks(self, first, next, variable, method='connect', params={}, current_record=None):

        if not method in ['connect','erode']:
            logging.warning(f"method {method} not recognised, defaulting to 'connect'")
            method = 'connect'

        # shift second mask
        shift = first[variable].max()
        next[variable] = (next[variable] + shift).where(next[variable] > 0, 0) 

        # link using the adjacent times
        da_t1 = first[variable].isel(time=-1)
        da_t2 = next[variable].isel(time=0)
        joint = xr.concat([da_t1, da_t2], dim='time')

        # track on the adjacent times
        logging.info(f"{datetime.now()} tracking adjacent times...")
        tracked = track_connected_components(joint).fillna(0)

        # derive and apply mappings for first mask
        # 1. find mappings: (shared mask labels -> first mask labels), where the shared mask is the results of the tracking just performed on the two adjacent time steps
        ShareLabels = methods.ShareLabels(tracked.isel(time=0), da_t1, 'track_labels', variable)
        df1, df2 = self.find_mappings(ShareLabels)
        df1 = df1.rename(columns={variable: f'{variable}_new'})
        # df1[track_labels, variable_new]: (shared -> first)
        # df2[variable, variable_new]: (first -> shared -> first) collects many-to-one changes that result from the adjacent tracking

        # 2. find mappings: (next mask labels -> shared mask labels)
        ShareLabels = methods.ShareLabels(da_t2, tracked.isel(time=1), variable, 'track_labels')
        df3, _ = self.find_mappings(ShareLabels)
        df3 = df3.rename(columns={'track_labels': 'track_labels_new'})
        ShareLabels = methods.ShareLabels(tracked.isel(time=1), da_t2, 'track_labels', variable)
        df4, _ = self.find_mappings(ShareLabels)
        for s in np.unique(df4[variable]):
            df4.loc[df4[variable]==s, 'track_labels_new'] = df4.where(df4[variable]==s)['track_labels'].min()
        df4 = df4.drop(columns=[variable,])
        # df3[variable, track_labels_new]: (next -> shared)
        # df4[track_labels, track_labels_new]: (shared -> next -> shared) collects many-to-one changes that result from the adjacent tracking

        # 3. derive the mappings to actulaly use that capture all the above information
        df_first = df2.merge(df1, left_on=f'{variable}_new', right_on=f'{variable}_new', how='left')
        df_first = df_first.merge(df4, left_on='track_labels', right_on='track_labels', how='left')
        df_first['track_labels_new'] = df_first['track_labels_new'].fillna(0)
        df_first.loc[df_first['track_labels_new']==0, 'track_labels_new'] = df_first['track_labels']
        reverse_many_to_one = df1[f'{variable}_new'].map(df1.set_index('track_labels')[f'{variable}_new'])
        one_to_many_to_one = reverse_many_to_one.where(reverse_many_to_one>0, df1[f'{variable}_new'])
        df_first[f'{variable}_linked'] = df_first['track_labels_new'].map(one_to_many_to_one).fillna(0).astype(int)
        df_first = df_first[[variable, f'{variable}_linked']].drop_duplicates()
        df_first = self.expand_mappings(df_first, first[variable], variable, f'{variable}_linked').fillna(0) # expand to full datasets

        df_next = df3.merge(df1, left_on='track_labels_new', right_on='track_labels', how='left')
        df_next[f'{variable}_linked'] = df_next[f'{variable}_new'].fillna(0).astype(int)
        shift_value = df_first[f'{variable}_linked'].max() + 1 # value to shift by for new features
        is_unmapped = np.logical_and(df_next[f'{variable}_linked'] == 0, df_next[variable]>0)
        df_next.loc[is_unmapped, f'{variable}_linked'] = (df_next.loc[is_unmapped, variable] + shift_value).astype(int)
        df_next = df_next[[variable, f'{variable}_linked']].drop_duplicates()
        df_next = self.expand_mappings(df_next, next[variable], variable, f'{variable}_linked').fillna(0)
        
        # 4. apply the results
        # df_first = self.clean_mappings(df_first, variable, f'{variable}_linked')
        # df_next = self.clean_mappings(df_next, variable, f'{variable}_linked')
        first[variable] = methods.ShareLabels().apply_mapping(first, df_first, variable, f'{variable}_linked')
        next[variable] = methods.ShareLabels().apply_mapping(next, df_next, variable, f'{variable}_linked')
        
        # record keeping, update the record for 'first' if a past record of mappings has been provided
        if current_record is not None:
            # update record keeping with the past mappings that were applied to current mask
            df_first = df_first.rename(columns={variable: f'{variable}_secondary',
                                                f'{variable}_linked': f'{variable}_linked_secondary'})
            # concat them together, keeping all record columns
            df_first = pd.concat([current_record, df_first], axis=1)

        # return
        return first, next, df_first, df_next

    def find_mappings(self, ShareLabels):
        # prep records
        current_vals = np.unique(ShareLabels.current.values)
        df = pd.DataFrame({ShareLabels.current_col: current_vals[~np.isnan(current_vals)]})
        update_vals = np.unique(ShareLabels.update.values)
        inverse_df = pd.DataFrame({ShareLabels.update_col: update_vals[~np.isnan(update_vals)]})
        # derive mappings
        mappings, inverse_mappings = ShareLabels.find_labels_parallel()
        # collect and return
        df = ShareLabels.mappings_to_dataframe(df, mappings)
        inverse_df = ShareLabels.mappings_to_dataframe(inverse_df, inverse_mappings, ShareLabels.update_col, f"{ShareLabels.update_col}_new")
        return df, inverse_df
    
    def expand_mappings(self, df, current, current_col, update_col):
        # get all indexes
        current_vals = np.unique(current.values)
        current_vals = current_vals[~np.isnan(current_vals)]
        # for those indexes not in df, add them with mapping to themselves
        missing = set(current_vals) - set(df[current_col])
        if missing:
            missing_df = pd.DataFrame({current_col: list(missing), update_col: list(missing)})
            df = pd.concat([df, missing_df], ignore_index=True)
        return df

    def clean_mappings(self, df, current_col, update_col):
        # remove any mappings that are not one-to-one, replace the update value with the minimum of the update values for each current value
        counts = df.groupby(current_col)[update_col].nunique()
        non_one_to_one = counts[counts > 1].index
        df_cleaned = df[~df[current_col].isin(non_one_to_one)].copy()
        for val in non_one_to_one:
            min_update = df[df[current_col] == val][update_col].min()
            di = {current_col: val, update_col: min_update}
            df_cleaned = pd.concat([df_cleaned, pd.DataFrame(di, index=[0])], ignore_index=True)
        return df_cleaned