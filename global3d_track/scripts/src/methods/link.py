'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import numpy as np
import logging
from datetime import datetime
import xarray as xr
import glob
import os
from .connect_contiguous import Connect
from ..utils import tools


def find_matches(target, cond, k):
    out = np.unique(target.where(cond == k).values)
    return out[~np.isnan(out)]

def collect_matches(k_vals, cond, before, after):
    # where aarr contains repeated labels, update rows with repeat labels in both arrays with the union of those rows
    barr = np.full((k_vals.size, 100), 0)
    aarr = np.full((k_vals.size, 100), 0)
    for i,k in enumerate(k_vals):
        bmatches = find_matches(before, cond, k)
        amatches = find_matches(after, cond, k)
        barr[i,:bmatches.size] = bmatches
        aarr[i,:amatches.size] = amatches
    return barr, aarr

def update_matches(barr, aarr):
    # where aarr contains repeated labels, update rows with repeat labels in both arrays with the union of those rows
    unique_vals, _, counts = np.unique(aarr, return_index=True, return_counts=True)
    repeated_vals = unique_vals[counts>1]
    repeated_vals = repeated_vals[repeated_vals!=0]
    for val in repeated_vals:
        repeat_idxs = np.where(aarr == val)[0]
        aunion = np.unique(aarr[repeat_idxs])
        bunion = np.unique(barr[repeat_idxs])
        aarr[repeat_idxs, :aunion.size-1] = aunion[aunion!=0]
        barr[repeat_idxs, :bunion.size-1] = bunion[bunion!=0]
    aarr = np.where(aarr>0, aarr, np.nan)
    barr = np.where(barr>0, barr, np.nan)
    return barr, aarr

def link_chunks(first_mask, next_mask, variable='system'):

    ''' Ensure contiguity of mask labels between adjacent time chunks. '''

    NAN = -9
    first_mask = first_mask.where(first_mask != NAN)
    next_mask = next_mask.where(next_mask != NAN)

    # 1 - shift all labels in next file to ensure no repition
    # push new labels up using addition
    max_mask1 = first_mask[variable].max().compute()
    min_mask2 = next_mask[variable].min().compute()
    shift = max_mask1 - min_mask2
    # print(f'{shift=}')
    next_mask[variable+'_shift'] = next_mask[variable].where(next_mask[variable] > 0) + shift

    # 2 - find and re-label overlapping features (no issues if the masks are not actually adjacent in time, delt with in multivariate_tobac.Connect)
    # grab adjacent times
    before = first_mask.isel(time=-1)[variable]
    after = next_mask.isel(time=0)[variable+'_shift']
    join = xr.concat([before, after], dim='time').to_dataset(name='old_'+variable)
    # find shared systems
    join['connected'] = (join['old_'+variable].dims, Connect(join['old_'+variable].values > 0).get_components())
    join['connected'] = join.connected.where(join.connected > 0)
    join['new_'+variable] = join.connected
    # create dict of label replacements
    cond = join.connected
    k_vals = np.unique(cond.values)
    bmatches, amatches = collect_matches(k_vals[~np.isnan(k_vals)], cond, before, after)
    bmatches, amatches = update_matches(bmatches, amatches)
    # replace labels of first_mask and next_mask with paired labels from first_mask
    next_mask[variable+'_update'] = next_mask[variable+'_shift'].copy()
    first_mask[variable+'_update'] = first_mask[variable].copy()
    for i, k in enumerate(k_vals):
        # each k is a feature label, shared (or non shared)
        b_all = bmatches[i, :][~np.isnan(bmatches[i, :])] # coincident labels from first mask
        a_all = amatches[i, :][~np.isnan(amatches[i, :])] # coincident labels from next mask
        b = np.nanmin(b_all) # value from first to use as replacement
        if np.any(a_all):
            if np.any(b_all):
                # if feature exists in both first and next mask, replace next with value from first
                next_mask[variable+'_update'] = next_mask[variable+'_update'].where(~next_mask[variable+'_shift'].isin(a_all), b)
            # otherwise, no change
        if b_all.size > 1:
            # if multiple features within the first mask, replace with chosen (minimum) value
            first_mask[variable+'_update'] = first_mask[variable+'_update'].where(~first_mask[variable+'_update'].isin(b_all), b)
        # otherwise, no change
            
    # 3 - collect and return updated mask
    for v in first_mask.data_vars:
        if '_update' in v:
            v_name = v.replace('_update','')
            # drop old
            first_mask_updated = first_mask.drop_vars(v_name)
            next_mask_updated = next_mask.drop_vars(v_name)
            # keep new
            first_mask_updated = first_mask_updated.rename({v:v_name})
            next_mask_updated = next_mask_updated.rename({v:v_name})
    for v in first_mask_updated.data_vars:
        if '_shift' in v:
            # drop all
            first_mask_updated = first_mask_updated.drop_vars(v)
    for v in next_mask_updated.data_vars:
        if '_shift' in v:
            next_mask_updated = next_mask_updated.drop_vars(v)

    return first_mask_updated, next_mask_updated


def link_files(files, vars_to_update, fname_suffix='_linked'):

    NAN = -9
    
    # 2 - link chuncks
    logging.info(f"{datetime.now()} Linking tracks across time")
    # init
    if not files:
        logging.info(f"{datetime.now()} no files passed, exiting")
        return
    if len(files) == 1:
        logging.info(f"{datetime.now()} only one file passed, copying to {files[0].replace('.nc',f'{fname_suffix}.nc')}")
        os.system(f"scp {files[0]} {files[0].replace('.nc',f'{fname_suffix}.nc')}")
        return
    current_file = files.pop(0)
    current_mask = xr.open_dataset(current_file)
    # loop
    files_remaining = len(files)
    while files_remaining:
        fresh_list = vars_to_update.copy()
        # load next mask
        next_file = files.pop(0)
        next_mask = xr.open_dataset(next_file)
        # update
        v_to_link = fresh_list.pop(0)
        logging.info(f"{datetime.now()} linking {current_file} for variable={v_to_link}...")
        previous_mask, current_mask = link_chunks(current_mask, next_mask, variable=v_to_link)
        while fresh_list:
            v_to_link = fresh_list.pop(0)
            logging.info(f"{datetime.now()} linking {current_file} for variable={v_to_link}...")
            previous_mask, current_mask = link_chunks(previous_mask, current_mask, variable=v_to_link)
        files_remaining = len(files)
        # save
        tools.compress_and_save(previous_mask.fillna(NAN).astype(np.int64), current_file.replace('.nc',f'{fname_suffix}.nc'))
        current_file = next_file
    tools.compress_and_save(current_mask.fillna(NAN).astype(np.int64), next_file.replace('.nc',f'{fname_suffix}.nc'))
