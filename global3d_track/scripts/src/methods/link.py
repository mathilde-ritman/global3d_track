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
    return tuple(out[~np.isnan(out)])

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
    match_vals = {k: (find_matches(before, cond, k), find_matches(after, cond, k)) for k in k_vals if not np.isnan(k)}
    # replace labels of next_mask with paired labels from first_mask
    next_mask[variable+'_update'] = next_mask[variable+'_shift'].copy()
    first_mask[variable+'_update'] = first_mask[variable].copy()
    for k, sys in match_vals.items():
        # each k is a feature label, shared (or non shared)
        b_all, a = sys
        text = 'keep'
        if a:
            a = np.nanmin(a) # grab label from tuple
            if b_all:
                # if feature exists in both first and next mask, replace with value from first
                b = np.nanmin(b_all)
                next_mask[variable+'_update'] = next_mask[variable+'_update'].where(next_mask[variable+'_shift'] != a, b)
                text, out = 'replace', b
            else:
                # otherwise, no change
                out = a
        if len(b_all) > 1:
            # if multiple features within the same group, replace with first feature value
            b = np.nanmin(b_all) # get first feature label
            first_mask[variable+'_update'] = first_mask[variable+'_update'].where(~first_mask[variable+'_update'].isin(b_all), b)
            text, out = 'replace', b
        elif b_all:
            # otherwise, no change
            out = b_all
        # verbose
        # print(text, k, sys, 'as ->', np.nanmin(out))
            
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
