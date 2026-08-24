'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import yaml
import pickle
import logging
import warnings
import argparse
import xarray as xr
import os
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import shutil
import sys
from ..src import utils, methods
Checkpoint = utils.checkpoint.Checkpoint

'''

General tracking processor. Loads tobac objects from the designated 'feature data directory'. Saves results to the designated 'data directory'. 

Checkpointing is implemented following key steps. Data checkpointed are saved to the designated to 'checkpoint directory' / 'current date'. Each date processed will have its own checkpoint directory and summary file.

'''


def track_object(di, obj_name, start_date, end_date):

    # - run and checkpoint management

    overwrite, overwrite_tobac, restart_checkpoints = di['overwrite'], di.get('overwrite_tobac', di['overwrite']), di['restart_checkpoints']

    # directories
    datestr = f"{start_date.strftime('%Y%m%dT%H%M')}_{end_date.strftime('%Y%m%dT%H%M')}"
    tobac_dir = Path(di['data_directory'], utils.tools.version_name(di, start_date=start_date, use_tobac_version=True))
    data_dir = Path(di['data_directory'], utils.tools.version_name(di, datestr=datestr))
    if di['checkpoint_directory'] is not None:
        check_dir = Path(di['checkpoint_directory'], utils.tools.version_name(di, datestr=datestr))
        # check_dir = Path(di['checkpoint_directory'], utils.tools.version_name(di, start_date=start_date))
    else:
        check_dir = None
    
    # specifications
    obj_di = di['objects'][obj_name] # choices for variable being tracked
    name = obj_di['name'] # variable name
    tobac_config = utils.tools.load_yaml(obj_di['tobac_config']) # tobac parameters
    vdim = tobac_config.get('vdim', 'level_full') # vertical dimension to use for processing, if not specified in tobac config, default to level_full

    # checkpoints
    checkpoint = Checkpoint(check_dir, overwrite=restart_checkpoints) # checkpoints
    n = f'{name}_tracking/' # subfolder for checkpointing current object
    if check_dir is not None:
        (check_dir/n).mkdir(parents=True, exist_ok=True) # make subdirectory for current object
    inner_checkpoint = dict(checkpoint=(checkpoint if di.get('inner_checkpoint', False) else None), checkpoint_name=n, check_n=di.get('topography_check_n', None))

    # output paths
    final_tracks_path = Path(data_dir) / f"{name}_tracks.nc"
    tobac_table_path = Path(tobac_dir) / f'{name}/tracked_features.h5' # record of tracked cells (this gets created after the tobac tracking completes)
    table_path = Path(data_dir) / f"{name}_tracks.h5" # file to which the subsequent tracking results should be added

    # - what tracking are we doing?

    skip_contiguity = skip_erode = True
    if obj_di['methods'].get('erode', 0) > 0:
        skip_contiguity = True # this is wrapped up in erode, so don't do it twice
        skip_erode = False
    elif obj_di['methods'].get('connect', False):
        skip_contiguity = False # perform some connected object tracking please
    elif not obj_di['methods'].get('tobac', False):
        warnings.warn('Tracking without using tobac first has not been implemented.')

    # - done already?!
        
    if final_tracks_path.exists() and not overwrite:
        logging.info(f"{datetime.now()} loaded {name} tracks from {final_tracks_path}")
        track_mask = xr.open_dataset(final_tracks_path)
        return track_mask
    
    if check_dir is not None and (check_dir / f'{n}final_tracks.nc').exists() and not (overwrite or restart_checkpoints):
        logging.info(f"{datetime.now()} loaded {name} tracks from {check_dir / n}final_tracks.nc...")
        track_mask = xr.open_dataset(check_dir / f'{n}final_tracks.nc')
        return track_mask

    # - details

    region = di['region']
    PBC_flag = None
    if region == 'tropics':
        PBC_flag = "hdim_2"
    if region == 'global':
        PBC_flag = "hdim_2"
    modify_parameters = dict(savedir=tobac_dir, PBC_flag=PBC_flag)

    # - processing

    if tobac_table_path.exists() and not overwrite_tobac:
        tobac_tracks_path = Path(tobac_dir) / f"{name}/tracked_mask.nc"
        logging.info(f"{datetime.now()} loading {name} tobac tracks from {tobac_tracks_path}...")
        track_mask = xr.open_dataset(tobac_tracks_path).sel(time=slice(start_date, end_date)) # tobac tracks the full day, but the input trange may be smaller
    
    elif checkpoint.checkpoint_reached(f'{n}tobac_tracks'):
        track_mask = checkpoint.load_dataset(f'{n}tobac_tracks').sel(time=slice(start_date, end_date))

    else:
        # prep
        logging.info(f"{datetime.now()} tobac tracking...")
        tic = time.perf_counter()
        tobac_methods = {'detect':True, 'segment':True, 'track': True, 'save': True}
        utils.tools.collect_tobac_features(tobac_dir, name, remove=False) # collects all detection and segmentation results within the time range being tracked, and removes the individual files

        # track using tobac
        module = methods.tobac_wrapper.Track(None, None, tobac_config, overwrite_tracks=overwrite_tobac, track_params=modify_parameters)
        track_mask, _ = module.perform(**tobac_methods)
        checkpoint.checkpoint_dataset(track_mask, f'{n}tobac_tracks')

        # copy tobac tracking record to the main tracking output directory, so that the next steps are recorded
        shutil.copy2(tobac_table_path, table_path)

        # remove files
        Path(tobac_dir, name, 'features.h5').unlink(missing_ok=True)
        Path(tobac_dir, name, 'segmented_mask.nc').unlink(missing_ok=True)

        toc = time.perf_counter()
        logging.info(f"{datetime.now()} done with tobac tracking, time taken: {toc - tic:0.1f} seconds")

    # chunk input
    track_mask = track_mask.chunk(utils.tools.get_chunks(track_mask))

    if skip_erode:
        pass

    elif checkpoint.checkpoint_reached(f'{n}erode-resulting_tracks'):
        track_mask['tracks'] = checkpoint.load_dataarray(f'{n}erode-resulting_tracks')

    else:
        tic = time.perf_counter()
        flag = 0
        if checkpoint.checkpoint_reached(f'{n}erode-erode_track'):
            erode_track = checkpoint.load_dataarray(f'{n}erode-erode_track')

        elif checkpoint.checkpoint_reached(f'{n}erode-erode_mask'):
            erode_mask = checkpoint.load_dataarray(f'{n}erode-erode_mask')
            flag = 1

        else:
            # perform erosion of mask
            logging.info(f"{datetime.now()} eroding mask...")
            erody_by = obj_di['methods']['erode']
            erode_mask = methods.Erode(**inner_checkpoint).weighted_erode(track_mask.cell, value=erody_by, vdim=vdim)
            checkpoint.checkpoint_dataset(erode_mask, f'{n}erode-erode_mask')
            flag = 1

        if flag:
            # track eroded mask
            logging.info(f"{datetime.now()} tracking eroded mask...")
            erode_track = methods.misc.track_connected_components(erode_mask, PBC_flag=PBC_flag)
            checkpoint.checkpoint_dataset(erode_track, f'{n}erode-erode_track')
        
        # share result to main mask
        logging.info(f"{datetime.now()} share labels...")
        eroded_tracks = methods.ShareLabels(track_mask.cell, erode_track, 'cell', 'econtiguity').tobac_like(table_path)
        eroded_tracks = methods.misc.force_consecutive_labels(eroded_tracks, table_path, current_col='econtiguity', update_col='tracks', new_tobac_table=False)
        track_mask['tracks'] = eroded_tracks

        # checkpoint.checkpoint_dataset(eroded_tracks, f'{n}erode-resulting_tracks')
        toc = time.perf_counter()
        logging.info(f"{datetime.now()} done with erosion-contiguity tracking, time taken: {toc - tic:0.1f} seconds")

    if skip_contiguity:
        pass

    elif checkpoint.checkpoint_reached(f'{n}connect-resulting_tracks'):
        track_mask['tracks'] = checkpoint.load_dataarray(f'{n}connect-resulting_tracks')

    else:
        tic = time.perf_counter()
        if checkpoint.checkpoint_reached(f'{n}connect-connect_track'):
            connect_track = checkpoint.load_dataarray(f'{n}connect-connect_track')
        else:
            # track using contiguity
            logging.info(f"{datetime.now()} contiguity tracking...")
            connect_track = methods.misc.track_connected_components(track_mask.feature)
            checkpoint.checkpoint_dataset(connect_track, f'{n}connect-connect_track')
        
        # share result to main mask
        logging.info(f"{datetime.now()} share labels...")
        connected_tracks = methods.ShareLabels(track_mask.cell, connect_track, 'cell', 'contiguity').tobac_like(table_path)
        connected_tracks = methods.misc.force_consecutive_labels(connected_tracks, table_path, current_col='contiguity', update_col='tracks', new_tobac_table=False)
        track_mask['tracks'] = connected_tracks

        # checkpoint.checkpoint_dataset(connected_tracks, f'{n}connect-resulting_tracks')
        toc = time.perf_counter()
        logging.info(f"{datetime.now()} done with contiguity tracking, time taken: {toc - tic:0.1f} seconds")

    # final result
    logging.info(f"{datetime.now()} saving...")
    if not 'tracks' in track_mask.data_vars:
        track_mask['tracks'] = track_mask.cell
    track_mask = track_mask.drop_vars(['feature', 'cell'])
    utils.tools.save_xarray(track_mask, final_tracks_path)
    if check_dir is not None:
        shutil.copy2(final_tracks_path, check_dir / f'{n}final_tracks.nc') # copy result to checkpoint directory

    logging.info(f"{datetime.now()} Saved {name} result to {final_tracks_path}.")

    return track_mask


def perform(yaml_file, start_date, end_date):

    # - load yaml and set up

    di = utils.tools.load_yaml(yaml_file)
    overwrite, restart_checkpoints = di['overwrite'], di['restart_checkpoints']
    variables_to_track = di['objects'].keys()

    # directories
    datestr = f"{start_date.strftime('%Y%m%dT%H%M')}_{end_date.strftime('%Y%m%dT%H%M')}"
    version_name = utils.tools.version_name(di, datestr=datestr)
    data_dir = Path(di['data_directory'], version_name)
    if di['checkpoint_directory'] is not None:
        check_dir = Path(di['checkpoint_directory'], version_name)
        utils.tools.make_directories((data_dir, check_dir))
    else:
        check_dir = None
        utils.tools.make_directories((data_dir, ))

    # checkpoints
    checkpoint = Checkpoint(check_dir, overwrite=restart_checkpoints) # define class

    # output paths
    tracks_record_path = data_dir / f"system_label_maps.h5"
    final_tracks_path = data_dir / f"system_tracks.nc"

    # - done already?!
        
    if final_tracks_path.exists() and not overwrite:
        logging.info(f"{datetime.now()} system tracks exist already at {final_tracks_path}")
        sys.exit()

    #  - track each object as per yaml

    tracking_results = {}
    for variable in variables_to_track:
        task_start = datetime.now()
        logging.info(f"{task_start} procesing object {variable}")
        tracking_results[variable] = track_object(di, variable, start_date, end_date)['tracks']
        logging.info(f"{datetime.now()} done with {variable}. Took {datetime.now() - task_start}.")
        
        # result is a dataset with variables: tracks

    # - apply required overlap of one tracked variable with another (e.g., updrafts must overlap with condensate)
        
    for variable in variables_to_track:
        variable_di = di['objects'][variable] # specifications for the current variable
        must_overlap_with = variable_di.get('require_overlap_with', False)
        if not isinstance(must_overlap_with, str):
            continue
        if checkpoint.checkpoint_reached(f'{variable}_tracking/overlap_tracks'):
            tracking_results[variable] = checkpoint.load_dataset(f'{variable}_tracking/overlap_tracks')['overlap_tracks'] # replace dataarray with that containing only overlapping tracks
        else:
            # calculate
            logging.info(f"{datetime.now()} requiring overlap of {variable} with {must_overlap_with}...")
            datasets = (tracking_results[must_overlap_with], tracking_results[variable]) # parent, child
            overlap_tracks = methods.misc.wrap_map_blocks(methods.misc.child_that_overlaps_array, datasets)
            tracking_results[variable] = overlap_tracks # replace tracks dataarray with overlap tracks
            checkpoint.checkpoint_dataset(overlap_tracks.to_dataset(name='overlap_tracks'), f'{variable}_tracking/overlap_tracks')

    # - the main game happens here, multivaraite tracking. This is where the individually tracked variables are used to define one overall 'system'. E.g., if the order given is condensate->updrafts, the labels of each condensate object (cloud) are used to label each system, and we find all the updrafts that belong to that system, and merge the result
    
    # setup
    logging.info(f"{datetime.now()} multivariate tracking with order {di['multivariate']['link_order']}...")
    order = di['multivariate']['link_order'].split('->')
    variable_to_use_labels = order[0] # variable whose labels we want to share (e.g., condensate)
    variable_to_get_labels = order[1] # variable getting new labels from the other mask (e.g., updrafts)
    mask_to_use_labels = tracking_results[variable_to_use_labels]
    mask_to_get_labels = tracking_results[variable_to_get_labels]

    # derive the overall system
    logging.info(f"{datetime.now()} share labels from {variable_to_use_labels} to {variable_to_get_labels}...")
    result, extra_mappings = methods.ShareLabels(mask_to_get_labels, mask_to_use_labels, variable_to_get_labels, variable_to_use_labels).tobac_like(tracks_record_path, return_extra_mappings=True)
    result = methods.ShareLabels().apply_mapping(result.to_dataset(name=variable_to_use_labels), extra_mappings, variable_to_use_labels, f"{variable_to_use_labels}_new")
    overall_system = methods.misc.union_all([result, mask_to_use_labels])

    # save extra mappings
    extra_maps_path = tracks_record_path.with_name(tracks_record_path.name.replace('system_label_maps', 'system_extra_maps'))
    extra_mappings.to_hdf(extra_maps_path, 'table')

    # collect results into a new dataset
    final = xr.Dataset({'system': overall_system,})
    
    # add individual tracks if you said you wanted them kept
    if di['objects'][variable_to_use_labels].get('keep_result', False):
        name = di['objects'][variable_to_use_labels]['shortname']
        final[f'{name}_tracks'] = mask_to_use_labels
    if di['objects'][variable_to_get_labels].get('keep_result', False):
        name = di['objects'][variable_to_get_labels]['shortname']
        final[f'{name}_tracks'] = mask_to_get_labels

    # final result
    logging.info(f"{datetime.now()} saving...")
    utils.tools.save_xarray(final, final_tracks_path)

    # remove the individual masks
    for variable in variables_to_track:
        variable_name = di['objects'][variable]['name'] # choices for variable being tracked
        file_path = Path(data_dir) / f"{variable_name}_tracks.nc"
        file_path.unlink(missing_ok=True)

    logging.info(f"{datetime.now()} Saved result to {final_tracks_path}.")
        


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="path to configuration file", type=str)
    parser.add_argument("-s", help="date on which to start process", type=str)
    parser.add_argument("-e", help="date on which to end process", type=str)
    args = parser.parse_args()

    # parse dates
    start_date = datetime.strptime(args.s, "%Y-%m-%d-%H:%M:%S")
    end_date = datetime.strptime(args.e, "%Y-%m-%d-%H:%M:%S")

    # go
    proc_start = datetime.now()
    logging.info(f"{proc_start} Commencing tracking")
    logging.info(f"Configuration file: {args.yaml}")
    logging.info(f"Start date: {start_date.isoformat()}")
    logging.info(f"End date: {end_date.isoformat()}")

    perform(args.yaml, start_date, end_date)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")

