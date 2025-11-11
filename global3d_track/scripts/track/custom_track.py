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
from ..src import utils, methods
Checkpoint = utils.checkpoint.Checkpoint

'''

General tracking processor. Loads tobac objects from the designated 'feature data directory'. Saves results to the designated 'data directory'. 

Checkpointing is implemented following key steps. Data checkpointed are saved to the designated to 'checkpoint directory' / 'current date'. Each date processed will have its own checkpoint directory and summary file.

'''


def track_object(di, obj_name, start_date, end_date, version):

    NAN = -9

    # - run and checkpoint management

    overwrite, restart_checkpoints = di['overwrite'], di['restart_checkpoints']
    fdata_dir = Path(di['feature_data_directory']) / f"{di['region']}/{start_date.strftime('%Y%m%d')}"
    data_dir = Path(di['data_directory']) / version
    check_dir = Path(di['checkpoint_directory']) / version
    checkpoint = Checkpoint(check_dir)
    obj_di = di['objects'][obj_name]
    tobac_config = utils.tools.load_yaml(obj_di['tobac_config'])
    name = obj_di['name']
    final_save_dir = Path(data_dir / f"{start_date.strftime('%Y%m%dT%H%M')}_{end_date.strftime('%Y%m%dT%H%M')}_{name}_tracks.nc")

    # - what tracking are we doing?

    skip_contiguity = skip_erode = False
    if obj_di['methods'].get('erode', 0) > 0:
        skip_contiguity = True # we want to perform some eroded object tracking
    elif obj_di['methods'].get('connect', False):
        skip_erode = True # we want to perform some connected object tracking
    elif 'track' not in obj_di['methods']['tobac']:
        warnings.warn('you need to specify a tracking method for tobac at a minumum.')

    # - done already?!
        
    if final_save_dir.exists() and not overwrite:
        logging.info(f"{datetime.now()} loaded {name} tracks from {final_save_dir}")
        track_mask = xr.open_dataset(final_save_dir)
        return track_mask.where(track_mask > 0)
    
    d = f'{name}_tracking/' # subfolder for checkpointing current object
    if checkpoint.checkpoint_reached(f'{d}tracks') and not (overwrite and restart_checkpoints):
        track_mask = checkpoint.load_dataset(f'{d}tracks', nan_value=NAN)
        return track_mask.where(track_mask > 0)

    # - details

    region = di['region']
    PBC_flag = None
    if region == 'tropics':
        PBC_flag = "hdim_2"
    if region == 'global':
        PBC_flag = "hdim_2"
    modify_parameters = dict(savedir=fdata_dir, PBC_flag=PBC_flag,)
    slabs_checkpoint = checkpoint if di['share_labels']['checkpoint'] else None
    n_k = di['share_labels'].get('n_k', 250)

    # - processing
    
    if checkpoint.checkpoint_reached(f'{d}tobac_tracks'):
        track_mask = checkpoint.load_dataset(f'{d}tobac_tracks', nan_value=NAN)

    else:
        # prep
        logging.info(f"{datetime.now()} tobac tracking...")
        tobac_methods = {m: True for m in obj_di['methods']['tobac']} | {'save': True}
        utils.tools.collect_tobac_features(fdata_dir, name)

        # track using tobac
        module = methods.tobac_wrapper.Track(None, None, tobac_config, overwrite_tracks=overwrite, track_params=modify_parameters)
        track_mask, _ = module.perform(**tobac_methods)
        track_mask = track_mask.where(track_mask > 0)
        checkpoint.checkpoint_dataset(track_mask.fillna(NAN).astype(np.int32), f'{d}tobac_tracks')

    # chunk input
    track_mask = track_mask.chunk({"time": 1, "lat": 128, "lon": 128})

    if skip_erode:
        pass

    elif checkpoint.checkpoint_reached(f'{d}erode-resulting_tracks'):
        track_mask['tracks'] = checkpoint.load_dataarray(f'{d}erode-resulting_tracks', nan_value=NAN)

    else:
        if checkpoint.checkpoint_reached(f'{d}erode-erode_mask'):
            erode_mask = checkpoint.load_dataarray(f'{d}erode-erode_mask', nan_value=NAN)
        else:
            # perform erosion of mask
            logging.info(f"{datetime.now()} eroding mask...")
            erody_by = obj_di['methods']['erode']
            erode_mask = methods.Erode().weighted_erode(track_mask.cell.fillna(0), value=erody_by)
            checkpoint.checkpoint_dataset(erode_mask.fillna(NAN).astype(np.int32), f'{d}erode-erode_mask')

        if checkpoint.checkpoint_reached(f'{d}erode-erode_track'):
            erode_track = checkpoint.load_dataarray(f'{d}erode-erode_track', nan_value=NAN)
        else:
            # track eroded mask
            logging.info(f"{datetime.now()} tracking eroded mask...")
            erode_track = methods.misc.track_connected_components(erode_mask, PBC_flag=PBC_flag)
            erode_track = erode_track.where(erode_track > 0)
            checkpoint.checkpoint_dataset(erode_track.fillna(NAN).astype(np.int32), f'{d}erode-erode_track')
        
        # share result to main mask
        logging.info(f"{datetime.now()} share labels...")
        eroded_tracks = methods.ShareLabels().share_labels_parallel(track_mask.cell, erode_track, nan_val=1e10, checkpoint=slabs_checkpoint, checkpoint_name=d, n_k=n_k)
        track_mask['tracks'] = eroded_tracks

        checkpoint.checkpoint_dataset(eroded_tracks.fillna(NAN).astype(np.int32), f'{d}erode-resulting_tracks')

    if skip_contiguity:
        pass

    elif checkpoint.checkpoint_reached(f'{d}connect-resulting_tracks'):
        track_mask['tracks'] = checkpoint.load_dataarray(f'{d}connect-resulting_tracks', nan_value=NAN)

    else:
        if checkpoint.checkpoint_reached(f'{d}connect-connect_track'):
            connect_track = checkpoint.load_dataarray(f'{d}connect-connect_track', nan_value=NAN)
        else:
            # track using contiguity
            logging.info(f"{datetime.now()} contiguity tracking...")
            connect_track = methods.misc.track_connected_components(track_mask.feature)
            checkpoint.checkpoint_dataset(connect_track.fillna(NAN).astype(np.int32), f'{d}connect-connect_track')
        
        # share result to main mask
        logging.info(f"{datetime.now()} share labels...")

        connect_tracks = methods.ShareLabels().share_labels_parallel(track_mask.cell, connect_track, nan_val=1e10, checkpoint=slabs_checkpoint, checkpoint_name=d, n_k=n_k)
        connect_tracks = methods.misc.force_consecutive_labels(connect_tracks)
        track_mask['tracks'] = connect_tracks

        checkpoint.checkpoint_dataset(connect_track.fillna(NAN).astype(np.int32), f'{d}connect-resulting_tracks')

    # final result
    logging.info(f"{datetime.now()} saving...")
    checkpoint.checkpoint_dataset(track_mask.fillna(NAN).astype(np.int32), f'{d}tracks')
    final_loc = checkpoint.record[f'{d}tracks']
    result = os.system(f'scp {final_loc} {final_save_dir}')
    if result != 0:
        logging.error("SCP command failed.")

    logging.info(f"{datetime.now()} Saved {name} result to {final_save_dir}.")

    return track_mask


def main(yaml_file, start_date, end_date):

    NAN = -9

    # - load yaml and set up

    track_di = utils.tools.load_yaml(yaml_file)
    overwrite, restart_checkpoints = track_di['overwrite'], track_di['restart_checkpoints']
    version = f"{track_di['link']['name']}/{track_di['region']}/{start_date.strftime('%Y%m%d')}"
    data_dir = Path(track_di['data_directory']) / version
    check_dir = Path(track_di['checkpoint_directory']) / version
    utils.tools.make_directories((data_dir, check_dir))
    checkpoint = Checkpoint(check_dir, overwrite = (overwrite and restart_checkpoints))
    objects_to_track = track_di['objects'].keys()
    final_save_dir = data_dir / f"{start_date.strftime('%Y%m%dT%H%M')}_{end_date.strftime('%Y%m%dT%H%M')}_system_tracks.nc"

    # - done already?!
        
    if final_save_dir.exists() and not overwrite:
        logging.info(f"{datetime.now()} system tracks exist already at {final_save_dir}")
        exit()

    n = 'system/tracks' # path for checkpointing
    if checkpoint.checkpoint_reached(n) and not restart_checkpoints:
        loc = checkpoint.record[n]
        result = os.system(f'scp {loc} {final_save_dir}')
        logging.info(f"System tracks already exist at {loc}. Copied to {final_save_dir}. Exiting.")
        exit()

    #  - track each object as per yaml

    tracked_mask = {}
    for obj in objects_to_track:
        task_start = datetime.now()
        logging.info(f"{task_start} procesing object {obj}")
        tracked_mask[obj] = track_object(track_di, obj, start_date, end_date, version).astype(np.int32)
        logging.info(f"{datetime.now()} done with {obj}. Took {datetime.now() - task_start}.")

    # - apply any required mask overlap filtering
        
    for obj in objects_to_track:
        obj_di = track_di['objects'][obj]
        if isinstance(obj_di['require_overlap'], str):
            if checkpoint.checkpoint_reached(f'{obj}_tracking/tracks') and 'overlap_tracks' in checkpoint.load_dataset(f'{obj}_tracking/tracks', nan_value=NAN).data_vars:
                tracked_mask[obj] = checkpoint.load_dataset(f'{obj}_tracking/tracks', nan_value=NAN)
            else:
                # calculate
                logging.info(f"{datetime.now()} requiring overlap of {obj} with {obj_di['require_overlap']}...")
                tracked_mask[obj]['overlap_tracks'] = methods.misc.child_that_overlaps(tracked_mask[obj_di['require_overlap']].tracks, tracked_mask[obj].tracks)
                checkpoint.checkpoint_dataset(tracked_mask[obj].fillna(NAN).astype(np.int32), f'{obj}_tracking/tracks')

    # - link the object tracks into one big tracked system
        
    order = track_di['link']['order'].split('->')
    slabs_checkpoint = checkpoint if track_di['share_labels']['checkpoint'] else None
    n_k = track_di['share_labels'].get('n_k', 250)
    if len(order) == 2:
        # collect tracks 
        logging.info(f"{datetime.now()} collecting tracks for order {track_di['link']['order']}...")
        def get_da(m):
            return m.tracks if not 'overlap_tracks' in m.data_vars else m.overlap_tracks
        mask_give = tracked_mask[order[0]]
        mask_get = tracked_mask[order[1]]
        gc.collect()

        # now get the overall system
        logging.info(f"{datetime.now()} share labels...")
        result = methods.ShareLabels().share_labels_parallel(get_da(mask_get), get_da(mask_give), nan_val=1e10, checkpoint=slabs_checkpoint, checkpoint_name='final/', n_k=n_k)
        overall_system = methods.misc.union_all([result, get_da(mask_give)])

        # collect results
        mask_give_n = track_di['objects'][order[0]]['shortname']
        mask_get_n = track_di['objects'][order[1]]['shortname']
        final = xr.Dataset({'system': overall_system,
                            f'{mask_give_n}_tracks': mask_give.tracks,
                            f'{mask_get_n}_tracks': mask_get.tracks})
    else:
        raise NotImplementedError(f"share labels for case like {track_di['link']['order']} not implemented")

    # save result :-)
    logging.info(f"{datetime.now()} saving...")
    checkpoint.checkpoint_dataset(final.fillna(NAN).astype(np.int32), n)
    final_loc = checkpoint.record[n]
    final_save_dir = data_dir / f"{start_date.strftime('%Y%m%dT%H%M')}_{end_date.strftime('%Y%m%dT%H%M')}_system_tracks.nc"
    result = os.system(f'scp {final_loc} {final_save_dir}')
    if result != 0:
        logging.error("SCP command failed.")

    logging.info(f"{datetime.now()} Saved result to {final_save_dir}.")
        


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="path to configuration file", type=str)
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

    main(args.yaml, start_date, end_date)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")

