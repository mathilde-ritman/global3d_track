'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import yaml
import pickle
import logging
import warnings
import argparse
import glob
import xarray as xr
import pandas as pd
import os
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from ..src import utils, methods, statistics
Checkpoint = utils.checkpoint.Checkpoint


'''

General group-wise results processor. This version saves each cloud track seperately.

'''


def process_group(di, mask, data):
    # unstack
    mask = mask.unstack([x for x in mask.dims][0])
    mask = mask.sortby('lat').sortby('lon')
    # get statistics
    to_get = di['results'].keys()
    all_results = {}
    for r in to_get:
        r_di = di['results'][r]
        func, params = r_di['function'], r_di['parameters']
        result = getattr(statistics, func)().get_everything(mask, data, **params).compute()
        all_results[r] = result
    return all_results

def process_file(di, file, version):

    logging.info(f"{datetime.now()} processing file {file}")
    result_dir = Path(di['results_directory']) / version
    check_dir = Path(di['checkpoint_directory']) / version
    checkpoint = Checkpoint(check_dir, overwrite=(di['overwrite'] and di['restart_checkpoints']))
    itr_chunk = di['batch_size']
    stats_to_get = list(di['results'].keys())

    # - load mask data
    mask_data = xr.open_mfdataset(file)
    mask_data = mask_data.where(mask_data>0)

    # - done already ?!
    if all([str(result_dir / stat) in glob.glob(f"{result_dir}/*") for stat in stats_to_get]) and not di['overwrite']:
        logging.info(f"{datetime.now()} results exist at {result_dir}")
        return

    # - iterate times
    durations = []
    for t_idx in range(0, mask_data.time.size, itr_chunk):
        itr_start = time.time()

        # grab data
        mask_i = mask_data.isel(time=slice(t_idx, t_idx + itr_chunk))
        current_time = mask_i.time[0]
        next_time = mask_i.time[-1]

        # exists already ?!
        result_times = f"{current_time.dt.strftime('%Y%m%dT%H%M').item()}_{next_time.dt.strftime('%Y%m%dT%H%M').item()}"
        if glob.glob(f"{stats_to_get[0]}/{result_times}*") and not di['restart_checkpoints']:
            logging.info(f"{datetime.now()} skipping times {result_times}")
            continue
        
        # load 
        logging.info(f"{datetime.now()} loading data for period {current_time.values} {next_time.values}...")
        variables = ['zg','ta','rlut','wa_phy','cli','clw','pr','qg','qr','qs','pfull']
        data_i = utils.data_tools.load_corresponding_data(mask_i, di['region'], variables)
        data_i = utils.data_tools.preprocess_for_tobac(data_i)

        # group mask data
        logging.info(f"{datetime.now()} grouping...")
        groups = mask_i.groupby(mask_i.system)
        group_keys = list(groups.groups.keys())

        # multiprocess group statistics
        logging.info(f"{datetime.now()} calculating statistics for each group...")
        results = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(process_group)(di, groups[k], data_i) for k in group_keys)

        # collect and save results for all groups, one file for each group object!
        for group_id, res in zip(group_keys, results):
            for stat, res_dataset in res.items():
                group_id = int(group_id)
                stat_dir = Path(checkpoint.checkpoint_dir) / stat

                # do preceding results exist for this group already?
                matching_files = sorted([stat_dir / f for f in os.listdir(stat_dir) if f.endswith(str(group_id))])
                if matching_files:
                    prev_file = matching_files[0] # there should only be one matching file, if any
                    prev_dataset = xr.open_mfdataset(prev_file)
                    
                    # combine with current result
                    res_dataset = xr.concat((prev_dataset, res_dataset), dim='time')
                    result_times = f"{res_dataset.time[0].dt.strftime('%Y%m%dT%H%M').item()}_{res_dataset.time[-1].dt.strftime('%Y%m%dT%H%M').item()}"
                    os.remove(prev_file)
                    prev_dataset.close()

                # ensure sorting if lat/lon dimensions exist
                if 'lat' in res_dataset.dims:
                    res_dataset = res_dataset.sortby('lat').sortby('lon')

                # checkpoint
                current_file =  f"{stat}/{result_times}_{group_id}"
                checkpoint.checkpoint_dataset(res_dataset, current_file)

        data_i.close()
        mask_i.close()

        durations.append(time.time() - itr_start)
        logging.info(f"{datetime.now()} average duration: {sum(durations) / len(durations):.4f} seconds")

    # - save final results
    logging.info(f"{datetime.now()} saving final results to {result_dir}...")
    for stat in stats_to_get:
        final_loc = result_dir / stat
        final_loc.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(f"{check_dir}/{stat}/*"):
            os.system(f"scp {f} {final_loc}")
    logging.info(f"{datetime.now()} done.")


def main(yaml_file, start_date, end_date):

    # - load yaml and set up

    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    version = f"{di['version']}/{di['region']}"
    data_dir = Path(di['data_directory']) / version
    result_dir = Path(di['results_directory']) / version
    stats_to_get = list(di['results'].keys())

    # - what are we doing?
    
    if not stats_to_get:
        logging.info(f"{datetime.now()} you haven't asked me to calculate anything..?")
        exit()

    # - done already?!
        
    def get_files(regex):
        if len(glob.glob(regex)) == 0:
            return []
        return utils.tools.sort_files([f for f in glob.glob(regex) if utils.tools.check_file_dates(f, start_date, end_date)])
    
    res_fname = stats_to_get[0]
    finput_regex = f"{data_dir}/*/*proc.nc" # all files in directory for the processed tracking output
    fres_regex = f"{result_dir}/*{res_fname}*"# stats results
    
    input_files = get_files(finput_regex)
    res_files = get_files(fres_regex)

    # - go
    if (overwrite or (len(res_files) < len(input_files))):
        task_start =  time.time()
        logging.info(f"{task_start} Commencing statistics calculation for {stats_to_get}")

        # compute
        for file in input_files:
            process_file(di, file, version)
        logging.info(f"{datetime.now()} All complete. Took {time.time() - task_start}")

    else:
        logging.info(f"{task_start} Statistics already complete at {result_dir}")




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
    logging.info(f"{proc_start} Commencing detection and segmentation")
    logging.info(f"Configuration file: {args.yaml}")
    logging.info(f"Start date: {start_date.isoformat()}")
    logging.info(f"End date: {end_date.isoformat()}")

    main(args.yaml, start_date, end_date)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")



