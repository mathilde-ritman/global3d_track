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
from ..src import utils, methods, statistics
Checkpoint = utils.checkpoint.Checkpoint


'''

General group-wise results processor.

'''


def process_group(di, mask, data, group_id):
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

    result_dir = Path(di['results_directory']) / version
    check_dir = Path(di['checkpoint_directory']) / version
    checkpoint = Checkpoint(check_dir)
    itr_chunk = di['batch_size']
    stats_to_get = di['results'].keys()

    # - done already ?!
    fpath_base = f"{result_dir}/{pd.to_datetime(start_time).strftime('%Y%m%dT%H%M')}_{pd.to_datetime(next_time).strftime('%Y%m%dT%H%M')}"
    if all([fpath_base + f"_{stat}.nc" in glob.glob(f"{result_dir}/*") for stat in stats_to_get]) and not di['overwrite']:
        logging.info(f"{datetime.now()} results exist at {fpath_base}")
        return

    # - load mask data
    mask_data = xr.open_mfdataset(file)
    mask_data = mask_data.where(mask_data>0)
    start_time = mask_data.time[0].values

    # - iterate times
    durations = []
    for t_idx in range(0, mask_data.time.size, itr_chunk):
        itr_start = datetime.now()

        # grab data
        mask_i = mask_data.isel(time=slice(t_idx, t_idx + args.chunk))
        current_time = mask_i.time[0].values
        next_time = mask_i.time[-1].values

        # exists already ?!
        itr_fname = f"{pd.to_datetime(current_time).strftime('%Y%m%dT%H%M')}_{pd.to_datetime(next_time).strftime('%Y%m%dT%H%M')}_{stats_to_get[0]}.nc"
        if checkpoint.checkpoint_reached(itr_fname) and not di['overwrite']:
            logging.info(f"{datetime.now()} skipping times {itr_fname}")
            continue
        
        # load 
        logging.info(f"{datetime.now()} loading data for period {current_time} {next_time}...")
        data_i = utils.data_tools.load_corresponding_data(mask_i)

        # group mask data
        logging.info(f"{datetime.now()} grouping...")
        groups = mask_i.groupby(mask_i.system)
        group_keys = list(groups.groups.keys())

        # multiprocess group statistics
        logging.info(f"{datetime.now()} calculating statistics for each group...")
        results = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(process_group)(di, groups[k], data_i, k) for k in group_keys)

        # collect results for each statistic bucket
        combined_results = {stat: {} for stat in stats_to_get}
        for group_id, res in zip(group_keys, results):
            for stat, res_data in res.items():
                combined_results[stat] = {group_id: res_data}

        # finish and checkpoint
        for stat, grouped_results in combined_results.items():
            # make dataset
            result_xr = xr.concat(list(grouped_results.values()), dim='system').assign_coords({'system': list(grouped_results.keys())}).sortby('lat').sortby('lon')
            # save / checkpoint
            itr_fname = f"{pd.to_datetime(current_time).strftime('%Y%m%dT%H%M')}_{pd.to_datetime(next_time).strftime('%Y%m%dT%H%M')}_{stat}.nc"
            checkpoint.checkpoint_dataset(result_xr, itr_fname)

            # next
            result_xr.close()

        data_i.close()
        mask_i.close()

        durations.append(datetime.now() - itr_start)
        logging.info(f"{datetime.now()} average duration: {sum(durations) / len(durations):.4f} seconds")


    # - save datasets as one file

    mask_coords = mask_data.drop_dims('time').coords
    mask_data.close()

    for stat in stats_to_get:
        all_files = glob.glob(f"{check_dir}/*_{stat}.nc")
        if len(all_files) == 0:
            logging.info(f"{datetime.now()} no files found for {stat}")
            continue

        logging.info(f"{datetime.now()} combining files for {stat}")
        datasets = []
        for f in all_files:
            datasets.append(xr.Dataset(coords=mask_coords).merge(xr.open_mfdataset(f)))
        all_data = xr.concat(datasets, dim='time')

        logging.info(f"{datetime.now()} saving...")
        fpath = f"{fpath_base}_{stat}.nc"
        all_data.to_netcdf(fpath)
        logging.info(f"{datetime.now()} saved {stat} results to {fpath}")


def main(yaml_file, start_date, end_date):

    # - load yaml and set up

    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    version = f"{di['link']['name']}/{di['region']}"
    data_dir = Path(di['data_directory']) / version
    result_dir = Path(di['results_directory']) / version
    stats_to_get = di['results'].keys()

    # - what are we doing?
    
    if not stats_to_get:
        logging.info(f"{datetime.now()} you haven't asked me to calculate anything..?")
        exit()

    # - done already?!
        
    def get_files(regex):
        if len(glob.glob(regex)) == 0:
            return []
        return utils.tools.sort_files([f for f in glob.glob(regex) if utils.tools.check_file_dates(f, start_date, end_date)])
    
    res_fname_suffix = f'_{stats_to_get[0]}'
        
    finput_regex = f"{data_dir}/*/*proc.nc" # all files in directory for the raw tracking output
    fres_regex = f"{result_dir}/*/*{res_fname_suffix}.nc" # linked tracking results with anvil heights
    
    input_files = get_files(finput_regex)
    res_files = get_files(fres_regex)


    # if need linking do that
        
    if (overwrite or (len(res_files) < len(input_files))):
        task_start = datetime.now()
        logging.info(f"{task_start} Commencing statistics calculation fo {stats_to_get}")
        logging.info(f"{datetime.now()} All complete. Took {datetime.now() - task_start}")

        # compute
        for file in input_files:
            process_file(di, file, version)
    
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



