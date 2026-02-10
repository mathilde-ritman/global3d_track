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
import time
import joblib
import dask
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from ..src import utils, statistics
Checkpoint = utils.checkpoint.Checkpoint

import psutil
def log_memory(string=""):
    mem = psutil.Process().memory_info().rss / 1e9
    logging.info(f"Memory usage ({string}): {mem:.2f} GB")

'''

General system-wise results processor.

'''


def process_system(di, data_dir, save_dir, sidx, NAN=-999.99):
    itr_start = time.time()
    logging.info(f"{datetime.now()} processing system {sidx}...")
    # - load mask and data
    log_memory(f"{sidx}: before loading data")
    mask = utils.data_tools.grab_system(sidx, data_dir)
    mask = mask.where(mask>0)
    # - check system size, should not ever user pecified max area (n pixels) or 50% of domain
    ssize = mask.system.sum(('lat','lon')).max(('time','level_full')).values
    domain_size = 300*400 ####! hardcoded !####
    logging.warning(f"{datetime.now()} System {sidx} has maximum size {ssize} points, covering {ssize/domain_size*100}% of the domain")
    if ssize > di.get('max_system_size', np.inf):
        logging.warning(f"System {sidx} is too large, skipping for now")
        del mask
        gc.collect()
        return # exit the function
    if ssize > di.get('max_system_domain', 1) * domain_size:
        logging.warning(f"System {sidx} covers {ssize/domain_size*100}% of domain, skipping for now")
        del mask
        gc.collect()
        return # exit
    variables = ['cli', 'clw', 'dzghalf', 'hus', 'pfull', 'pr', 'qg', 'qr', 'qs', 'rlut', 'ta', 'ts', 'ua', 'va', 'wa_phy']
    data = utils.data_tools.grab_system_data(mask, variables)
    log_memory(f"{sidx}: after loading data")
    # - compute statistics
    to_get = di['results'].keys()
    second_wave = []
    all_results = xr.Dataset()
    for r in to_get:
        r_di = di['results'][r]
        func, params = r_di['function'], r_di['parameters']
        if not params:
            params = {}
        if r_di.get('use_stats', False):
            # save for later
            second_wave.append(r_di)
            continue
        # compute now
        result = getattr(statistics, func)().get_everything(mask, data, **params)
        all_results.update(result)
        log_memory(f"{sidx}: after computing first wave")
        del result
    del data
    del mask
    # gc.collect()
    log_memory(f"{sidx}: after cleaning")
    # - now compute those that need the first wave results
    for r_di in second_wave:
        func, params = r_di['function'], r_di['parameters']
        if not params:
            params = {}
        # check which buckets are needed, and load results if they haven't just been computed
        buckets_needed = r_di.get('buckets_needed', {})
        results_to_pass = all_results.copy()
        for bucket in buckets_needed.keys():
            details = buckets_needed[bucket]
            # check whether just computed
            just_computed = np.all([x in to_get for x in details['function_names']])
            if not just_computed:
                # load existing
                bucket_results_path = save_dir.parents[0] / f"{details['result_path']}/cloud_{sidx}.nc"
                if bucket_results_path.exists():
                    logging.info(f"Loading {bucket_results_path} from file")
                    results_to_pass.update(xr.open_dataset(bucket_results_path))
                else:
                    logging.warning(f"Required bucket {bucket_results_path} not found for system {sidx}.")
                    gc.collect()
                    # exit
                    return
        result = getattr(statistics, func)().get_everything(results_to_pass, **params)
        if result is None:
            gc.collect()
            # exit
            return
        all_results.update(result)
        log_memory(f"{sidx}: after computing second wave")
        del result 
        del results_to_pass
        # gc.collect()
        log_memory(f"{sidx}: after cleaning")
    # - save results
    fpath = save_dir / f"cloud_{sidx}.nc"
    if fpath.exists():
        os.remove(fpath)
    all_results.fillna(NAN).to_netcdf(fpath)
    gc.collect()
    log_memory(f"{sidx}: after saving statistics")
    took = time.time() - itr_start
    logging.info(f"{datetime.now()} Done for system {sidx}, took {took} seconds. Saved to {fpath}")


def process_all(yaml_file):

    logging.info(f"{datetime.now()} commencing...")
    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    version = f"{di['version']}/{di['region']}"
    data_dir = Path(di['data_directory']) / version
    result_dir = Path(di['results_directory']) / version
    NAN = -999.99

    # - only process those that avoid the domain boundary
    df = pd.read_csv(data_dir / "data_filtering_stats/system_hits_boundary.csv", index_col="system_id")
    valid_systems = df.index[~df["hits_boundary"]] 
    if not isinstance(valid_systems, list):
        valid_systems = valid_systems.tolist()
    logging.info(f"{datetime.now()} found {len(valid_systems)} valid systems to process.")

    # - directory to use
    result_dir = result_dir / f"system-wise/{di['system_wise_version']}"
    result_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"{datetime.now()} using {result_dir} for results")

    # - already processed?!
    remaining_systems = valid_systems.copy()
    count = 0
    for sidx in valid_systems:
        result_path = result_dir / f"cloud_{sidx}.nc"
        if result_path.exists() and not overwrite:
            logging.info(f"{datetime.now()} skipping system {sidx} as already processed")
            remaining_systems.remove(sidx)
            count += 1
    logging.info(f"{datetime.now()} {count} systems already processed, {len(remaining_systems)} remaining")

    # - process in batches
    batch_size = di.get('batch_system', 1)
    n_batches = len(remaining_systems) // batch_size + 1
    i0 = di.get('start_idx', 0)
    if i0 >= len(remaining_systems):
        logging.warning(f"{datetime.now()} starting index {i0} is greater than number of systems {len(remaining_systems)}, setting to 0")
        i0 = 0
    logging.info(f"{datetime.now()} processing {len(remaining_systems)} systems in {n_batches} batches of {batch_size} systems each")
    durations = []
    for i in range(i0, len(remaining_systems), batch_size):
        itr_start = time.time()
        systems_in_batch = remaining_systems[i:i + batch_size]

        # - mulitiprocess batch
        logging.info(f"{datetime.now()} multiprocessing {batch_size} systems in batch {i // batch_size + 1} / {n_batches}...")
        # futures = [dask.delayed(process_system)(di, data_dir, result_dir, sidx, NAN=NAN) for sidx in systems_in_batch]
        # dask.compute(*futures)
        joblib.Parallel(n_jobs=-1, backend="multiprocessing")(joblib.delayed(process_system)(di, data_dir, result_dir, sidx, NAN=NAN) for sidx in systems_in_batch)

        logging.info(f"{datetime.now()} multiprocessing complete")

        # - next
        durations.append(time.time() - itr_start)
        logging.info(f"{datetime.now()} average batch duration: {sum(durations) / len(durations):.4f} seconds")

    logging.info(f"{datetime.now()} All complete.")



if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="path to configuration file", type=str)
    args = parser.parse_args()

    # go
    proc_start = datetime.now()
    logging.info(f"{proc_start} Commencing calculations")
    logging.info(f"Configuration file: {args.yaml}")

    process_all(args.yaml)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")



