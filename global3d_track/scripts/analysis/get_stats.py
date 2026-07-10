'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import logging
import argparse
import glob
import xarray as xr
import pandas as pd
import os
import time
import joblib
import dask
import tempfile
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

Get statistics for each DCC

'''

def grab_dcc(mask, data_dir, sidx):
    # load metadata
    df = pd.read_csv(data_dir / "track_stats/dcc_statistics.csv", parse_dates=['initial_time'], index_col='dcc')
    start_time = df.loc[sidx, 'initial_time']
    duration = pd.Timedelta(hours=df.loc[sidx, 'lifetime']+1)
    end_time = start_time + duration
    lat = df.loc[sidx, 'initial_lat']
    lon = df.loc[sidx, 'initial_lon']
    # subset mask locally
    mask_local = mask.sel(time=slice(start_time, end_time), lat=slice(lat-10, lat+10), lon=slice(lon-10, lon+10))
    # subset to exact system
    mask_sidx = mask_local.where(mask_local.system == sidx)
    mask_sidx = mask_sidx.dropna('lat', how='all').dropna('lon', how='all').dropna('time', how='all')
    if mask_sidx.time.size == 0 or mask_sidx.lat.size == 0 or mask_sidx.lon.size == 0:
        logging.warning(f"System {sidx} has no valid mask points after subsetting, masking from OG")
        mask_sidx = mask.where(mask.system == sidx).dropna('lat', how='all').dropna('lon', how='all').dropna('time', how='all')
    return mask_sidx

def save_xarray(data, path, engine="h5netcdf"):
    ''' compress and save via atomic write '''
    NAN = -999.99
    encoding = {}
    for var in data.data_vars:
        encoding_params = dict(zlib=True, complevel=4, _FillValue=NAN)
        encoding[var] = encoding_params
    # atomic write: write to a temporary file and replace
    path = Path(path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=str(path.parent))
    os.close(tmp_fd)
    try:
        data.to_netcdf(tmp_path, encoding=encoding, engine=engine)
        os.replace(tmp_path, str(path))
    finally:
        # ensure tmp removed if something failed
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def process_system(masks, di, data_dir, save_dir, sidx):
    # start logging
    itr_start = time.time()
    logging.info(f"{datetime.now()} processing system {sidx}...")

    # load cloud mask
    mask = grab_dcc(masks, data_dir, sidx)
    
    # skip big systems
    dcc_size = (mask.system>0).sum(('lat','lon')).max(('time','level_full')).values
    logging.info(f"{datetime.now()} System {sidx} has maximum size {dcc_size} points")

    if dcc_size > di.get('max_system_size', np.inf):
        logging.warning(f"System {sidx} is too large, skipping for now")
        return # exit
    
    # load data
    variables = ['cli', 'clw', 'dzghalf', 'hus', 'pfull', 'pr', 'qg', 'qr', 'qs', 'rlut', 'ta', 'ts', 'ua', 'va', 'wa_phy']
    data = utils.data_tools.grab_system_data(mask, variables)
    log_memory(f"{sidx}: after loading data")

    # calculate statistics
    to_get = di['results'].keys()
    results = xr.Dataset()
    for r in to_get:
        r_di = di['results'][r]
        func, params = r_di['function'], r_di.get('parameters', None)
        params = params if params is not None else {}
        results.update(getattr(statistics, func)().get_everything(mask, data, **params))
    log_memory(f"{sidx}: after computing statistics")

    # save results
    logging.info(f"{datetime.now()} saving results for system {sidx}...")
    fpath = save_dir / f"dcc_{sidx}.nc"
    if fpath.exists():
        os.remove(fpath)
    save_xarray(results, fpath)
    logging.info(f"{datetime.now()} saved results for system {sidx}")

    # final logging
    took = time.time() - itr_start
    logging.info(f"{datetime.now()} Done for system {sidx}, took {took} seconds. Saved to {fpath}")


def process_all(yaml_file):
    # input
    logging.info(f"{datetime.now()} commencing...")
    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']

    # directories
    version = Path(di['version_name'], di['region'])
    data_dir = Path(di['data_directory']) / version
    result_dir = Path(di['results_directory']) / version
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"{datetime.now()} using {result_dir} for results")

    # only process those that are valid
    df = pd.read_csv(data_dir / "track_stats/dcc_statistics.csv", index_col='dcc')
    # valid_systems = df.index[df["valid"]]
    valid_systems = df.index[df["valid_but_complex_convection"]]

    if not isinstance(valid_systems, list):
        valid_systems = valid_systems.tolist()
    logging.info(f"{datetime.now()} found {len(valid_systems)} valid systems to process.")

    # skip those that have already been processed
    remaining_systems = valid_systems.copy()
    count = 0
    for sidx in valid_systems:
        result_path = result_dir / f"dcc_{sidx}.nc"
        if result_path.exists() and not overwrite:
            remaining_systems.remove(sidx)
            count += 1
    logging.info(f"{datetime.now()} {count} systems already processed, {len(remaining_systems)} remaining")

    # load mask datasets
    mask_paths = glob.glob(str(data_dir / "*/*system_tracks_linked.nc"))
    masks = xr.open_mfdataset(mask_paths)

    # process in batches
    batch_size = di.get('batch_system', 1)
    n_batches = len(remaining_systems) // batch_size + 1
    logging.info(f"{datetime.now()} processing {len(remaining_systems)} systems in {n_batches} batches of {batch_size} systems each")

    # choose where to start from
    i0 = di.get('start_idx', 0)
    if i0 > len(remaining_systems):
        logging.warning(f"{datetime.now()} starting index {i0} is greater than number of systems ({len(remaining_systems)}), starting from 0 instead")
        i0 = 0
    logging.info(f"{datetime.now()} starting at index {i0}")

    durations = []
    for i in range(i0, len(remaining_systems), batch_size):
        itr_start = time.time()
        systems_in_batch = remaining_systems[i:i + batch_size]

        for sidx in systems_in_batch:
            # check not already processed (in case of restart)
            if (result_dir / f"dcc_{sidx}.nc").exists() and not overwrite:
                continue
            try:
                process_system(masks, di, data_dir, result_dir, sidx)
            except Exception as e:
                logging.warning(f"{datetime.now()} Error processing system {sidx}: {e}")

        # end batch
        durations.append(time.time() - itr_start)
        logging.info(f"{datetime.now()} average batch duration ({batch_size} DCCs): {sum(durations) / len(durations):.4f} seconds")

    logging.info(f"{datetime.now()} All done. Congratulations.")


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



