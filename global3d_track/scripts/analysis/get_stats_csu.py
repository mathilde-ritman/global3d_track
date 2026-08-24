'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import logging
import argparse
import xarray as xr
import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import intake
import easygems.healpix as egh

from ..src import utils, statistics

import psutil
def log_memory(string=""):
    mem = psutil.Process().memory_info().rss / 1e9
    logging.info(f"{datetime.now()} Memory usage ({string}): {mem:.2f} GB")

def est_memory():
    return psutil.Process().memory_info().rss / 1e9

'''

Get statistics for each DCC

'''


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


def get_storm_centre(meta_di, dcc_id):
    # find first updraft
    updrafts = meta_di['w'][meta_di['w'].dcc==dcc_id]
    first_times = updrafts.groupby('updraft')['time'].min()
    
    # there were multiple, take longest lived one?
    durations = updrafts.groupby('updraft')['time_cell'].max()
    durations[first_times == first_times.min()]
    core_id = durations[first_times == first_times.min()].idxmax()
        
    # get coordinates
    core_data = meta_di['w'][meta_di['w'].updraft == core_id]
    core_initial = core_data[core_data.time == core_data.time.min()].iloc[0]
    ctime = core_initial.time - timedelta(minutes=15)
    clon, clat = core_initial.longitude, core_initial.latitude
    
    return pd.Series(dict(lon=clon, lat=clat, time=ctime))


def get_dcc_properties(dcc_id, di, ds, mask, meta_di, savedir):

    memory_estimates = []

    # find DCC zone
    df = meta_di['df']
    times = df[df.dcc==dcc_id].time.values
    times = sorted([pd.to_datetime(t.isoformat()) for t in times])
    lonn = meta_di['cld'][meta_di['cld'].dcc==dcc_id].longitude.min()
    lonx = meta_di['cld'][meta_di['cld'].dcc==dcc_id].longitude.max()
    
    # subset mask
    mask = mask.sel(time=slice(times[0], times[-1]), lon=slice(lonn-2, lonx+2))
    mask = mask.where(mask.system == dcc_id).dropna('time', how='all').dropna('lat', how='all').dropna('lon', how='all')

    # subset data
    times_plus = [times[0] - timedelta(minutes=15)] + times
    ds = ds.sel(time=slice(times_plus[0], times_plus[-1]))
    bbox = (lonn-2, lonx+2, -15, 15)
    ds = utils.regrid.Regrid(bbox).perform(ds.drop_vars(["lat", "lon"], errors="ignore"), zoom=di['params']['zoom'], resolution=di['params']['resolution'])
    if 'qall' in ds.data_vars:
        ds['qfrozen'] = ds['qall'].where(ds.ta<273.15)
    else:
        ds = utils.data_tools.preprocess_for_tobac(ds)
    memory_estimates.append(est_memory())

    # match precision
    r = lambda x: ((x / di['params']['resolution']).round() * di['params']['resolution']).round(2)
    ds['lon'] = r(ds.lon)
    ds['lat'] = r(ds.lat)
    mask['lon'] = r(mask.lon)
    mask['lat'] = r(mask.lat)

    # get storm centre
    cpoint = get_storm_centre(meta_di, dcc_id)

    # calculate
    dxy, dz, zdim, pname = di['params']['dxy'], di['params']['dz'], di['params']['zdim'], di['params']['pname']
    radii = di['radii']
    get_these = di['derive_properties']
    stats = statistics.csu_fast.CSUStats(dxy, dz, zdim, pname)
    results = stats.get_everything(mask, ds, cpoint, radii, get_these).expand_dims({'system': [dcc_id, ]})
    memory_estimates.append(est_memory())

    save_path = savedir / f"dcc_{int(dcc_id)}.nc"
    save_xarray(results, save_path)
    memory_estimates.append(est_memory())

    logging.info(f"{datetime.now()} Peak memory used for DCC {dcc_id} was {np.max(memory_estimates):.2f} GB")

def init_worker(di, ds, mask, meta_di, results_dir):
    global _di, _ds, _mask, _meta_di, _results_dir
    _di = di
    _ds = ds
    _mask = mask
    _meta_di = meta_di
    _results_dir = results_dir
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(process)d] %(levelname)s: %(message)s",
        force=True,
    )

def process_one(dcc_id):
    start = time.perf_counter()
    get_dcc_properties(dcc_id, _di, _ds, _mask, _meta_di, _results_dir)
    end = time.perf_counter()
    return end - start

def main(yaml_file):

    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    start_date = datetime.strptime(di['start_date'], "%Y-%m-%d %H:%M:%S")
    # end_date = datetime.strptime(di['end_date'], "%Y-%m-%d %H:%M:%S")

    # directories
    data_dirs = [Path(x) for x in di['data_directories']]
    filtering_dir = Path(di['filtering_directory'])
    results_dir = Path(di['results_directory'], di['folder'])
    tbc_dir = Path(di['tobac_directory']) if 'tobac_directory' in di else None
    results_dir.mkdir(parents=True, exist_ok=True)

    # load list
    df = pd.read_csv(filtering_dir / "dcc_statistics.csv")
    valid_df = df[df['valid']]
    is_okay_to_use = np.logical_and(pd.to_datetime(valid_df.initial_time) > start_date, valid_df.duration_convection < 8)
    is_okay_to_use = np.logical_and(is_okay_to_use, valid_df.ncores < 10)
    is_okay_to_use = np.logical_and(is_okay_to_use, valid_df.n_pixels < 2000)
    valid_df = valid_df[is_okay_to_use]
    dcc_list = valid_df['dcc'].values

    # skip those complete, by finding a list of all existing file names
    is_complete = [] if overwrite else [int(f.stem.removeprefix("dcc_")) for f in results_dir.glob("dcc_*.nc")]
    dcc_list = [x for x in dcc_list if x not in is_complete]

    # apply starting index
    start_idx = di.get('start_index', 0)

    # report plan
    logging.info(f"{datetime.now()} Found {len(dcc_list)} DCCs to process, starting at index {start_idx} ({len(is_complete)} were already complete)")

    # load data
    logging.info(f"{datetime.now()} Loading data...")
    if di['model_version'] == '4008a':
        cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
        ds = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().pipe(egh.attach_coords)
    else:
        cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["UK"]
        ds = cat.icon_d3hp003aug(time="PT15M", zoom=10).to_dask().pipe(egh.attach_coords)

    # load tracking results
    logging.info(f"{datetime.now()} Loading tracking results...")
    tracks = di['tracks']
    data_paths = [list(x.glob(f"*{tracks}.nc"))[0] for x in data_dirs]
    mask = xr.open_mfdataset(data_paths)
    ### TEMP FIX ###
    mask = mask.drop_duplicates(dim='time', keep='first')
    ### TEMP FIX ###
    ds = ds.sel({di['params']['zdim']: mask[di['params']['zdim']]})

    if tracks == "system_tracks":
        meta_di = utils.tools.load_single_record(data_dirs[0], tbcdir=None)
    elif tracks == "system_tracks_linked":
        data_dir = data_dirs[0].parent
        dates = [x.name for x in data_dirs]
        meta_di = utils.tools.load_linked_records(data_dir, dates, tbcdir=tbc_dir)

    logging.info(f"{datetime.now()} Starting calculations (iterating)...")

    telapsed = []
    for i in range(start_idx, len(dcc_list)):
        dcc_id = dcc_list[i]
        logging.info(f"{datetime.now()} Processing DCC {dcc_id} ({i}/{len(dcc_list)})")
        start = time.perf_counter()
        get_dcc_properties(dcc_id, di, ds, mask, meta_di, results_dir)
        end = time.perf_counter()
        telapsed.append(end - start)
        logging.info(f"{datetime.now()} Saved results for DCC {dcc_id}| Time taken: {telapsed[-1]:.0f}s | Mean time: {np.mean(telapsed):.0f}s")

    N = di.get('max_workers', 1)

    logging.info(f"{datetime.now()} Starting calculations (multiprocessing with {N} workers)...")
    telapsed = []
    with ProcessPoolExecutor(max_workers=N, initializer=init_worker, initargs=(di, ds, mask, meta_di, results_dir),) as executor:
        futures = {
            executor.submit(process_one, dcc_id): dcc_id
            for dcc_id in dcc_list[start_idx:]
        }

        for future in as_completed(futures):
            dcc_id = futures[future]
            try:
                telapsed.append(future.result())
                logging.info(f"{datetime.now()} Saved results for DCC {dcc_id} | Time taken: {telapsed[-1]:.0f}s | Mean time: {np.mean(telapsed):.0f}s")
            except Exception as e:
                logging.error(f"{datetime.now()} Error processing DCC {dcc_id}: {e}")




if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="path to configuration file", type=str)
    args = parser.parse_args()

    # go
    proc_start = datetime.now()
    logging.info(f"{proc_start} Commencing calculations")
    logging.info(f"Configuration file: {args.yaml}")

    main(args.yaml)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")



