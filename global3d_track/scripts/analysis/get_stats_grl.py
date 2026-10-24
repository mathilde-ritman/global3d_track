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
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import dask
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import intake
import easygems.healpix as egh
import psutil

from ..src import utils, statistics

def log_memory(string=""):
    mem = psutil.Process().memory_info().rss / 1e9
    logging.info(f"Memory usage ({string}): {mem:.2f} GB")

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


def dcc_longitude_extent(lons):
    lons = np.sort(np.mod(lons, 360.0))
    if lons.size <= 1:
        v = float(lons[0]) if lons.size else 0.0
        return v, v
    gaps = np.diff(lons)
    wrap_gap = 360.0 - lons[-1] + lons[0]
    all_gaps = np.append(gaps, wrap_gap)
    idx = np.argmax(all_gaps)
    if idx == len(lons) - 1:
        lonn, lonx = lons[0], lons[-1]
    else:
        lonn, lonx = lons[idx + 1], lons[idx] + 360.0
    span = lonx - lonn
    lonn = ((lonn + 180.0) % 360.0) - 180.0
    return lonn, lonn + span


def get_dcc_properties(dcc_id, di, ds, mask, meta_di, savedir):

    memory_estimates = []

    # find DCC zone
    df = meta_di['df']
    times = df[df.dcc==dcc_id].time.values
    times = sorted([pd.to_datetime(t.isoformat()) for t in times])
    lons = meta_di['cld'][meta_di['cld'].dcc==dcc_id].longitude.values
    lonn, lonx = dcc_longitude_extent(lons)
    if lonx > 180:
        raise ValueError(f"DCC {dcc_id}: footprint straddles the antimeridian (lonn={lonn:.2f}, lonx={lonx:.2f}) - skipping")

    # subset mask
    dx = 5
    mask = mask.sel(time=slice(times[0], times[-1]), lon=slice(lonn-dx, lonx+dx))
    mask = mask.where(mask.system == dcc_id).dropna('time', how='all').dropna('lat', how='all').dropna('lon', how='all')

    # subset data
    times_plus = [times[0] - timedelta(minutes=15)] + times
    bbox = (lonn-dx, lonx+dx, -15, 15)
    ds = ds.sel(time=slice(times_plus[0], times_plus[-1]))
    ds = utils.regrid.Regrid(bbox).perform(ds.drop_vars(["lat", "lon"], errors="ignore"), zoom=di['params']['zoom'], resolution=di['params']['resolution'])
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
    radius = di['radius']
    stats = statistics.grl_fast.GRLStats()
    results = stats.get_everything(mask, ds, cpoint, radius).expand_dims({'system': [dcc_id, ]})
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

def run_resilient_pool(dcc_list, di, ds, mask, meta_di, results_dir, max_workers, max_tasks_per_child,
                        max_restarts=2, is_done=None, on_success=None, max_failures_per_dcc=1):

    if is_done is None:
        is_done = lambda dcc_id: (results_dir / f"dcc_{int(dcc_id)}.nc").exists()

    remaining = list(dcc_list)
    telapsed = []
    restarts = 0
    workers = max_workers
    failure_counts = {}
    given_up = []

    while remaining:
        logging.info(f"{datetime.now()} Starting pool for {len(remaining)} DCCs (attempt {restarts + 1}, {workers} workers)")
        pool_broke = False

        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                  initargs=(di, ds, mask, meta_di, results_dir),
                                  max_tasks_per_child=max_tasks_per_child) as executor:
            futures = {executor.submit(process_one, dcc_id): dcc_id for dcc_id in remaining}

            for future in as_completed(futures):
                dcc_id = futures[future]
                try:
                    telapsed.append(future.result())
                    logging.info(f"{datetime.now()} Saved results for DCC {dcc_id} | Time taken: {telapsed[-1]:.0f}s | Mean time: {np.mean(telapsed):.0f}s")
                    if on_success is not None:
                        on_success(dcc_id)
                except BrokenProcessPool as e:
                    logging.error(f"{datetime.now()} Process pool broke (DCC {dcc_id} may or may not be the cause): {e}")
                    pool_broke = True
                    break
                except Exception as e:
                    failure_counts[dcc_id] = failure_counts.get(dcc_id, 0) + 1
                    logging.error(f"{datetime.now()} Error processing DCC {dcc_id}: {e}")

        remaining = [d for d in remaining if not is_done(d)]

        newly_given_up = [d for d in remaining if failure_counts.get(d, 0) >= max_failures_per_dcc]
        if newly_given_up:
            logging.error(f"{datetime.now()} Giving up on {len(newly_given_up)} DCC(s) after {max_failures_per_dcc} failed "
                           f"attempts each: {newly_given_up}")
            given_up.extend(newly_given_up)
            remaining = [d for d in remaining if d not in newly_given_up]

        if not pool_broke:
            if remaining:
                logging.warning(f"{datetime.now()} {len(remaining)} DCCs finished this pool without a saved output "
                                 f"(real per-DCC errors, will be retried): {remaining[:10]}{' ...' if len(remaining) > 10 else ''}")
            break
        
        restarts += 1
        if restarts > max_restarts:
            logging.error(f"{datetime.now()} Exceeded {max_restarts} pool restarts with {len(remaining)} DCCs still unprocessed - giving up")
            break

    return telapsed, remaining + given_up

def main(yaml_file):

    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    start_date = datetime.strptime(di['start_date'], "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(di['end_date'], "%Y-%m-%d %H:%M:%S")

    # directories
    version = Path(di['version_name'], di['region'])
    data_dir = Path(di['data_directory']) / version
    results_dir = Path(di['results_directory']) / di['folder']
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"{datetime.now()} using {results_dir} for results")

    # load list
    df = pd.read_csv(data_dir / "track_stats/dcc_statistics.csv", index_col='dcc')

    # apply requirments
    validity = di.get('validity', 'valid')
    max_lifetime = di.get('max_lifetime', 100)
    max_size = di.get('max_size', 1e5)
    valid_df = df[df[validity]] if validity != 'invalid' else df[~df.invalid]
    is_okay_to_use = np.logical_and(valid_df.lifetime < max_lifetime, valid_df.npixels < max_size)
    is_okay_to_use = np.logical_and(is_okay_to_use, pd.to_datetime(valid_df.initial_time) > start_date)
    dcc_list = valid_df[is_okay_to_use].index.values

    # skip those complete, by finding a list of all existing file names
    is_complete = [] if overwrite else [int(f.stem.removeprefix("dcc_")) for f in results_dir.glob("dcc_*.nc")]
    dcc_list = [x for x in dcc_list if x not in is_complete]

    # apply starting index
    start_idx = di.get('start_index', 0)
    dcc_list = dcc_list[start_idx:]

    # report plan
    logging.info(f"{datetime.now()} Found {len(dcc_list)} DCCs to process, starting at index {start_idx} ({len(is_complete)} were already complete)")

    # load data
    logging.info(f"{datetime.now()} Loading data...")
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    ds = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().pipe(egh.attach_coords)

    # load tracking results
    logging.info(f"{datetime.now()} Loading tracking results...")
    mask_paths = glob.glob(str(data_dir / "*/*system_tracks_linked.nc"))
    mask = xr.open_mfdataset(mask_paths)
    dates = ('20210701','20210702','20210703','20210704')
    meta_di = utils.tools.load_linked_records(data_dir, dates)

    N = di.get('max_workers', 0)

    if N == 0:
        logging.info(f"{datetime.now()} Starting calculations (iterating)...")
        telapsed = []
        for i in range(len(dcc_list)):
            dcc_id = dcc_list[i]
            logging.info(f"{datetime.now()} Processing DCC {dcc_id} ({i}/{len(dcc_list)})")
            start = time.perf_counter()
            try:
                get_dcc_properties(dcc_id, di, ds, mask, meta_di, results_dir)
            except Exception as e:
                logging.info(f"Failed DCC {dcc_id} with exception: {e}")
            end = time.perf_counter()
            telapsed.append(end - start)
            logging.info(f"{datetime.now()} Saved results for DCC {dcc_id}| Time taken: {telapsed[-1]:.0f}s | Mean time: {np.mean(telapsed):.0f}s")

    else:
        max_tasks_per_child = di.get('max_tasks_per_child', 20)
        logging.info(f"{datetime.now()} Starting calculations (multiprocessing with {N} workers, recycled every {max_tasks_per_child} DCCs)...")
        telapsed, unresolved = run_resilient_pool(dcc_list, di, ds, mask, meta_di, results_dir, N, max_tasks_per_child)
        if unresolved:
            logging.error(f"{datetime.now()} {len(unresolved)} of {len(dcc_list)} DCCs were never successfully processed")



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

