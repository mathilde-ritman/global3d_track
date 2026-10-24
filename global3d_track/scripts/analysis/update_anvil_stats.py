'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2026

Re-run GRLStats for every DCC that already has a saved statistics file, so that
files computed before the anvil-masking fix (definitions.define_anvil chaining
two .where() calls, which let NaN "below the anvil base" levels get treated as
truthy and kept unmasked) get their anvil_*/ABH values corrected in place.
env_*/cloud_*/core_* fields are recomputed too as a side effect, but are not
affected by the fix and should come out numerically unchanged.

'''

import logging
import argparse
import glob
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
import intake
import easygems.healpix as egh

from ..src import utils
from . import get_stats_grl as gs


def load_completed_ids(path):
    ''' DCCs already recorded as successfully updated, from a previous run of this script. '''
    path = Path(path)
    if not path.exists():
        return set()
    return set(pd.read_csv(path)['dcc'].tolist())


def append_completed_id(path, dcc_id):
    ''' Record one more successfully-updated DCC, so a later restart of this script can skip it. '''
    path = Path(path)
    pd.DataFrame({'dcc': [dcc_id]}).to_csv(path, mode='a', header=not path.exists(), index=False)


def main(yaml_file):

    di = utils.tools.load_yaml(yaml_file)

    # directories
    version = Path(di['version_name'], di['region'])
    data_dir = Path(di['data_directory']) / version
    results_dir = Path(di['results_directory']) / di['folder']
    logging.info(f"{datetime.now()} Updating anvil statistics in {results_dir}")

    # target the DCCs listed in dcc_list_csv, rather than searching results_dir
    dcc_list = sorted(pd.read_csv(di['dcc_list_csv'])['dcc'].tolist())
    start_idx = di.get('start_index', 0)
    completed_csv = Path(di['completed_csv'])
    completed_ids = load_completed_ids(completed_csv)
    dcc_list = [d for d in dcc_list if d not in completed_ids]
    n = len(dcc_list)
    dcc_list = dcc_list[start_idx:]
    logging.info(f"{datetime.now()} Found {n} DCCs to update, starting at index {start_idx} ({len(completed_ids)} already recorded as done in {completed_csv})")

    def mark_completed(dcc_id):
        completed_ids.add(dcc_id)
        append_completed_id(completed_csv, dcc_id)

    # load data
    logging.info(f"{datetime.now()} Loading data...")
    cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
    ds = cat.ICON.ngc4008a(time="PT15M", zoom=9).to_dask().pipe(egh.attach_coords)

    # load tracking results
    logging.info(f"{datetime.now()} Loading tracking results...")
    mask_paths = glob.glob(str(data_dir / "*/*system_tracks_linked.nc"))
    mask = xr.open_mfdataset(mask_paths)
    dates = ('20210701', '20210702', '20210703', '20210704')
    meta_di = utils.tools.load_linked_records(data_dir, dates)

    N = di.get('max_workers', 0)

    if N == 0:
        logging.info(f"{datetime.now()} Starting updates (iterating)...")
        telapsed = []
        for i, dcc_id in enumerate(dcc_list):
            logging.info(f"{datetime.now()} Updating DCC {dcc_id} ({i}/{len(dcc_list)})")
            start = time.perf_counter()
            gs.get_dcc_properties(dcc_id, di, ds, mask, meta_di, results_dir)
            telapsed.append(time.perf_counter() - start)
            mark_completed(dcc_id)
            logging.info(f"{datetime.now()} Updated DCC {dcc_id} | Time taken: {telapsed[-1]:.0f}s | Mean time: {np.mean(telapsed):.0f}s")

    else:
        max_tasks_per_child = di.get('max_tasks_per_child', 20)
        logging.info(f"{datetime.now()} Starting updates (multiprocessing with {N} workers, recycled every {max_tasks_per_child} DCCs)...")
        telapsed, unresolved = gs.run_resilient_pool(dcc_list, di, ds, mask, meta_di, results_dir, N, max_tasks_per_child,
                                                      is_done=lambda d: d in completed_ids, on_success=mark_completed)

        if unresolved:
            logging.error(f"{datetime.now()} {len(unresolved)} of {len(dcc_list)} DCCs were never successfully updated")
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="path to configuration file", type=str)
    args = parser.parse_args()

    proc_start = datetime.now()
    logging.info(f"{proc_start} Commencing anvil-statistics update")
    logging.info(f"Configuration file: {args.yaml}")

    main(args.yaml)

    logging.info(f"{datetime.now()} Finished, time elapsed: {datetime.now() - proc_start}")
