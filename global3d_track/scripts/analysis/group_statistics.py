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
import numpy as np
from pathlib import Path
from datetime import datetime
from ..src import utils, methods, statistics
Checkpoint = utils.checkpoint.Checkpoint


'''

General group-wise results processor.

'''


def process_group(di, mask, data):
    # unstack
    mask = mask.unstack([x for x in mask.dims][0])
    mask = mask.sortby('lat').sortby('lon')
    # shrink data to domain
    try:
        data_shrunken = data.sel(lat=mask.lat, lon=mask.lon)
    except:
        logging.warning(f"Could not shrink data to mask, using full data instead.")
        logging.info(f"{mask.lat=}")
        logging.info(f"{mask.lon=}")
        t_str = mask.time[0].dt.strftime('%Y%m%dT%H%M').item()
        mask.to_netcdf(f"/scratch/b/b382635/temp/problem_mask-{t_str}.nc")
        data.to_netcdf(f"/scratch/b/b382635/temp/problem_data-{t_str}.nc")
        logging.info(f"saved to /scratch/b/b382635/temp/")
        data_shrunken = data
    # data_shrunken = data
    # get statistics
    to_get = di['results'].keys()
    all_results = {}
    for r in to_get:
        r_di = di['results'][r]
        func, params = r_di['function'], r_di['parameters']
        if not params:
            params = {}
        result = getattr(statistics, func)().get_everything(mask, data_shrunken, **params)
        # return to original lat/lon extent
        result = xr.merge((result, data[['lat','lon']]))
        all_results[r] = result
    return all_results

def is_time_range_already_processed(current_start, current_end, check_dir, stats_to_get):
    ''' Check if the current time range is already contained in any existing files.'''
    stats_dir = check_dir / stats_to_get[0]
    
    if not stats_dir.exists():
        return False
    
    current_start = pd.to_datetime(current_start.values)
    current_end = pd.to_datetime(current_end.values)

    for f in stats_dir.glob("*.nc"):
        try:
            name = f.stem  # strip .nc
            parts = name.split("_")
            if len(parts) != 2:
                continue  # not in expected format
            existing_start = datetime.strptime(parts[0], "%Y%m%dT%H%M")
            existing_end = datetime.strptime(parts[1], "%Y%m%dT%H%M")
            if existing_start <= current_start and current_end <= existing_end:
                return True
        except Exception as e:
            logging.warning(f"Could not parse time from file {f.name}: {e}")
            continue

    return False

def process_file(di, file, version):

    logging.info(f"{datetime.now()} processing file {file}")
    result_dir = Path(di['results_directory']) / version
    check_dir = Path(di['checkpoint_directory']) / version
    checkpoint = Checkpoint(check_dir, overwrite=di['restart_checkpoints'])
    itr_chunk = di['batch_size']
    stats_to_get = list(di['results'].keys())
    NAN = -999.99

    # - done already ?!
    file_times = '_'.join(file.split('/')[-1].split('_')[:2])
    if all([f"{result_dir}/{stat}/{file_times}" in glob.glob(f"{result_dir}/*") for stat in stats_to_get]) and not di['overwrite']:
        logging.info(f"{datetime.now()} results exist at {result_dir}")
        return
    logging.info(f"{datetime.now()} processing file: {file}")

    # - load mask data
    mask_data = xr.open_mfdataset(file, chunks={'time': 1, 'level_full': 10, 'lat': 50, 'lon': 50})
    mask_data = mask_data.where(mask_data>0)
    mask_data['lat'] = mask_data.lat.round(2)
    mask_data['lon'] = mask_data.lon.round(2)

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
        # check_fname = f"{stats_to_get[0]}/{result_times}"
        if is_time_range_already_processed(current_time, next_time, check_dir, stats_to_get) and not di['restart_checkpoints']:
            logging.info(f"{datetime.now()} skipping times {result_times}")
            continue
        
        # load 
        logging.info(f"{datetime.now()} loading data for period {current_time.values} {next_time.values}...")
        variables = ['zg','dzghalf','ts','ta','rlut','wa_phy','cli','clw','pr','qg','qr','qs','hus','pfull', 'ua','va']
        data_i = utils.data_tools.load_corresponding_data(mask_i, di['region'], variables, preceeding_mins=0)
        data_i = utils.data_tools.preprocess_for_tobac(data_i)
        data_i['lat'] = data_i.lat.round(2)
        data_i['lon'] = data_i.lon.round(2)

        # group mask data
        logging.info(f"{datetime.now()} grouping...")
        groups = mask_i.groupby(mask_i.system)
        group_keys = list(groups.groups.keys())

        # multiprocess group statistics
        logging.info(f"{datetime.now()} calculating statistics for each group...")
        # results = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(joblib.delayed(process_group)(di, groups[k], data_i) for k in group_keys)
        futures = [dask.delayed(process_group)(di, groups[k], data_i) for k in group_keys]
        results = dask.compute(*futures)

        # close
        del data_i, mask_i

        # collect results for each statistic bucket
        results_di = {stat: {} for stat in stats_to_get}
        for group_id, res in zip(group_keys, results):
            for stat, res_data in res.items():
                results_di[stat][int(group_id)] = res_data # append group result

        # finish and checkpoint, one file for the whole domain in the current time segment
        for stat, grouped_results in results_di.items():
            # make dataset
            result_xr = xr.concat(list(grouped_results.values()), dim='system').assign_coords({'system': list(grouped_results.keys())}).sortby('lat').sortby('lon')
            # drop placeholder coords
            result_xr = result_xr.sel(core=result_xr.core.values[~np.equal(result_xr.core.values, None)])
            # save / checkpoint
            file_name = f"{stat}/{result_times}"
            checkpoint.checkpoint_dataset(result_xr.fillna(NAN), file_name)

            # next
            del result_xr


        durations.append(time.time() - itr_start)
        logging.info(f"{datetime.now()} average duration: {sum(durations) / len(durations):.4f} seconds")


    # - save datasets as one file

    # mask_coords = mask_data.drop_dims('time').coords
    # mask_data.close()

    # for stat in stats_to_get:
    #     # search
    #     all_files = glob.glob(str(check_dir / f"{stat}/{file_times.split('_')[0]}*.nc"))
    #     if len(all_files) == 0:
    #         logging.info(f"{datetime.now()} no files found for {stat}")
    #         continue

    #     # collect
    #     logging.info(f"{datetime.now()} combining files for {stat}")
    #     datasets = []
    #     for f in all_files:
    #         datasets.append(xr.Dataset(coords=mask_coords).merge(xr.open_mfdataset(f)))
    #     all_datasets = xr.concat(datasets, dim='time')

    #     # save
    #     file_path = result_dir / f"{stat}/{file_times}.nc"
    #     file_path.parent.mkdir(parents=True, exist_ok=True)
    #     logging.info(f"{datetime.now()} found {len(datasets)} files for {stat} to save to {file_path}")
    #     logging.info(f"{datetime.now()} saving...")
    #     all_datasets.fillna(NAN).to_netcdf(file_path)
    #     logging.info(f"{datetime.now()} saved {stat} results to {file_path}")

    logging.info(f"{datetime.now()} done with {file}.")


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
    fres_regex = f"{result_dir}/{res_fname}/*" # stats results
    
    input_files = get_files(finput_regex)
    # res_files = get_files(fres_regex)
    res_files = []

    # - go
    logging.info(f"{datetime.now()} input files: {input_files}")
        
    if (overwrite or (len(res_files) < len(input_files))):
        task_start = time.time()
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
    logging.info(f"{proc_start} Commencing calculations")
    logging.info(f"Configuration file: {args.yaml}")
    logging.info(f"Start date: {start_date.isoformat()}")
    logging.info(f"End date: {end_date.isoformat()}")

    main(args.yaml, start_date, end_date)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")



