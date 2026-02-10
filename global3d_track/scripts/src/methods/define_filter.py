'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2025

'''


import logging
import xarray as xr
import pandas as pd
import os
import numpy as np
import time
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from ..utils import checkpoint, definitions, tools, data_tools
Checkpoint = checkpoint.Checkpoint


# functions to define the anvil base height


def process_group(mask, data):
    # unstack
    mask = mask.unstack([x for x in mask.dims][0])
    mask = mask.sortby('lat').sortby('lon')
    # check sufficient vertical extent 
    if mask.level_full.size < 3:
        ABH = [np.nan] * mask.time.size
    else:
        # find ABH at time t
        masked_data = data[['cli','clw']].where(mask.system>0)
        cl_profile = (masked_data.cli + masked_data.clw).sum(('lat','lon'))
        ABH = definitions.discover_abh(cl_profile).values
    # get max cloud height (note model levels decrease with height)
    max_cloud_height = mask.system.max(('lat','lon')).sortby('level_full').idxmin('level_full').values
    # check core exists at time t
    core_exists = (mask.u_tracks.max(('level_full','lat','lon')).values > 0)
    # ensure output shape okay
    if mask.time.size < data.time.size:
        n_pad = data.time.size - len(ABH) # pad end of arrays with nan
        ABH = np.append(ABH, [np.nan] * n_pad)
        max_cloud_height = np.append(max_cloud_height, [np.nan] * n_pad)
        core_exists = np.append(core_exists, [np.nan] * n_pad)
    return ABH, max_cloud_height, core_exists

# calculating the above

def iterate_groups_define(di, files, check_dir, data_dir):

    """
    Iterates through each file separately before processing time chunks.
    Processes tracked clouds and calculates a statistic for each time step.

    Args:
        files (list): List of file paths to process.
        check_dir (str or Path): path to checkpoint progress

    Returns:
        DataFrame: Processed results with statistics for each system over time.
    """

    start_date, end_date = di['start_date'], di['end_date']
    region = di['region']
    t_chunk = di['batch_size']['define']
    checkpoint = Checkpoint(Path(check_dir) / 'quantities', overwrite=di['restart_checkpoints'])

    # find last checkpoint
    last_checkpoint = checkpoint.get_last_checkpoint(regex="system_abh")
    if last_checkpoint and not di['overwrite']:
        logging.info(f"Resuming from last checkpoint: {last_checkpoint}")
        abh_df = pd.read_csv(last_checkpoint, index_col='system_id')
        cth_df = pd.read_csv(last_checkpoint.replace('abh','cth'), index_col='system_id')
        core_df = pd.read_csv(last_checkpoint.replace('abh','core_exists'), index_col='system_id')
        last_time_reached = pd.to_datetime(abh_df.columns[-1]) + timedelta(minutes=15)
        file_dates = [pd.to_datetime(Path(f).parts[-2]) for f in files] # dates of each file
        files = [f for f, d in zip(files, file_dates) if d.date() >= last_time_reached.date()] # only process files after or includinglast checkpoint

    else:
        # initalise
        last_time_reached = pd.to_datetime(start_date)
        abh_df = pd.DataFrame()
        cth_df = pd.DataFrame()
        core_df = pd.DataFrame()

    def make_df(array, index, columns):
        df = pd.DataFrame(array, index=index, columns=columns)
        df.index.name = 'system_id'
        return df

    durations = []
    prev_fpath = None
    for file_name in files:
        logging.info(f"{datetime.now()} Processing file: {file_name}")

        # Load and open dataset
        mask_data = xr.open_dataset(file_name)
        mask_data = mask_data.where(mask_data>0)

        for t_idx in range(0, mask_data.time.size, t_chunk):
            itr_start = time.time()

            # Select time chunk
            mask_i = mask_data.isel(time=slice(t_idx, t_idx + t_chunk))
            current_time = mask_i.time[0].values
            next_time = mask_i.time[-1].values

            if pd.to_datetime(next_time) < last_time_reached:
                logging.info(f"{datetime.now()} Skipping chunk from {current_time} to {next_time}, already processed.")
                continue
            
            # Load corresponding data
            logging.info(f"{datetime.now()} Loading data for period {current_time} to {next_time}...")
            data_i = data_tools.load_corresponding_data(mask_i, region)
            data_i = data_i.sel(lat=mask_i.lat, lon=mask_i.lon).sel(level_full=slice(40,90))

            # Group mask data
            logging.info(f"{datetime.now()} Grouping...")
            groups = mask_i.groupby(mask_i.system)
            group_keys = list(groups.groups.keys())

            # Process each group in parallel
            logging.info(f"{datetime.now()} Calculating statistics for each group...")
            results = joblib.Parallel(n_jobs=-1, prefer="threads")(
                joblib.delayed(process_group)(groups[k], data_i) for k in group_keys
            )

            # collect results for anvil base heights
            abh_df_i = make_df([result[0] for result in results], group_keys, mask_i.time.values)
            abh_df = pd.concat((abh_df, abh_df_i), axis=1)

            # collect results for cloud top heights
            cth_df_i = make_df([result[1] for result in results], group_keys, mask_i.time.values)
            cth_df = pd.concat((cth_df, cth_df_i), axis=1)

            # and for the presence of a core
            core_df_i = make_df([result[2] for result in results], group_keys, mask_i.time.values)
            core_df = pd.concat((core_df, core_df_i), axis=1)

            # cleanup
            del data_i, mask_i
            durations.append(time.time() - itr_start)
            avg_duration = sum(durations) / len(durations)
            logging.info(f"{datetime.now()} Average duration: {avg_duration:.4f} seconds")

            # save checkpoint
            fname = f"system_abh-{pd.to_datetime(start_date).strftime('%Y%m%dT%H%M')}_{pd.to_datetime(next_time).strftime('%Y%m%dT%H%M')}"
            checkpoint.checkpoint_dataframe(abh_df, fname)
            current_fpath = checkpoint.record[fname]
            cth_df.to_csv(current_fpath.replace('abh','cth'))
            core_df.to_csv(current_fpath.replace('abh','core_exists'))
            if prev_fpath:
                os.system(f'rm {prev_fpath}')
                os.system(f'rm {prev_fpath.replace("abh","cth")}')
                os.system(f'rm {prev_fpath.replace("abh","core_exists")}')
            prev_fpath = current_fpath

        # Close dataset after processing all time chunks
        mask_data.close()
    
    # save final
    final_fpath = current_fpath.replace(str(checkpoint.checkpoint_dir), str(data_dir))
    abh_df.to_csv(final_fpath)
    cth_df.to_csv(final_fpath.replace('abh','cth'))
    core_df.to_csv(final_fpath.replace('abh','core_exists'))

    logging.info(f"{datetime.now()} Final results at: {current_fpath}, {current_fpath.replace('abh','cth')} and {current_fpath.replace('abh','core_exists')}")
    return abh_df, cth_df, core_df


def filter_results(di, df_abh, df_cth, df_core, save_dir):
    ''' Apply data filtering requirements using the quantities calculated in the passed dataframes. '''

    save_dir = Path(save_dir)

    # calculate system lifetimes, and save
    object_start = df_abh.notna().idxmax(axis=1)
    object_end = df_abh.notna().iloc[:,::-1].idxmax(axis=1)
    lifetime = pd.to_datetime(object_end) - pd.to_datetime(object_start)
    lifetime.astype(str).to_csv(save_dir / "system_lifetime.csv")

    # calculate mean anvil base height for each system, and apply threshold
    abh_series = df_abh.mean(axis=1)
    abh_series.to_csv(save_dir / "system_anvil_base_height.csv")

    # find anvils - get times where cloud top height is above the anvil base height (< used as these are model levels which decrease with height)
    anvil_exists = df_cth.apply(lambda x: x <= abh_series, axis=0)
    anvil_exists.to_csv(save_dir / "system_anvil_exists.csv")

    # exclude systems with lifetimes less than threshold
    lifetime_threshold = di['post_processing']['require_lifetime']
    df_keep = pd.Series(True, index=df_abh.index)
    if lifetime_threshold > 0:
        df_keep = df_keep.where(lifetime > timedelta(hours=di['post_processing']['require_lifetime']), False)

    # exclude systems where the core doesn't exist at the anvil start time
    core_must_preceed = di['post_processing']['require_core_first']
    if core_must_preceed:
        anvil_start = anvil_exists.idxmax(axis=1)
        core_start = df_core.idxmax(axis=1)
        core_precedes = core_start <= anvil_start
        df_keep = df_keep.where(core_precedes, False)

    # save systems to keep
    df_keep.to_csv(save_dir / "systems_to_keep.csv")


def apply_threshold(da, value):
        da = da.unstack([x for x in da.dims][0]).system
        return da.where(da.level_full <= value)


def iterate_groups_apply(di, files, anvil_tresholds, check_dir):

    NAN = -9
    start_date, end_date = di['start_date'], di['end_date']
    current_day = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0)
    t_chunk = di['batch_size']['filter']

    checkpoint = Checkpoint(Path(check_dir) / 'filtered_masks')
    last_checkpoint = checkpoint.get_last_checkpoint(regex="filtered_anvil")
    last_time_reached = pd.to_datetime(last_checkpoint.split('_')[-1].split('.')[0]) if last_checkpoint else pd.to_datetime(start_date)

    # iterate each data file, to reduce memory overhead
    count = 0
    while current_day < pd.to_datetime(end_date):
        file_name = files[count]
        processed_file = file_name.replace('.nc', '_proc.nc')

        if Path(processed_file).exists() and not di['overwrite']:
            logging.info(f"{datetime.now()} skipping: {file_name}")
            current_day += timedelta(days=1)
            count += 1
            continue

        logging.info(f"{datetime.now()} processing: {file_name}")
        next_day = current_day + timedelta(days=1)
        mask_day = xr.open_dataset(file_name).sel(time=slice(current_day, next_day - timedelta(seconds=1)))
        mask_day = mask_day.where(mask_day > 0)

        # reduce number of systems to process according to the dataframe provided
        n_groups_big = np.unique(mask_day.system).size
        mask_day = mask_day.where(mask_day.system.isin(anvil_tresholds.index))
        n_groups_reduced = np.unique(mask_day.system).size
        logging.info(f"{datetime.now()} filtered from {n_groups_big} to {n_groups_reduced} groups")

        # iterate times in file and apply threshold
        anvil_day = None
        prev_fname = None
        for t_idx in range(0, mask_day.time.size, t_chunk):

            # Select time chunk
            mask_i = mask_day.isel(time=slice(t_idx, t_idx + t_chunk))
            time_i, time_i_end = mask_i.time[0].values, mask_i.time[-1].values

            if time_i_end < last_time_reached:
                logging.info(f"{datetime.now()} Skipping chunk from {time_i} to {time_i_end}, already processed.")
                continue
            logging.info(f"{datetime.now()} processing times: {time_i} to {time_i_end}")
            
            # group
            groups = mask_i.groupby(mask_i.system)
            group_keys = list(groups.groups)

            # apply threshold in parallel
            results = joblib.Parallel(n_jobs=-1, prefer="threads")(
                joblib.delayed(apply_threshold)(groups[k], anvil_tresholds[k]) for k in group_keys
            )
            result = xr.concat(results, dim='_system', coords='all').assign_coords({'_system': group_keys})
            anvil_i = result.sum('_system')

            # merge results with previous iteration
            anvil_day = xr.concat((anvil_day, anvil_i), dim='time') if anvil_day is not None else anvil_i

            # checkpoint
            fname = f"filtered_anvil-T{pd.to_datetime(current_day).strftime('%Y%m%dT%H%M')}_T{pd.to_datetime(time_i_end).strftime('%Y%m%dT%H%M')}"
            checkpoint.checkpoint_dataset(anvil_day.fillna(NAN).astype(np.int64), fname)
            if prev_fname:
                prev_fpath = checkpoint.record[prev_fname]
                os.system(f'rm {prev_fpath}')
            prev_fname = fname

            del mask_i, groups, result, anvil_i  # Free memory

        mask_day['anvil'] = anvil_day
        tools.compress_and_save(mask_day.fillna(NAN).astype(np.int64), processed_file)
        del mask_day  # Free memory

        current_day += timedelta(days=1)
        count += 1
