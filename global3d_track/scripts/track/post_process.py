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
import numpy as np
from pathlib import Path
from datetime import datetime
from ..src import utils, methods

'''

General tracking processor.

'''


def main(yaml_file, start_date, end_date):

    # - load yaml and set up

    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    version = f"{di['link']['name']}/{di['region']}"
    data_dir = Path(di['data_directory']) / version

    # - what are we doing?
    
    link_files = define_anvil = require_lifetime = False
    if di['post_processing'].get('link_files', 0):
        link_files = True # we want to link tracks across files
    if di['post_processing'].get('define_anvil'):
        define_anvil = True # we want to define the cloud anvil for each tracked cloud
    if di['post_processing'].get('require_lifetime'):
        require_lifetime = di['post_processing']['require_lifetime'] # get rid of clouds that don't live long enough
    if link_files + define_anvil + require_lifetime == 0:
        warnings.warn("you haven't chosen and post processing steps...? Exiting.")
        exit()

    # - done already?!
        
    def get_files(regex):
        if len(glob.glob(regex)) == 0:
            return []
        return utils.tools.sort_files([f for f in glob.glob(regex) if utils.tools.check_file_dates(f, start_date, end_date)])
    
    link_fname_suffix = '_linked'
    proc_fname_suffix = '_proc'
        
    fraw_regex = f"{data_dir}/*/*system_tracks.nc" # all files in directory for the raw tracking output
    flink_regex = f"{data_dir}/*/*system_tracks{link_fname_suffix}.nc" # linked tracking results
    fproc_regex = f"{data_dir}/*/*system_tracks{proc_fname_suffix}.nc" # linked tracking results with anvil heights
    
    raw_files = get_files(fraw_regex)
    linked_files = get_files(flink_regex)
    proc_files = get_files(fproc_regex)

    if not raw_files:
        logging.warning(f"no files found matching {fraw_regex}")

    # if need linking do that
        
    if link_files and (overwrite or (len(linked_files) < len(raw_files))):
        task_start = datetime.now()
        logging.info(f"{task_start} Commencing linking")

        if not di['overwrite']:
            remaining_files = [f for f in raw_files if f.replace('.nc',f'{link_fname_suffix}.nc') not in linked_files]
        else:
            remaining_files = raw_files

        vars_to_update = [f'{x}_tracks' for k in di['objects'] for x in di['objects'][k]['shortname'] if di['objects'][k]['keep']] + ['system']
        
        methods.link.link_files(remaining_files, vars_to_update, fname_suffix=link_fname_suffix)
        linked_files = get_files(flink_regex)
        logging.info(f"{datetime.now()} linking complete. Took {datetime.now() - task_start}")
    
    else:
        logging.info(f"{datetime.now()} linking already complete.")

    # if need definitions and tracks filtering do that
                
    if (define_anvil or require_lifetime) and (overwrite or (len(proc_files) < len(raw_files))):
        task_start = datetime.now()
        logging.info(f"{task_start} Entering anvil definition and lifetime requirement process")

        if not di['overwrite']:
            remaining_files = [f for f in linked_files if f.replace('.nc',f'{proc_fname_suffix}.nc') not in proc_files]
        else:
            remaining_files = linked_files

        logging.info(f"Remaining files: {remaining_files}")

        check_dir = Path(di['checkpoint_directory']) / f'{version}/data_filtering'
        data_dir = Path(di['data_directory']) /  f'{version}/data_filtering_stats'
        check_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # compute needed statistics for each tracked cloud and timestep
        abh_df, cth_df, core_df = methods.define_filter.iterate_groups_define(di, remaining_files, check_dir, data_dir)

        # collect results using dataframe summaries
        logging.info(f"{task_start} Definitions complete, filtering results")
        methods.define_filter.filter_results(di, abh_df, cth_df, core_df, data_dir)

        df_keep = pd.read_csv(data_dir / "systems_to_keep.csv", index_col='system_id')
        abh_series = pd.read_csv(data_dir / "system_anvil_base_height.csv", index_col='system_id')
        anvil_tresholds = abh_series[df_keep.values].iloc[:,0] # mask results and get as series

        # get resulting mask
        logging.info(f"{task_start} Definitions and filtering complete, applying to data and saving the anvil mask.")
        methods.define_filter.iterate_groups_apply(di, remaining_files, anvil_tresholds, check_dir)

        logging.info(f"{datetime.now()} stage complete. Took {datetime.now() - task_start}")

    else:
        logging.info(f"{datetime.now()} filtering and anvil def already complete.")
    
    

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



