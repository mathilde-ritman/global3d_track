'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import pathlib
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


def perform(yaml_file, start_date, end_date):

    # - load yaml and set up

    di = utils.tools.load_yaml(yaml_file)
    overwrite = di['overwrite']
    data_dir = pathlib.Path(di['data_directory'], di['version_name'], di['region'])

    # - what are we doing?
    
    link_files = di['post_processing'].get('link_files', False)
    filter_tracks = di['post_processing'].get('filter_tracks', False)
    if not link_files and not filter_tracks:
        warnings.warn("you haven't chosen and post processing steps...? Exiting.")
        exit()

    # - done already?!
        
    def get_files(regex):
        if len(glob.glob(regex)) == 0:
            return []
        return utils.tools.sort_files([f for f in glob.glob(regex) if utils.tools.check_file_dates(f, start_date, end_date)])
    
    link_fname_suffix = '_linked'
        
    fraw_regex = f"{data_dir}/*/*system_tracks.nc" # all files in directory for the raw tracking output
    flink_regex = f"{data_dir}/*/*system_tracks{link_fname_suffix}.nc" # linked tracking results
    
    raw_files = get_files(fraw_regex)
    linked_files = get_files(flink_regex)

    if not raw_files:
        logging.warning(f"no files found matching {fraw_regex}")

    # if need linking do that
        
    if link_files and (overwrite or (len(linked_files) < len(raw_files))):
        task_start = datetime.now()
        logging.info(f"{task_start} Commencing linking")

        # # where di we get to?
        # remaining_files = [f for f in raw_files if f.replace('.nc',f'{link_fname_suffix}.nc') not in linked_files]
        # last_completed_file = linked_files[-1] if linked_files else None

        # if di['overwrite']:
        remaining_files = raw_files
        last_completed_file = None

        remaining_files = raw_files
        methods.misc.Link().link_files(di, remaining_files, link_fname_suffix, last_completed_file)
        linked_files = get_files(flink_regex)
        logging.info(f"{datetime.now()} linking complete. Took {datetime.now() - task_start}")
    
    else:
        logging.info(f"{datetime.now()} linking already complete.")

    
    

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="path to configuration file", type=str)
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

    perform(args.yaml, start_date, end_date)

    logging.info(f"{datetime.now()} Finished successfully, time elapsed: {datetime.now() - proc_start}")



