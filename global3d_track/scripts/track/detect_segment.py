'''

Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import yaml
import pickle
import logging
import warnings
import argparse
import functools
import xarray as xr
import os
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
from ..src import utils, methods

'''

General tobac detect and segment processor. Output all goes to the designated feature data directory.

'''


def detect_segment_object(di, obj_name, start_date, end_date):

    # - run and checkpoint management

    overwrite = di['overwrite']
    data_dir = Path(di['data_directory']) / utils.tools.version_name(di, use_tobac_version=True, start_date=start_date)
    data_dir.mkdir(parents=True, exist_ok=True)
    obj_di = di['objects'][obj_name] # choices for the variable being tracked
    version_suffix = f"{obj_di['name']}/{start_date.strftime('T%H%M')}_{end_date.strftime('T%H%M')}"
    tobac_config = utils.tools.load_yaml(obj_di['tobac_config'])
    shutil.copy2(obj_di['tobac_config'], data_dir) # record parameter setup with data output

    # - details

    region = di['region']
    PBC_flag = None
    if region == 'tropics':
        PBC_flag = "hdim_2"
    modify_parameters = dict(savedir=data_dir, PBC_flag=PBC_flag, version_name=version_suffix)
    
    # - load data

    sel, seg = tobac_config['select_data'], tobac_config['segment_data']
    # ensure is list
    if not isinstance(sel, list):
        sel = [sel,]
    if not isinstance(seg, list):
        seg = [seg,]
    selseg = list(set(sel+seg))
    logging.info(f"{datetime.now()} loading data for detection and segmentation: {selseg}")
    data = utils.data_tools.load_tobac_data(selseg, di['region'], start_date, end_date, di['model_version'])
    # add input variables if multiple (e.g., cli+clw)
    if isinstance(sel, list):
        data['+'.join(sel)] = functools.reduce(np.add, [data[var] for var in sel])
        sel = '+'.join(sel)
    if isinstance(seg, list):
        data['+'.join(seg)] = functools.reduce(np.add, [data[var] for var in seg])
        seg = '+'.join(seg)

    # - processing

    module = methods.tobac_wrapper.Track(data[sel], data[seg], tobac_config, overwrite=overwrite, track_params=modify_parameters)
    tobac_methods = {'detect': True, 'segment': True, 'save': True}
    module.perform(**tobac_methods)



def perform(yaml_file, start_date, end_date):

    track_di = utils.tools.load_yaml(yaml_file)
    objects_to_track = track_di['objects'].keys()

    # detect and segment each object as per yaml

    for obj in objects_to_track:
        logging.info(f"{datetime.now()} procesing object {obj}")
        detect_segment_object(track_di, obj, start_date, end_date)
        logging.info(f"{datetime.now()} done with {obj}.")
        



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

