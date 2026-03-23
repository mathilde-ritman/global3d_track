''' Mathilde Ritman, 2025 '''


import os
import pickle
import logging
from datetime import datetime
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import re


class Checkpoint:
    ''' 
    Class to manage checkpoints and record progress.
    NOTE: the dataset saver will fill NaN with 0s, essentially assuming that dataset being saved contains only label variables that count upwards from 1. 
    '''

    def __init__(self, checkpoint_dir: str, overwrite=False):
        ''' Create a dict file to record progress and saved checkpoints. '''
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_fpath = self.checkpoint_dir / 'record.pkl'
        
        if self.checkpoint_fpath.exists() and not overwrite:
            self.record = self.load_record()
        else:
            self.record = {}
            self.save_record()
     
    def save_record(self):
        with self.checkpoint_fpath.open('wb') as f:
            pickle.dump(self.record, f)

    def load_record(self):
        with self.checkpoint_fpath.open('rb') as f:
            return pickle.load(f)
        
    def record_action(self, name, data_path):
        self.record[name] = data_path
        self.save_record()

    def checkpoint_reached(self, name):
        return name in self.record
    
    def load_dataset(self, name):
        data_path = Path(self.record[name])
        logging.info(f"{datetime.now()} loading checkpoint at {data_path}")
        ds = xr.open_dataset(data_path)
        return ds
    
    def load_dataarray(self, name):
        data_path = Path(self.record[name])
        logging.info(f"{datetime.now()} loading checkpoint at {data_path}")
        da = xr.open_dataarray(data_path)
        return da

    def checkpoint_dataset(self, ds, name):
        # save dataset to checkpoint
        data_path = self.checkpoint_dir / f"{name}.nc"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        # compress
        if isinstance(ds, xr.DataArray):
            if ds.name is None:
                ds = ds.rename("data")
            if not np.issubdtype(ds.dtype, np.integer):
                ds = ds.astype(np.int32)
            encoding = {ds.name: {"zlib": True, "complevel": 4}}
        else:
            encoding = {}
            for var in ds.data_vars:
                if not np.issubdtype(ds[var].dtype, np.integer):
                    ds[var] = ds[var].astype(np.int32)
                encoding[var] = {"zlib": True, "complevel": 4}
        ds.fillna(0).to_netcdf(data_path, encoding=encoding, engine="h5netcdf")
        # record action
        self.record[name] = str(data_path)
        self.save_record()
        logging.info(f"{datetime.now()} checkpointed {name} to {data_path}")

    def checkpoint_dataframe(self, df, name):
        # save dataframe to checkpoint
        data_path = self.checkpoint_dir / f"{name}.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path)
        # record action
        self.record[name] = str(data_path)
        self.save_record()
        logging.info(f"{datetime.now()} checkpointed {name} to {data_path}")

    def load_dataframe(self, name):
        data_path = Path(self.record[name])
        logging.info(f"{datetime.now()} loading checkpoint at {data_path}")
        return pd.read_csv(data_path)

    def checkpoint_array(self, arr, name):
        # save array to checkpoint
        data_path = self.checkpoint_dir / f"{name}.npy"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(data_path, arr)
        # record action
        self.record[name] = str(data_path)
        self.save_record()
        logging.info(f"{datetime.now()} checkpointed {name} to {data_path}")

    def load_array(self, name):
        data_path = Path(self.record[name])
        logging.info(f"{datetime.now()} loading checkpoint at {data_path}")
        return np.load(data_path)

    def get_last_checkpoint(self, regex=''):
        pattern = re.compile(regex)
        relevant_checkpoints = [k for k in self.record if pattern.search(k)]
        if not relevant_checkpoints:
            result = None
        else:
            result = max(relevant_checkpoints, key=lambda k: int(pattern.search(k).group(1)))
        return result
        
    def remove_file(self, path):
        # remove file safely using pathlib
        path = Path(path)
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            raise RuntimeError(f"failed to remove {path}") from e
        
    def remove_old(self, name):
        data_path = Path(self.record[name])
        if data_path.exists():
            self.remove_file(data_path)
            del self.record[name]
            self.save_record()
        else:
            logging.warning(f"{datetime.now()} checkpoint {data_path} does not exist, cannot remove.")