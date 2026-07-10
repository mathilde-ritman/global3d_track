import subprocess
import logging
import time
import sys
from pathlib import Path
from ..src import utils
import argparse
import shutil
import os
import re
import glob
import pathlib
from datetime import datetime, timedelta

'''
master process to manage cloud tracking over one dataset with multiple slurm job submission scripts.

'''

#### --------------------------------- ####

def submit_job(script, arguments):
    # submit .sh file and get slurm job ID
    cmd = ["sbatch", str(script)] + list(arguments)
    out = subprocess.run(cmd, check=True, capture_output=True)
    job_id = re.search(rb'job\s+(\d+)', out.stdout).group(1).decode()
    logging.info(f"{time.perf_counter()}: job submitted with ID {job_id}")
    return job_id

def get_job_state(job_id):
    # check whether the job is running, completed or failed
    cmd = [
        "sacct",
        "-j", job_id,
        "--format=State",
        "--noheader"
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    state = out.split()[0]
    return state

def wait_for_job(job_id, poll_interval=300):
    while True:
        state = get_job_state(job_id)

        if state == "COMPLETED":
            logging.info(f"{time.perf_counter()}: Job {job_id} completed successfully")
            return

        if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            raise RuntimeError(f"{time.perf_counter()}: Job {job_id} failed with state {state}")

        time.sleep(poll_interval)
        

#### --------------------------------- ####

def submit_detection(script, config, start_date, end_date, hours):
    di = utils.tools.load_yaml(config)
    job_ids = {}

    # checking if the daily results are done
    if not di['overwrite']:
        logging.info(f"checking tracking output in {pathlib.Path(di['version_name'], di['region'])}")
        current_date = start_date
        while current_date < end_date:
            version_name = utils.tools.version_name(di, start_date=current_date, use_tobac_version=True)
            data_dir = Path(di['data_directory'], version_name)
            files = list(data_dir.glob(f'*/tracked_mask.nc'))
            if len(files) < 2:
                # daily results for ice and updraft not done/collected, start from this point
                start_date = current_date
                break
            current_date = current_date + timedelta(hours=24)
    
    # iterate the time periods
    current_date = start_date
    while current_date < end_date:
        current_end_date = current_date + timedelta(hours=hours)
        version_name = utils.tools.version_name(di, start_date=current_date, use_tobac_version=True)
        data_dir = Path(di['data_directory'], version_name)
        files = list(data_dir.glob(f'*/{current_date.strftime("T%H%M")}_{current_end_date.strftime("T%H%M")}/segmented_mask.nc'))
        if len(files) == 0 or di['overwrite']:
            # period not done, submit job
            logging.info(f"{datetime.now()}: submitting detection for period {current_date} to {current_end_date}")
            params = (config, current_date.strftime("%Y-%m-%d-%H:%M:%S"), current_end_date.strftime("%Y-%m-%d-%H:%M:%S"))
            job_id = submit_job(script, params)
            job_ids[job_id] = (script, params)
        current_date = current_end_date

    return job_ids

def submit_tracking(script, config, start_date, end_date, hours):
    di = utils.tools.load_yaml(config)
    job_ids = {}
    
    # iterate the time periods
    current_date = start_date
    while current_date < end_date:
        current_end_date = current_date + timedelta(hours=hours)
        version_name = utils.tools.version_name(di, start_date=current_date)
        data_dir = Path(di['data_directory'], version_name)
        file = data_dir / f'{current_date.strftime("%Y%m%dT%H%M")}_{current_end_date.strftime("%Y%m%dT%H%M")}_system_tracks.nc'
        if not file.exists() or di['overwrite']:
            # period not done, submit job
            logging.info(f"{datetime.now()}: submitting tracking for period {current_date} to {current_end_date}")
            params = (config, current_date.strftime("%Y-%m-%d-%H:%M:%S"), current_end_date.strftime("%Y-%m-%d-%H:%M:%S"))
            job_id = submit_job(script, params)
            job_ids[job_id] = (script, params)
        current_date = current_end_date

    return job_ids

#### --------------------------------- ####

def submit_many(script, config, start_date, end_date, hours):
    # submit many jobs to cover the time period
    current_start_date = start_date
    job_ids = {}
    while current_start_date < end_date:
        current_end_date = current_start_date + timedelta(hours=hours)
        params = (config, current_start_date.strftime("%Y-%m-%d-%H:%M:%S"), current_end_date.strftime("%Y-%m-%d-%H:%M:%S"))
        job_id = submit_job(script, params)
        job_ids[job_id] = (script, params)
        # next
        current_start_date = current_end_date
    return job_ids

def wait_for_many(job_ids, poll_interval=300):
    # wait for many jobs to complete
    remaining_jobs = list(set(list(job_ids.keys())))
    while remaining_jobs:
        for job_id in remaining_jobs:
            state = get_job_state(job_id)

            if state == "COMPLETED":
                logging.info(f"{time.perf_counter()}: Job {job_id} completed successfully")
                remaining_jobs.remove(job_id)

            elif state in {"TIMEOUT"}:
                logging.warning(f"{time.perf_counter()}: Job {job_id} timed out, resubmitting")
                new_job_id = submit_job(*job_ids[job_id])
                remaining_jobs.append(new_job_id)
                remaining_jobs.remove(job_id)
                
            elif state in {"FAILED", "CANCELLED", "OUT_OF_MEMORY"}:
                raise RuntimeError(f"{time.perf_counter()}: Job {job_id} failed with state {state}")

        time.sleep(poll_interval)

#### --------------------------------- ####

def master(config):

    # load run specifications
    di = utils.tools.load_yaml(config)
    start_date, end_date, detect_hours, track_hours = di['start_date'], di['end_date'], str(di['detect_segment_hours']), str(di['track_hours'])
    version_name = utils.tools.version_name(di)
    os.makedirs(str(Path(di['data_directory']) / version_name), exist_ok=True)
    shutil.copy2(config, Path(di['data_directory']) / version_name)

    submission_files = Path('/home/b/b382635/s/global3d_track/global3d_track/submission_files')

    # parse data
    start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    detect_hours = int(detect_hours)
    track_hours = int(track_hours)
        
    # # detect and segment
    # logging.info(f"{datetime.now()}: sending off detection...")
    # job_ids = submit_detection(submission_files / '_detect_itr.sh', config, start_date, end_date, detect_hours)
    # wait_for_many(job_ids)

    # # track
    # logging.info(f"{datetime.now()}: sending off tracking...")
    # job_ids = submit_tracking(submission_files / '_track_itr.sh', config, start_date, end_date, track_hours)
    # wait_for_many(job_ids)
       
    # post process
    logging.info(f"{datetime.now()}: sending off post-processing...")
    job_id = submit_job(submission_files / '_post_process.sh', (config, start_date.strftime("%Y-%m-%d-%H:%M:%S"), end_date.strftime("%Y-%m-%d-%H:%M:%S")))
    wait_for_job(job_id)


#### --------------------------------- ####

if __name__ == "__main__":

    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="path to configuration file", type=str)
    args = parser.parse_args()

    # go
    proc_start = time.perf_counter()
    logging.info(f"\n{proc_start}: Commencing pipeline with configuration file: {args.yaml}")
    master(args.yaml)
    logging.info(f"{time.perf_counter()}: Finished successfully, time elapsed: {time.perf_counter() - proc_start}")

