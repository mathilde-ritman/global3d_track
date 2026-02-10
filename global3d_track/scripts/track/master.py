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
import datetime

'''
master process to manage cloud tracking over one dataset with multiple slurm job submission scripts.

'''

#### --------------------------------- ####

def submit_job(script, arguments):
    # submit .sh file and get slurm job ID
    cmd = ["sbatch", str(script)] + list(arguments)
    out = subprocess.run(cmd, check=True, capture_output=True)
    job_id = re.search(rb'job\s+(\d+)', out.stdout).group(1).decode()
    logging.info(f"{time.ctime()}: job submitted with ID {job_id}")
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
            logging.info(f"{time.ctime()}: Job {job_id} completed successfully")
            return

        if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            raise RuntimeError(f"{time.ctime()}: Job {job_id} failed with state {state}")

        time.sleep(poll_interval)

#### --------------------------------- ####

def submit_many(script, config, start_date, end_date, hours):
    # submit many jobs to cover the time period
    current_start_date = start_date
    job_ids = []
    while current_start_date < end_date:
        current_end_date = current_start_date + datetime.timedelta(hours=hours)
        job_id = submit_job(script, (config, current_start_date.strftime("%Y-%m-%d-%H:%M:%S"), current_end_date.strftime("%Y-%m-%d-%H:%M:%S")))
        job_ids.append(job_id)
        # next
        current_start_date = current_end_date
    return job_ids

def wait_for_many(job_ids, poll_interval=300):
    # wait for many jobs to complete
    remaining_jobs = set(job_ids)
    while remaining_jobs:
        for job_id in list(remaining_jobs):
            state = get_job_state(job_id)

            if state == "COMPLETED":
                logging.info(f"{time.ctime()}: Job {job_id} completed successfully")
                remaining_jobs.remove(job_id)

            elif state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                raise RuntimeError(f"{time.ctime()}: Job {job_id} failed with state {state}")

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
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    detect_hours = int(detect_hours)
    track_hours = int(track_hours)
        
    # detect and segment
    job_ids = submit_many(submission_files / '_detect_itr.sh', config, start_date, end_date, detect_hours)
    wait_for_many(job_ids)

    # track
    job_ids = submit_many(submission_files / '_track_itr.sh', config, start_date, end_date, track_hours)
    wait_for_many(job_ids)

    # # post process
    # job_id = submit_job(submission_files / '_post_process.sh', (config, start_date, end_date))
    # wait_for_job(job_id)


    logging.info('lets test this out in a bit more detail...')
    # testing again.

#### --------------------------------- ####

if __name__ == "__main__":

    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", help="path to configuration file", type=str)
    args = parser.parse_args()

    # go
    proc_start = time.ctime()
    logging.info(f"\n{proc_start}: Commencing tracking with configuration file: {args.yaml}")
    master(args.yaml)
    logging.info(f"{time.ctime()}: Finished successfully, time elapsed: {time.ctime() - proc_start}")

