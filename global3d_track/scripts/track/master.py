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

'''
master process to manage cloud tracking over one dataset with multiple slurm job submission scripts.

'''

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

def wait_for_job(job_id, poll_interval=60):
    while True:
        state = get_job_state(job_id)

        if state == "COMPLETED":
            logging.info(f"{time.ctime()}: Job {job_id} completed successfully")
            return

        if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
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
        
    # detect and segment
    job_id = submit_job(submission_files / '_detect.sh', (config, start_date, end_date, detect_hours))
    wait_for_job(job_id)

    # track
    job_id = submit_job(submission_files / '_track.sh', (config, start_date, end_date, track_hours))
    wait_for_job(job_id)

    # post process
    job_id = submit_job(submission_files / '_post_process.sh', (config, start_date, end_date))
    wait_for_job(job_id)

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

