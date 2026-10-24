#!/bin/bash

# Mathilde Ritman
# University of Oxford
#
# Runs the grl_stats stream monitor (monitor.py) as a proper SLURM job instead
# of a login-node background process, since login nodes reap long-lived stray
# processes. Uses the interactive partition (not compute/shared) because the
# bb1153 account only allows 2 concurrent compute/shared jobs, which the two
# grl_stats streams already consume; interactive has its own separate job
# quota. Time limit here must match MONITOR_JOB_TIME_LIMIT_SECONDS in
# monitor.py - the monitor resubmits itself with time to spare, so it never
# gets cut off mid-cycle.

#SBATCH --job-name=grl_monitor
#SBATCH --partition=interactive
#SBATCH --time=11:30:00
#SBATCH --mem=1GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/s/global3d_track/global3d_track/submission_files/monitor/slurm_monitor.o%j

module load python3
source /home/b/b382635/.bashrc
source activate hackathon_env

cd /home/b/b382635/s/global3d_track/global3d_track/submission_files/monitor
python3 -u monitor.py
