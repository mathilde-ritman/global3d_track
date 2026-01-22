#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=track_itr
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/track/job.o%j

# Amazon:
# takes 1hr - 1hr40min to run 1 day over the Amazon. No issues at 250 GB.
# takes 1hr - 1hr15min to run 1 day over the Amazon with 2 nodes and 128 tasks per node and 2 cpus per task. No issues at 250 GB.

# Tropics:
# took 30 mins to get to share_labels for updrafts, then couldn't finish sharing, on 1 day over the tropics with 500 GB, 4 nodes, 34 tasks per node and 1 cpu per task.
# running with checkpoints for 8 hrs, with 500 GB, 1 node, 64 tasks per node and 1 cpu per task.
# took ~4hrs to perform erode-connect ice tracking (settings above).
# OOM at 500GB, 32 nodes, batch size for label mapping of 200.

module load python3
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml=$1
start_date=$2
end_date=$3

echo "submitted with arguments: " $yaml $start_date $end_date

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.track.custom_track $yaml -s $start_date -e $end_date