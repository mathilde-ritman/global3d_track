#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=track_itr
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --mem=300GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/track/job.o%j

# Amazon:
# takes 1hr - 1hr40min to run 1 day over the Amazon. No issues at 250 GB.
# takes 1hr - 1hr15min to run 1 day over the Amazon with 2 nodes and 128 tasks per node and 2 cpus per task. No issues at 250 GB.

# Tropics:
# [2H, 300GB, 1node, 64tasks, 1cpuptask] took 60 mins to get to applying the share labels mapping for updrafts, then OOM kill. 
# [2H, 300GB, 1node, 1task 48cpuptask] took 60 mins to share the label mappings for updrafts.
# [4H, 250GB, 1node, 1task 48cpuptask] slowly working its way through erosion, checkpoints very necessary. Got to applying share labels, them OOM.
# [4H, 300GB, 1node, 1task 48cpuptask] OOM kill at force labels to be consecutive
# [2H, 500GB, 1node, 1task 48cpuptask] finished frozen force consecutive, then ran out of time
# [2H, 250GB, 1node, 1task 48cpuptask] finished requiring overlap, OOM at share labels for the final system result
# [2H, 350GB, 1node, 1task 48cpuptask] finished sucessfully
# [2H, 150GB, 1node, 1task 48cpuptask] OOM kill at find label mappings



module load python3
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml_file=$1
start_date=$2
end_date=$3

echo "submitted with arguments: " $yaml_file $start_date $end_date

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.track.custom_track --yaml $yaml_file -s $start_date -e $end_date