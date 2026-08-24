#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=detect_itr
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --mem=300GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/detect_segment/job.o%j

# takes approx 10 mins to process Amazon for 12 hours. No issues at 250 GB.
# taks 17 mins to process tropics (to eq+-15 degrees) for 4 hours. No issues at 100 GB.

# Trpoics:
# [1H, 100GB, shared] OOM kill at data load
# [1H, 100GB, compute] OOM kill at data load
# [1H, 200GB, compute] done, took ~40 mins per 4 hours of data

# Maor [starting from 12PM]:
# [1H, 100GB, shared] took 15 mins to complete

# Maor [full 36 hours]:
# [1H, 100GB, shared]

# CSU high-res
# [1H, 100GB, shared] OOM
# [1H, 100GB, compute] OOM
# [1H, 400GB, compute] Got through data loading
# [1H, 400GB, compute] Failed after loading data
# [1H, 400GB, compute] Now for just 2H of data, finished in 35 mins


module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml_file=$1
start_date=$2
end_date=$3

echo "submitted with arguments: " $yaml_file $start_date $end_date

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.track.detect_segment --yaml $yaml_file -s $start_date -e $end_date