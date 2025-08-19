#!/bin/bash


# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=detect_itr
#SBATCH --partition=shared
#SBATCH --time=00:20:00
#SBATCH --mem=200GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/detect_segment/job.o%j


# takes approx 10 mins to process Amazon for 12 hours. No issues at 250 GB.
 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml=$1
start_date=$2
end_date=$3

echo "submitted with arguments: " $yaml $start_date $end_date

# python /home/b/b382635/s/global3d_track/global3d_track/scripts/track/detect_segment.py $yaml -s $start_date -e $end_date

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.track.detect_segment $yaml -s $start_date -e $end_date