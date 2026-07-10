#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=stats
#SBATCH --partition=shared
#SBATCH --time=4:00:00
#SBATCH --mem=200GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/stats/job.o%j

 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_2.yaml

echo "submitted with arguments: " $yaml

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.analysis.get_stats $yaml


