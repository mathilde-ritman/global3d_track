#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=stats
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --mem=200GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/stats/job.o%j

module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

# yaml=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_1.yaml
# yaml=/home/b/b382635/s/my_notebooks/CSU/paper_working/tracking_configs/get_stats_csu_10km.yaml
# yaml=/home/b/b382635/s/my_notebooks/CSU/paper_working/tracking_configs/get_stats_csu_5km.yaml

yaml_file=$1
cd /home/b/b382635/s/global3d_track

python -m global3d_track.scripts.analysis.get_stats_csu $yaml_file
