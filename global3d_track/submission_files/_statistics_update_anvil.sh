#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=update_anvil_stats
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --mem=300GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/stats/job.o%j

module load python3
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml_file=$1
cd /home/b/b382635/s/global3d_track

python -m global3d_track.scripts.analysis.update_anvil_stats $yaml_file
