#!/bin/bash

#SBATCH --job-name=test_grl_fast
#SBATCH --partition=interactive
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/stats/test_grl_fast.o%j

module load python3
source /home/b/b382635/.bashrc
source activate hackathon_env

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.analysis.test_grl_fast
