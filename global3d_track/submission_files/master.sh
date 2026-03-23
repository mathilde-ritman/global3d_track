#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=master
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00
#SBATCH --mem=1GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/master/master.o%j

module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

# yaml=/home/b/b382635/s/global3d_track/global3d_track/tracking_configs/track_master.yaml
# yaml=/home/b/b382635/s/global3d_track/global3d_track/tracking_configs/track_master_high_res.yaml
# yaml=/home/b/b382635/s/my_notebooks/CSU/tracking_configs/track_master.yaml
yaml=/home/b/b382635/s/my_notebooks/model_paper/grl_submission/tracking_configs/track_master.yaml
# yaml=/home/b/b382635/s/my_notebooks/model_paper/grl_submission/tracking_configs/track_master_no_ec.yaml

# yaml=/home/b/b382635/s/global3d_track/global3d_track/tracking_configs/track_master_dev.yaml

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.track.master $yaml


# # & now compare to results using g3d:
# yaml=/home/b/b382635/s/my_notebooks/CSU/tracking_configs/track_master_g3d.yaml
# python -m global3d_track.scripts.track.master $yaml