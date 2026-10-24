#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=stats
#SBATCH --partition=shared
#SBATCH --time=0:10:00
#SBATCH --mem=1GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/stats/job.o%j

module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

# yaml=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_1.yaml
# yaml=/home/b/b382635/s/my_notebooks/CSU/paper_working/tracking_configs/get_stats_csu_10km.yaml
# yaml=/home/b/b382635/s/my_notebooks/CSU/paper_working/tracking_configs/get_stats_csu_5km.yaml

cd /home/b/b382635/s/global3d_track
# yaml_a=/home/b/b382635/s/my_notebooks/CSU/paper_working/tracking_configs/get_stats_csu_5km.yaml
# yaml_c=/home/b/b382635/s/my_notebooks/CSU/paper_working/tracking_configs/get_stats_csu_10km.yaml
# yaml_a=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/update_anvil_stats.yaml
# yaml_b=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/update_anvil_stats1000.yaml

prev_a=""
prev_b=""

yaml_a=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_grl.yaml
yaml_b=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_grl1000.yaml

N=1

for ((i=0; i<N; i++)); do

    if [[ -z "$prev_a" ]]; then
        job_a=$(sbatch --parsable \
            /home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh \
            "$yaml_a")
    else
        job_a=$(sbatch --parsable \
            --dependency=after:"$prev_a" \
            /home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh \
            "$yaml_a")
    fi

    if [[ -z "$prev_b" ]]; then
        job_b=$(sbatch --parsable \
            /home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh \
            "$yaml_b")
    else
        job_b=$(sbatch --parsable \
            --dependency=after:"$prev_b" \
            /home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh \
            "$yaml_b")
    fi

    prev_a="$job_a"
    prev_b="$job_b"

done
