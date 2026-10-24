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

cd /home/b/b382635/s/global3d_track
yaml=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_grl.yaml

N=5
prev_job=""

for ((i=0; i<N; i++)); do

    if [[ -z "$prev_job" ]]; then
        job_id=$(sbatch --parsable /home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh "$yaml")
    else
        job_id=$(sbatch --parsable --dependency=afterok:"$prev_job" /home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh "$yaml")
    fi

    echo "submitted job $job_id"
    prev_job="$job_id"

done