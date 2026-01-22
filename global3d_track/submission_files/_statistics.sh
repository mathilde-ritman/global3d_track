#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=analysis
#SBATCH --partition=shared
#SBATCH --time=08:00:00
#SBATCH --mem=150GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/analysis/job.o%j

 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml=$1
start_date=$2
end_date=$3

echo "submitted with arguments: " $yaml $start_date $end_date

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.analysis.systemwise_statistics $yaml


