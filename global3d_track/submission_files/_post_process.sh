#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=proc
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --mem=150GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/proc/job.o%j

# For 7 days in the Amazon it:
# Took approx. 1hr 40min to complete linking.
# Took approx. 2 hours to complete the definitions for data filtering.
 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

yaml=$1
start_date=$2
end_date=$3

echo "submitted with arguments: " $yaml $start_date $end_date

cd /home/b/b382635/s/global3d_track
python -m global3d_track.scripts.track.master $yaml -s $start_date -e $end_date