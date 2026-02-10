#!/bin/bash


# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=notebook
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=250GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/notebook/job.o%j

 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

nb_path=/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/data_preparation/01.vertical_velocity_histograms

jupyter nbconvert $nb_path.ipynb --to python
n=1

while (( n <= 100 ))
do
    # submit job
    echo "Processing script for itr: $n"
    python $nb_path.py $n

    # next iteration
    n=$((n + 1))

done
