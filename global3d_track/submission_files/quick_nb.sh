#!/bin/bash


# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=notebook
#SBATCH --partition=shared
#SBATCH --time=04:00:00
#SBATCH --mem=50GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/notebook/job.o%j

 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

nb_path=/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/data_preparation/03.aggregate_statistics
# nb_path=/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/data_preparation/00.data_filtering


jupyter nbconvert $nb_path.ipynb --to python
batch=15
size=50

while (( batch * size <= 1000 ))
do
    # submit job
    echo "Processing script for batch: $batch with size: $size"
    python $nb_path.py $batch $size

    # next iteration
    batch=$((batch + 1))

done
