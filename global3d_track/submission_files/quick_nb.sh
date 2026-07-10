#!/bin/bash


# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=notebook
#SBATCH --partition=shared
#SBATCH --time=02:00:00
#SBATCH --mem=100GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/notebook/job.o%j

 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

# nb_path=/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/data_preparation/03.aggregate_statistics
# nb_path=/home/b/b382635/s/my_notebooks/CSU/07.1.track_comparison_statistics
# nb_path='/home/b/b382635/s/my_notebooks/model_paper/grl_submission/scripts/analysis/02.view_tropics'
# nb_path='/home/b/b382635/s/my_notebooks/Maor/02.develop_tracking'
# nb_path='/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/data_analysis/11.condensate_histograms'
nb_path='/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/data_analysis/rev.03.sensitivity'

jupyter nbconvert $nb_path.ipynb --to python

# python $nb_path.py

batch=7
size=15

while (( batch * size <= 1000 ))
do
    # submit job
    echo "Processing script for batch: $batch with size: $size"
    python $nb_path.py $batch $size

    # next iteration
    batch=$((batch + 1))

done
