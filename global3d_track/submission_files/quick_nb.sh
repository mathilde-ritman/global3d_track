#!/bin/bash


# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=notebook
#SBATCH --partition=shared
#SBATCH --time=03:00:00
#SBATCH --mem=100GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/notebook/job.o%j

 
module load python3 
source /home/b/b382635/.bashrc
source activate hackathon_env

nb_path=/home/b/b382635/s/my_notebooks/dataset_paper/acp_submission/scripts/03.aggregate_statistics
# nb_path=/home/b/b382635/s/my_notebooks/conv_anvil_paper/12.data_results
# nb_path="/home/b/b382635/s/my_notebooks/EGU2025/2.3.results_big"
# nb_path=/home/b/b382635/s/my_notebooks/dataset_paper/02.evaluation_using_stats
# nb_path="/home/b/b382635/s/my_notebooks/conv_anvil_paper/01.cmfi"
# nb_path=/home/b/b382635/s/my_notebooks/conv_anvil_paper/02.cmfi_anvil
# nb_path=/home/b/b382635/s/my_notebooks/conv_anvil_paper/11.data_filtering

# jupyter nbconvert $nb_path.ipynb --to python
# python $nb_path.py

jupyter nbconvert $nb_path.ipynb --to python
batch=14
size=50

while (( batch * size <= 1200 ))
do
    # submit job
    echo "Processing script for batch: $batch with size: $size"
    python $nb_path.py $batch $size

    # next iteration
    batch=$((batch + 1))

done
