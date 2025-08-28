#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=detect_many
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/detect_segment/job.o%j

# Specify configuration file
# yaml_file="/home/b/b382635/s/global3d_track/global3d_track/scripts/track/track_config_tropics.yaml"
yaml_file="/home/b/b382635/s/global3d_track/global3d_track/scripts/track/track_dev_config.yaml"

# CHOOSE OPTIONS
start_date=2021-07-01
end_date=2021-07-03
hours=12 # number of hours of data per job

# Convert the dates to seconds since the Unix epoch
start_date_sec=$(date -d "$start_date" +%s)
end_date_sec=$(date -d "$end_date" +%s)
hours_sec=$(($hours * 3600))

# Loop over each period ($hours) between the start and end dates

# Initialise
count=0
current_date_sec=$start_date_sec
current_end_date_sec=$(($start_date_sec + $hours_sec))
echo processing $region between dates $start_date and $end_date
echo batch period is $hours hours

# Loop
chmod +x /home/b/b382635/s/global3d_track/global3d_track/submission_files/detect_itr.sh
while [ $current_end_date_sec -le $end_date_sec ]
do
    # Convert the current date in seconds to the YYYY-MM-DD format
    current_start_dt=$(date -d "@$current_date_sec" +%Y-%m-%d-%H:%M:%S)
    current_end_dt=$(date -d "@$current_end_date_sec" +%Y-%m-%d-%H:%M:%S)

    # Submit job
    echo submited for start $datetime
    sbatch /home/b/b382635/s/global3d_track/global3d_track/submission_files/detect_itr.sh $yaml_file $current_start_dt $current_end_dt

    # Next iteration
    current_date_sec=$current_end_date_sec
    current_end_date_sec=$(($current_end_date_sec + $hours_sec))

    ((count++))

done

echo $count jobs submitted