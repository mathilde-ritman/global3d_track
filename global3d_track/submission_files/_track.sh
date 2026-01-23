#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=track_many
#SBATCH --partition=shared
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/master/track.o%j

# specifications
yaml=$1
start_date=$2
end_date=$3

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
chmod +x /home/b/b382635/s/global3d_track/global3d_track/submission_files/_track_itr.sh
while [ $current_end_date_sec -le $end_date_sec ]
do
    # Convert the current date in seconds to the YYYY-MM-DD format
    current_start_dt=$(date -d "@$current_date_sec" +%Y-%m-%d-%H:%M:%S)
    current_end_dt=$(date -d "@$current_end_date_sec" +%Y-%m-%d-%H:%M:%S)

    # Submit job
    echo submitted for start $datetime
    sbatch /home/b/b382635/s/global3d_track/global3d_track/submission_files/_track_itr.sh $yaml_file $current_start_dt $current_end_dt

    # Next iteration
    current_date_sec=$current_end_date_sec
    current_end_date_sec=$(($current_end_date_sec + $hours_sec))

    ((count++))

    if [ $count -eq 2 ]; then
        echo "Stopping for testing purposes."
        break
    fi

done

echo $count jobs submitted