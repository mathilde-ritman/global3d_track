#!/bin/bash

# Mathilde Ritman
# University of Oxford

#SBATCH --job-name=stats_monitor
#SBATCH --partition=shared
#SBATCH --time=14:00:00
#SBATCH --mem=1GB
#SBATCH --account=bb1153
#SBATCH --output=/home/b/b382635/job_outfiles/tracking/global3d_track/stats/monitor.o%j

# Overnight monitor for the get_stats_grl statistics jobs (a/b, run concurrently).
# Polls sacct every POLL_INTERVAL seconds until DEADLINE_EPOCH. Whenever a job
# finishes it is resubmitted so the pair keeps running:
#   - OUT_OF_MEMORY  -> max_workers decremented by 1 in that job's yaml first
#   - COMPLETED      -> if max_lifetime < 24 in that job's yaml, raise it to 24 first
#   - CANCELLED      -> stop resubmitting that slot (assume user cancelled deliberately)
#   - anything else (TIMEOUT, FAILED, ...) -> resubmit as-is
#
# Usage: sbatch overnight_monitor.sh <deadline_epoch> [job_a_id] [job_b_id]
# If job_a_id/job_b_id are omitted, fresh jobs are submitted for both.

set -uo pipefail

SUBMIT_SCRIPT=/home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh
YAML_A=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_grl.yaml
YAML_B=/home/b/b382635/s/global3d_track/global3d_track/scripts/analysis/get_stats_grl1000.yaml

LOGDIR=/home/b/b382635/job_outfiles/tracking/global3d_track/stats
LOGFILE="$LOGDIR/overnight_monitor_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}.log"

POLL_INTERVAL=600          # seconds between checks
DEADLINE=${1:?deadline epoch required}
job_a=${2:-}
job_b=${3:-}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"; }

reduce_max_workers() {
    local yaml=$1 current new
    current=$(grep -oP '^max_workers:\s*\K[0-9]+' "$yaml")
    if [[ -z "$current" ]]; then
        log "WARNING: could not parse max_workers in $yaml; leaving unchanged"
        return
    fi
    if (( current <= 1 )); then
        log "WARNING: max_workers already at $current in $yaml; not reducing further"
        return
    fi
    new=$(( current - 1 ))
    sed -i "s/^max_workers:.*/max_workers: $new/" "$yaml"
    log "Reduced max_workers in $yaml from $current to $new"
}

raise_max_lifetime() {
    local yaml=$1 current
    current=$(grep -oP '^max_lifetime:\s*\K[0-9]+' "$yaml")
    if [[ -z "$current" ]]; then
        log "WARNING: could not parse max_lifetime in $yaml; leaving unchanged"
        return
    fi
    if (( current >= 24 )); then
        log "max_lifetime in $yaml already $current (>=24); leaving unchanged"
        return
    fi
    sed -i "s/^max_lifetime:.*/max_lifetime: 24/" "$yaml"
    log "Raised max_lifetime in $yaml from $current to 24"
}

submit_job() {
    sbatch --parsable "$SUBMIT_SCRIPT" "$1"
}

job_state() {
    sacct -n -X -j "$1" --format=State%30 2>/dev/null | tr -d ' '
}

log "=== Monitor (job ${SLURM_JOB_ID:-N/A}) started/resumed, deadline $(date -d "@$DEADLINE") ==="

if [[ -z "$job_a" ]]; then
    job_a=$(submit_job "$YAML_A")
    log "Submitted initial job_a=$job_a (yaml=$YAML_A)"
else
    log "Resuming with existing job_a=$job_a (yaml=$YAML_A)"
fi

if [[ -z "$job_b" ]]; then
    job_b=$(submit_job "$YAML_B")
    log "Submitted initial job_b=$job_b (yaml=$YAML_B)"
else
    log "Resuming with existing job_b=$job_b (yaml=$YAML_B)"
fi

while (( $(date +%s) < DEADLINE )); do
    sleep "$POLL_INTERVAL"

    for slot in a b; do
        if [[ $slot == a ]]; then jobid=$job_a; yaml=$YAML_A; else jobid=$job_b; yaml=$YAML_B; fi
        [[ -z "$jobid" ]] && continue

        state=$(job_state "$jobid")
        [[ -z "$state" ]] && continue   # not finished / not yet in sacct

        case "$state" in
            RUNNING|PENDING|REQUEUED|SUSPENDED|COMPLETING)
                continue
                ;;
            CANCELLED*)
                log "Job $jobid (slot $slot, $yaml) was CANCELLED; not resubmitting automatically"
                if [[ $slot == a ]]; then job_a=""; else job_b=""; fi
                continue
                ;;
            OUT_OF_MEMORY*)
                log "Job $jobid (slot $slot, $yaml) ended with OUT_OF_MEMORY"
                reduce_max_workers "$yaml"
                ;;
            COMPLETED)
                log "Job $jobid (slot $slot, $yaml) COMPLETED (stream finished its run)"
                raise_max_lifetime "$yaml"
                ;;
            *)
                log "Job $jobid (slot $slot, $yaml) ended with state $state"
                ;;
        esac

        newjob=$(submit_job "$yaml")
        log "Resubmitted slot $slot as job $newjob"
        if [[ $slot == a ]]; then job_a=$newjob; else job_b=$newjob; fi
    done
done

log "=== Deadline reached; monitor exiting. Any currently running jobs (job_a=$job_a job_b=$job_b) will continue to completion but will not be auto-resubmitted further. ==="
