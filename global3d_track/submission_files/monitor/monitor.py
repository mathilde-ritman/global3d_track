#!/usr/bin/env python3
"""
Monitors the two grl_stats DCC-statistics job streams and keeps exactly two
streams active until all DCCs are processed:

- job ended in SLURM state OUT_OF_MEMORY (or FAILED with an OOM signature in
  its log)         -> decrement max_workers in that stream's yaml, resubmit
- job TIMEOUT                                  -> resubmit the same yaml unchanged
                                                   (per-DCC results are already
                                                   saved to disk, so the next
                                                   run picks up where it left off)
- job COMPLETED with no DCCs left to process    -> retire this stream; duplicate
                                                   the *other* active stream's
                                                   yaml with start_index moved to
                                                   the halfway point of its
                                                   remaining DCCs, and submit that
                                                   as a new stream in this slot
- job COMPLETED but some DCCs were permanently
  given up on (non-OOM per-DCC errors)          -> resubmit the same yaml unchanged
                                                   (same reasoning as TIMEOUT)
- anything else (FAILED without OOM signature,
  CANCELLED by someone else, NODE_FAIL, ...)    -> resubmit unchanged, but halt
                                                   that stream after 3 consecutive
                                                   unrecognised outcomes so a real
                                                   bug can't loop forever
"""

import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

BASE = Path(__file__).resolve().parent
STATE_FILE = BASE / "state.json"
LOG_FILE = BASE / "monitor.log"
STOP_FILE = BASE / "STOP"
SPLIT_DIR = BASE / "splits"
SUBMIT_SCRIPT = "/home/b/b382635/s/global3d_track/global3d_track/submission_files/_statistics_itr_grl.sh"
MONITOR_SUBMIT_SCRIPT = BASE / "monitor_job.sh"
JOB_LOG_DIR = Path("/home/b/b382635/job_outfiles/tracking/global3d_track/stats")

POLL_INTERVAL = 120  # seconds
MIN_WORKERS = 1
MAX_CONSECUTIVE_ANOMALIES = 3

# The monitor itself runs as a 7-day SLURM job (see monitor_job.sh) rather than
# a login-node background process, because login nodes silently reap
# long-lived stray processes (observed after ~18h in practice). Before its own
# walltime runs out, it resubmits itself and exits cleanly so monitoring never
# has a gap.
SELF_RESUBMIT_MARGIN_SECONDS = 1 * 3600  # resubmit with 1h of walltime left
MONITOR_JOB_TIME_LIMIT_SECONDS = int(11.5 * 3600)  # must match monitor_job.sh's #SBATCH --time
START_TIME = time.time()


def log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state():
    return json.loads(STATE_FILE.read_text())


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def sacct_state(job_id):
    out = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=JobID,State", "-n", "-P"],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 2 and parts[0] == str(job_id):
            return parts[1].strip()
    return None


def job_log_path(job_id):
    return JOB_LOG_DIR / f"job.o{job_id}"


def log_contains(job_id, *patterns):
    path = job_log_path(job_id)
    if not path.exists():
        return False
    text = path.read_text(errors="ignore")
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def submit_job(yaml_path):
    out = subprocess.run(
        ["sbatch", "--parsable", SUBMIT_SCRIPT, str(yaml_path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed for {yaml_path}: {out.stderr.strip()}")
    job_id = out.stdout.strip()
    log(f"  -> submitted job {job_id} for {yaml_path}")
    return job_id


def read_max_workers(yaml_path):
    di = yaml.safe_load(Path(yaml_path).read_text())
    return int(di.get("max_workers", 0))


def decrement_max_workers(yaml_path):
    path = Path(yaml_path)
    text = path.read_text()
    ends_with_newline = text.endswith("\n")
    lines = text.splitlines()
    new_val = None
    for i, line in enumerate(lines):
        m = re.match(r"^(max_workers:\s*)(\d+)(.*)$", line)
        if m:
            old_val = int(m.group(2))
            new_val = max(MIN_WORKERS, old_val - 1)
            lines[i] = f"{m.group(1)}{new_val}{m.group(3)}"
            break
    if new_val is None:
        raise RuntimeError(f"could not find max_workers line in {yaml_path}")
    path.write_text("\n".join(lines) + ("\n" if ends_with_newline else ""))
    return old_val, new_val


def set_start_index(yaml_path, new_index):
    path = Path(yaml_path)
    text = path.read_text()
    ends_with_newline = text.endswith("\n")
    lines = text.splitlines()
    found = False
    for i, line in enumerate(lines):
        m = re.match(r"^(start_index:\s*)(\d+)(.*)$", line)
        if m:
            lines[i] = f"{m.group(1)}{new_index}{m.group(3)}"
            found = True
            break
    if not found:
        raise RuntimeError(f"could not find start_index line in {yaml_path}")
    path.write_text("\n".join(lines) + ("\n" if ends_with_newline else ""))


def compute_remaining(yaml_path):
    """Replicates the filtering logic in get_stats_grl.py's main() to work out
    exactly how many DCCs a fresh run of this yaml would report as remaining,
    and the start_index currently in effect."""
    di = yaml.safe_load(Path(yaml_path).read_text())
    start_date = datetime.strptime(di["start_date"], "%Y-%m-%d %H:%M:%S")

    version = Path(di["version_name"], di["region"])
    data_dir = Path(di["data_directory"]) / version
    results_dir = Path(di["results_directory"]) / di["folder"]

    df = pd.read_csv(data_dir / "track_stats/dcc_statistics.csv", index_col="dcc")

    validity = di.get("validity", "valid")
    max_lifetime = di.get("max_lifetime", 100)
    max_size = di.get("max_size", 1e5)
    valid_df = df[df[validity]] if validity != "invalid" else df[~df.invalid]
    is_okay = (valid_df.lifetime < max_lifetime) & (valid_df.npixels < max_size)
    is_okay = is_okay & (pd.to_datetime(valid_df.initial_time) > start_date)
    dcc_list = valid_df[is_okay].index.values

    overwrite = di.get("overwrite", 0)
    is_complete = [] if overwrite else [
        int(f.stem.removeprefix("dcc_")) for f in results_dir.glob("dcc_*.nc")
    ]
    dcc_list = [x for x in dcc_list if x not in is_complete]

    start_idx = di.get("start_index", 0)
    remaining = dcc_list[start_idx:]
    return len(remaining), start_idx


def classify(job_id):
    """Returns one of: 'running', 'oom', 'timeout', 'complete', 'partial_complete', 'anomaly'."""
    state = sacct_state(job_id)
    if state is None:
        return "running", state  # accounting lag, treat as still going

    if state in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "RESIZING"):
        return "running", state

    if state == "OUT_OF_MEMORY":
        return "oom", state

    if state == "TIMEOUT":
        return "timeout", state

    if state == "FAILED" and log_contains(job_id, r"memoryerror", r"oom", r"out of memory"):
        return "oom", state

    if state == "COMPLETED":
        if log_contains(job_id, r"were never successfully processed", r"Exceeded \d+ pool restarts"):
            return "partial_complete", state
        return "complete", state

    return "anomaly", state


def do_split(finished_name, state):
    """Retire the finished stream's slot and split off half of whichever other
    stream is still active into a new stream taking its place."""
    others = [n for n in state["streams"] if n != finished_name]
    active_others = [n for n in others if state["streams"][n].get("status", "active") == "active"]

    if not active_others:
        log(f"  -> no other active stream to split; nothing left to do for this slot.")
        state["streams"][finished_name]["status"] = "retired_no_partner"
        return

    partner_name = active_others[0]
    partner_yaml = state["streams"][partner_name]["yaml"]

    remaining, start_idx = compute_remaining(partner_yaml)
    log(f"  -> partner stream {partner_name} ({partner_yaml}) currently has {remaining} DCCs remaining (start_index={start_idx})")

    if remaining < 2:
        log(f"  -> too few DCCs remaining in {partner_name} to split further; retiring this slot, continuing with a single active stream.")
        state["streams"][finished_name]["status"] = "retired_no_split_needed"
        return

    half = remaining // 2
    new_start_index = start_idx + half

    state["split_counter"] += 1
    n = state["split_counter"]
    new_yaml = SPLIT_DIR / f"{Path(partner_yaml).stem}_split{n}.yaml"
    new_yaml.write_text(Path(partner_yaml).read_text())
    set_start_index(new_yaml, new_start_index)

    log(f"  -> splitting {partner_name}: new stream split{n} = copy of {partner_yaml} with start_index {start_idx} -> {new_start_index}")
    new_job_id = submit_job(new_yaml)

    new_name = f"split{n}"
    state["streams"][finished_name] = {
        "yaml": str(new_yaml),
        "job_id": new_job_id,
        "status": "active",
        "consecutive_anomalies": 0,
        "history": [f"born from split of {partner_name} after {finished_name} fully completed"],
    }
    # rename key: we overwrote finished_name's slot in place with the new stream
    state["streams"][new_name] = state["streams"].pop(finished_name)


def handle_stream(name, state):
    entry = state["streams"][name]
    if entry.get("status", "active") != "active":
        return

    job_id = entry["job_id"]
    outcome, sacct_st = classify(job_id)

    if outcome == "running":
        return

    log(f"Stream {name}: job {job_id} finished (sacct state={sacct_st}) -> classified as '{outcome}'")
    entry.setdefault("history", []).append(f"job {job_id}: sacct={sacct_st} outcome={outcome}")

    if outcome == "oom":
        old_w, new_w = decrement_max_workers(entry["yaml"])
        log(f"  -> OOM: reduced max_workers {old_w} -> {new_w} in {entry['yaml']}")
        new_job_id = submit_job(entry["yaml"])
        entry["job_id"] = new_job_id
        entry["consecutive_anomalies"] = 0

    elif outcome == "timeout":
        log(f"  -> TIMEOUT: resubmitting {entry['yaml']} unchanged")
        new_job_id = submit_job(entry["yaml"])
        entry["job_id"] = new_job_id
        entry["consecutive_anomalies"] = 0

    elif outcome == "partial_complete":
        log(f"  -> COMPLETED but some DCCs were permanently given up on (non-OOM per-DCC errors): resubmitting {entry['yaml']} unchanged")
        new_job_id = submit_job(entry["yaml"])
        entry["job_id"] = new_job_id
        entry["consecutive_anomalies"] = 0

    elif outcome == "complete":
        log(f"  -> COMPLETED: all DCCs in this stream are done. Retiring this slot and splitting the other active stream.")
        do_split(name, state)

    elif outcome == "anomaly":
        n = entry.get("consecutive_anomalies", 0) + 1
        entry["consecutive_anomalies"] = n
        if n >= MAX_CONSECUTIVE_ANOMALIES:
            log(f"  -> ALERT: {n} consecutive unrecognised outcomes for stream {name} (last sacct state={sacct_st}). "
                f"Halting automatic resubmission for this stream - needs manual review.")
            entry["status"] = "halted_needs_attention"
        else:
            log(f"  -> unrecognised sacct state '{sacct_st}' ({n}/{MAX_CONSECUTIVE_ANOMALIES}); resubmitting {entry['yaml']} unchanged as a safe default")
            new_job_id = submit_job(entry["yaml"])
            entry["job_id"] = new_job_id


def all_done(state):
    for name, entry in state["streams"].items():
        if entry.get("status", "active") == "active":
            return False
    return True


def resubmit_self():
    out = subprocess.run(
        ["sbatch", "--parsable", str(MONITOR_SUBMIT_SCRIPT)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed for monitor self-resubmit: {out.stderr.strip()}")
    return out.stdout.strip()


def main():
    log(f"=== monitor starting (pid {os.getpid()}, SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}) ===")
    state = load_state()
    save_state(state)

    while True:
        if STOP_FILE.exists():
            log("STOP file found - exiting monitor loop.")
            break

        state = load_state()
        for name in list(state["streams"].keys()):
            try:
                handle_stream(name, state)
            except Exception:
                log(f"ERROR while handling stream {name}:\n{traceback.format_exc()}")
        save_state(state)

        if all_done(state):
            log("All streams retired / no active streams remain. Exiting monitor loop.")
            break

        if time.time() - START_TIME >= MONITOR_JOB_TIME_LIMIT_SECONDS - SELF_RESUBMIT_MARGIN_SECONDS:
            log("Approaching this monitor job's own walltime limit - resubmitting self before exiting.")
            try:
                new_job_id = resubmit_self()
                log(f"  -> monitor resubmitted as job {new_job_id}")
            except Exception:
                log(f"ERROR resubmitting monitor itself:\n{traceback.format_exc()}")
            break

        time.sleep(POLL_INTERVAL)

    log("=== monitor stopped ===")


if __name__ == "__main__":
    main()
