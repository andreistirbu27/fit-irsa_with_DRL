#%%

import os
import sys
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt


def get_last_k_logs_and_config(run_dir, k=100):
    """
    Returns:
        logs: list of last k json-parsed log entries (oldest to newest)
        config: dict loaded from config.json
    """
    # Find log file (train_log.jsonl or train_log.jsonl.gz)
    log_path = None
    for fname in os.listdir(run_dir):
        if fname == "train_log.jsonl":
            log_path = os.path.join(run_dir, fname)
            break
        elif fname == "train_log.jsonl.gz":
            log_path = os.path.join(run_dir, fname)
            break
    if log_path is None:
        raise FileNotFoundError("No train_log.jsonl or train_log.jsonl.gz found in {}".format(run_dir))

    # Read last k lines efficiently
    def read_last_k_lines(path, k, gzipped=False):
        lines = []
        if gzipped:
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"
        with open_fn(path, mode) as f:
            # Always read all lines, then take last k (works for both gzip and normal files)
            all_lines = []
            for line in f:
                if line.strip():
                    all_lines.append(line)
            lines = all_lines[-k:]
        return lines

    gzipped = log_path.endswith(".gz")
    log_lines = read_last_k_lines(log_path, k, gzipped=gzipped)
    logs = [json.loads(line) for line in log_lines]

    # Load config.json
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found in {}".format(run_dir))
    with open(config_path, "r") as f:
        config = json.load(f)

    return config, logs

NB_LAST_BATCHES = 20

for nb_users in range(2,11,2):
    log_file_name = f"all-res/res-u{nb_users}-s{nb_users//2}-s1"
    #log_file_name = f"all-res/res-plain-u{nb_users}-s{nb_users}-s1"
    cfg, last_log_lines = get_last_k_logs_and_config(log_file_name, k=NB_LAST_BATCHES)
    #print(last_log_lines)
    all_decoded_array  = np.array([line["decoded_array"] for line in last_log_lines]).flatten()
    print(nb_users, all_decoded_array.mean(), len(all_decoded_array))



