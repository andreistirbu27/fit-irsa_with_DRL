"""Logging and model checkpoint helpers shared across training scripts."""
import glob
import os
import sys

import torch

try:
    import gzip
except ImportError:
    gzip = None


def get_log_file(result_dir, compress):
    log_path = os.path.join(result_dir, "train_log.jsonl" + (".gz" if compress else ""))
    if compress:
        if gzip is None:
            raise RuntimeError("gzip module not available for compression")
        f = gzip.open(log_path, "at")
    else:
        f = open(log_path, "a")
    return f, log_path


def save_model(policy, result_dir, epoch=None):
    if epoch is None:
        fname = os.path.join(result_dir, "policy_final.pt")
    else:
        fname = os.path.join(result_dir, f"policy_epoch{epoch}.pt")
    torch.save(policy.state_dict(), fname)


def cleanup_old_models(result_dir, keep_last=2):
    """Keep only the last `keep_last` policy_epoch*.pt files. policy_final.pt is untouched."""
    pattern = os.path.join(result_dir, "policy_epoch*.pt")
    files = glob.glob(pattern)

    def extract_epoch(f):
        base = os.path.basename(f)
        try:
            return int(base.replace("policy_epoch", "").replace(".pt", ""))
        except Exception:
            return -1

    files_epochs = [(f, extract_epoch(f)) for f in files]
    files_epochs = sorted([fe for fe in files_epochs if fe[1] >= 0], key=lambda x: x[1])
    if keep_last > 0 and len(files_epochs) > keep_last:
        for f, _ in files_epochs[:-keep_last]:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove old model {f}: {e}", file=sys.stderr)
