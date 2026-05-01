import os
import sys
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt

def find_jsonl_file(run_dir):
    """
    Find a .jsonl or .jsonl.gz file in the given directory.
    Prefers .jsonl.gz over .jsonl when both exist.
    Returns the path to the file, or raises FileNotFoundError.
    """
    gz = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.jsonl.gz')), None)
    if gz is not None:
        return gz
    plain = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.jsonl')), None)
    if plain is not None:
        return plain
    raise FileNotFoundError(f"No .jsonl or .jsonl.gz file found in {run_dir}")

def load_jsonl(path):
    """
    Load a jsonl or jsonl.gz file and return a list of dicts.
    """
    if path.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'
    with open_fn(path, mode) as f:
        return [json.loads(line) for line in f if line.strip()]

def smooth_sma(x, k=31):
    # mode="same" pads with zeros at boundaries; first/last k//2 samples droop toward zero.
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def main():
    # Default: use current directory or first argument as run dir
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    try:
        jsonl_path = find_jsonl_file(run_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"Loading: {jsonl_path}")
    data = load_jsonl(jsonl_path)

    # Expected keys from the 1-round trainer (uncomment logging in the trainer):
    # rec = {"epoch": epoch, "reward": float(baseline),
    #        "avg_decoded": float(avg_decoded_history[-1]),
    #        "activity": last_activity, "lambda": lam}
    epochs       = [d.get("epoch", i) for i, d in enumerate(data)]
    avg_reward   = [d.get("reward", d.get("avg_reward", np.nan)) for d in data]
    avg_decoded  = [d.get("avg_decoded", np.nan) for d in data]
    activity     = [d.get("activity", np.nan) for d in data]     # optional
    lam_series   = [d.get("lambda",  np.nan) for d in data]      # optional

    # Smooth
    avg_reward_s  = smooth_sma(avg_reward,  k=31)
    avg_decoded_s = smooth_sma(avg_decoded, k=31)
    activity_s    = smooth_sma(activity,    k=31)  # may be NaN if not logged

    # Plot rewards
    plt.figure(figsize=(8,4))
    plt.plot(epochs, avg_reward,  alpha=0.25, linewidth=1, label="Avg Reward (raw)")
    plt.plot(epochs, avg_reward_s,            linewidth=2, label="Avg Reward (smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Training Reward (1 round, no feedback)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot decoded users (+ optional activity on a secondary axis if present)
    plt.figure(figsize=(8,4))
    plt.plot(epochs, avg_decoded,  alpha=0.25, linewidth=1, label="Avg decoded (raw)")
    plt.plot(epochs, avg_decoded_s,            linewidth=2, label="Avg decoded (smoothed)")
    # If activity is available, overlay on right axis
    if not np.all(np.isnan(activity_s)):
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(epochs, activity_s, linestyle="--", linewidth=1.5, label="Activity (smoothed)")
        ax2.set_ylabel("Mean activity")
        # Build a joint legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")
    else:
        plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Users decoded")
    plt.title("Decoded Users (1 round, no feedback)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()