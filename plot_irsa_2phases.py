import os
import sys
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt

def find_jsonl_file(run_dir):
    """
    Find a .jsonl or .jsonl.gz file in the given directory.
    Returns the path to the file, or raises FileNotFoundError.
    """
    for fname in os.listdir(run_dir):
        if fname.endswith('.jsonl'):
            return os.path.join(run_dir, fname)
        if fname.endswith('.jsonl.gz'):
            return os.path.join(run_dir, fname)
    raise FileNotFoundError("No .jsonl or .jsonl.gz file found in {}".format(run_dir))

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
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def smooth_sma_nan(x, k=51):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k, dtype=float)
    mask = ~np.isnan(x)
    x_filled = np.nan_to_num(x, nan=0.0)
    num = np.convolve(x_filled, w, mode="same")
    den = np.convolve(mask.astype(float), w, mode="same")
    y = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return y

def main():
    # Default: use current directory or first argument as run dir
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "."

    try:
        jsonl_path = find_jsonl_file(run_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"Loading: {jsonl_path}")
    data = load_jsonl(jsonl_path)

    # Extract metrics
    epochs = [d.get("epoch", i) for i, d in enumerate(data)]
    avg_reward = [d.get("avg_reward", np.nan) for d in data]
    avg_unique = [d.get("avg_unique", np.nan) for d in data]
    frac_decR1_txR2 = [d.get("frac_decR1_txR2", np.nan) for d in data]

    # Smooth
    avg_reward_s = smooth_sma(avg_reward, k=31)
    avg_unique_s = smooth_sma(avg_unique, k=31)
    frac_s = smooth_sma_nan(frac_decR1_txR2, k=51)

    # Plot frac_decR1_txR2
    plt.figure(figsize=(8,4))
    plt.plot(epochs, frac_decR1_txR2, alpha=0.25, linewidth=1, label="Frac (raw)")
    plt.plot(epochs, frac_s, linewidth=2, label="Frac (SMA, k=51)")
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Fraction")
    plt.title("Do R1-decoded users keep transmitting in R2?")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot rewards and unique
    plt.figure(figsize=(8,4))
    plt.plot(epochs, avg_reward, alpha=0.25, linewidth=1, label="Avg Reward (raw)")
    plt.plot(epochs, avg_reward_s, linewidth=2, label="Avg Reward (smoothed)")
    plt.plot(epochs, avg_unique, alpha=0.25, linewidth=1, label="Avg unique decoded (raw)")
    plt.plot(epochs, avg_unique_s, linewidth=2, label="Avg unique decoded (smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress (smoothed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
