import os
import sys
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt


def find_jsonl_file(run_dir):
    gz = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.jsonl.gz')), None)
    if gz is not None:
        return gz
    plain = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.jsonl')), None)
    if plain is not None:
        return plain
    raise FileNotFoundError(f"No .jsonl or .jsonl.gz file found in {run_dir}")


def load_jsonl(path):
    open_fn = gzip.open if path.endswith('.gz') else open
    mode = 'rt' if path.endswith('.gz') else 'r'
    with open_fn(path, mode) as f:
        return [json.loads(line) for line in f if line.strip()]


def smooth_sma(x, k=31):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


def detect_k(records):
    k = 0
    for d in records:
        for key in d.keys():
            if key.startswith("activity_phase"):
                idx = int(key[len("activity_phase"):])
                if idx + 1 > k:
                    k = idx + 1
    return k


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        jsonl_path = find_jsonl_file(run_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"Loading: {jsonl_path}")
    data = load_jsonl(jsonl_path)
    if not data:
        print("Empty log.")
        sys.exit(1)

    k = detect_k(data)
    if k == 0:
        print("No activity_phase* keys found — is this a k-phase run?")
        sys.exit(1)
    print(f"Detected k={k} phases")

    epochs = [d.get("epoch", i) for i, d in enumerate(data)]
    avg_reward = [d.get("avg_reward", np.nan) for d in data]
    avg_unique = [d.get("avg_unique", np.nan) for d in data]
    activities = [[d.get(f"activity_phase{i}", np.nan) for d in data] for i in range(k)]
    lambdas = [[d.get(f"lambda_phase{i}", np.nan) for d in data] for i in range(k)]

    avg_reward_s = smooth_sma(avg_reward, k=31)
    avg_unique_s = smooth_sma(avg_unique, k=31)
    activities_s = [smooth_sma(a, k=31) for a in activities]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(epochs, avg_reward, alpha=0.25, linewidth=1, label="Avg Reward (raw)")
    ax.plot(epochs, avg_reward_s, linewidth=2, label="Avg Reward (smoothed)")
    ax.plot(epochs, avg_unique, alpha=0.25, linewidth=1, label="Avg unique decoded (raw)")
    ax.plot(epochs, avg_unique_s, linewidth=2, label="Avg unique decoded (smoothed)")
    ax.set_ylabel("Value")
    ax.set_title(f"Training Progress (k={k} phases)")
    ax.grid(True)
    ax.legend(loc="lower right")

    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i in range(k):
        color = cmap(i)
        ax.plot(epochs, activities_s[i], color=color, linewidth=2, label=f"activity P{i}")
        ax.plot(epochs, lambdas[i], color=color, linestyle="--", linewidth=1.2, alpha=0.7, label=f"λ P{i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Activity / λ")
    ax.set_title("Per-phase activity (smoothed) and sparsity λ schedule")
    ax.grid(True)
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
