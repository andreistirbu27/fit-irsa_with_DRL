#%%

import os
import sys
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

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



#%%

# Plot decoded_mean and 95% confidence interval as a colored area


CONFIDENCE = 0.99

def compute_array_stats(
    all_data, 
    K=1, 
    confidence=0.99, 
    array_key="decoded_array", 
    label="Mean"
):
    """
    Compute means and confidence intervals over batches of data for a specified array key.

    Returns:
        epoch_centers, means, cis_lower, cis_upper
    """
    means = []
    cis_lower = []
    cis_upper = []
    epoch_centers = []

    num_batches = len(all_data) // K
    for i in range(num_batches):
        batch = all_data[i*K:(i+1)*K]
        # Concatenate all arrays in the batch
        batch_array = np.concatenate([np.array(d[array_key]) for d in batch])
        mean = batch_array.mean()
        std = batch_array.std(ddof=1)
        n = len(batch_array)
        # confidence interval
        if n > 1:
            ci = t.ppf(confidence, n-1) * std / np.sqrt(n)
        else:
            ci = 0.0
        means.append(mean)
        cis_lower.append(mean - ci)
        cis_upper.append(mean + ci)
        # Use the center epoch of the batch for x-axis
        batch_epochs = [d.get("epoch", idx) for idx, d in enumerate(batch, start=i*K)]
        epoch_centers.append(np.mean(batch_epochs))

    # Handle any remaining data at the end
    if len(all_data) % K != 0:
        batch = all_data[num_batches*K:]
        batch_array = np.concatenate([np.array(d[array_key]) for d in batch])
        mean = batch_array.mean()
        std = batch_array.std(ddof=1)
        n = len(batch_array)
        if n > 1:
            ci = t.ppf(confidence, n-1) * std / np.sqrt(n)
        else:
            ci = 0.0
        means.append(mean)
        cis_lower.append(mean - ci)
        cis_upper.append(mean + ci)
        batch_epochs = [d.get("epoch", idx) for idx, d in enumerate(batch, start=num_batches*K)]
        epoch_centers.append(np.mean(batch_epochs))

    return epoch_centers, means, cis_lower, cis_upper





#%%

# #print(os.listdir("res-long"))

# dir_name = "res-long/res-1p-u10-s10-e1000-b1000-s1"
# dir_name = "res-long/res-u10-s5-e1000-b1000-s1"
# #dir_name = "res-long/res-load-1p-u20-s20-e1000-b1000-s1"
# #dir_name = "res-long/res-load-u20-s10-e1000-b1000-s1"
# log_file_name = find_jsonl_file(dir_name)
# # all_data = load_jsonl(log_file_name)

# K = 250  # Set the window size for statistics
# ARRAY_KEY = "decoded_array"  # Change this to the desired key if needed
# LABEL = "Mean"  # Change this to a more descriptive label if needed

# epoch_centers, means, cis_lower, cis_upper = compute_array_stats(
#     all_data, K=K, confidence=CONFIDENCE, array_key=ARRAY_KEY, label=LABEL
# )


#%%


def get_load_data(nb_users, nb_slots=20, one_phase=False, seed=1, prefix="-load"):
    if one_phase:
        dir_name = f"res-long/res{prefix}-1p-u{nb_users}-s{nb_slots}-e1000-b1000-s{seed}"
    else:
        dir_name = f"res-long/res{prefix}-u{nb_users}-s{nb_slots//2}-e1000-b1000-s{seed}"
    log_file_name = find_jsonl_file(dir_name)
    all_data = load_jsonl(log_file_name)

    NB_PARTS = 10 # we take statistics on the last 1/NB_PARTS

    K = int(len(all_data)//NB_PARTS)
    assert (len(all_data) % K) == 0
    #K = 250  # Set the window size for statistics
    ARRAY_KEY = "decoded_array"  # Change this to the desired key if needed
    LABEL = "Mean"  # Change this to a more descriptive label if needed

    epoch_centers, means, cis_lower, cis_upper = compute_array_stats(
        all_data, K=K, confidence=CONFIDENCE, array_key=ARRAY_KEY, label=LABEL
    )
    return epoch_centers[-1], means[-1], cis_lower[-1], cis_upper[-1]

import numpy as np
import matplotlib.pyplot as plt

def plot_throughtput_vs_users_slots(prefix="-load", normalize=False, nb_slots=20, xlim=None, ylim=None, fig_file_name=None):
    if nb_slots is not None:
        user_range = range(2, 31)  # Users from 2 to 30
    else:
        user_range = range(2, 31, 2)  # Users from 2 to 30, even number
    seeds = range(1, 5)        # Seeds 1 to 4

    # Store results per seed and per phase
    results = {
        True: {  # one_phase=True
            "avgs": {seed: [] for seed in seeds},
            "lowers": {seed: [] for seed in seeds},
            "uppers": {seed: [] for seed in seeds},
        },
        False: {  # one_phase=False
            "avgs": {seed: [] for seed in seeds},
            "lowers": {seed: [] for seed in seeds},
            "uppers": {seed: [] for seed in seeds},
        }
    }

    for nb_users in user_range:
        for one_phase in [False, True]:
            for seed in seeds:
                try:
                    actual_nb_slots = nb_users if (nb_slots is None) else nb_slots                    
                    _, avg, lower, upper = get_load_data(nb_users, nb_slots=actual_nb_slots, seed=seed, one_phase=one_phase, prefix=prefix)
                    if normalize:
                        avg = avg / nb_users
                        lower = lower / nb_users # XXX: check
                        upper = upper / nb_users
                    results[one_phase]["avgs"][seed].append(avg)
                    results[one_phase]["lowers"][seed].append(lower)
                    results[one_phase]["uppers"][seed].append(upper)
                except Exception as e:
                    print(f"Skipping users={nb_users}, seed={seed}, one_phase={one_phase} due to error: {e}")
                    results[one_phase]["avgs"][seed].append(np.nan)
                    results[one_phase]["lowers"][seed].append(np.nan)
                    results[one_phase]["uppers"][seed].append(np.nan)

    print(results)

    x = np.array(list(user_range))
    bar_width = 0.18
    offsets = np.linspace(-bar_width*1.5, bar_width*1.5, len(seeds))

    plt.figure(figsize=(12,6))
    colors = plt.cm.tab10.colors

    # Plot one_phase=False bars first (behind)
    for i, seed in enumerate(seeds):
        y = np.array(results[False]["avgs"][seed])
        yerr_lower = y - np.array(results[False]["lowers"][seed])
        yerr_upper = np.array(results[False]["uppers"][seed]) - y
        plt.bar(
            x + offsets[i], y, width=bar_width*1.15,  # slightly longer bars
            yerr=[yerr_lower, yerr_upper],
            align='center', alpha=0.4, ecolor='black', capsize=4,
            label=f"Seed {seed} (two-phase)", color=colors[i % len(colors)], zorder=1
        )

    # Plot one_phase=True bars in front
    for i, seed in enumerate(seeds):
        y = np.array(results[True]["avgs"][seed])
        yerr_lower = y - np.array(results[True]["lowers"][seed])
        yerr_upper = np.array(results[True]["uppers"][seed]) - y
        plt.bar(
            x + offsets[i], y, width=bar_width,
            yerr=[yerr_lower, yerr_upper],
            align='center', alpha=0.8, ecolor='black', capsize=4,
            label=f"Seed {seed} (one-phase)", color=colors[i % len(colors)], zorder=2, edgecolor='k'
        )

    plt.xlabel("Number of Users")
    plt.ylabel("Mean Decoded")
    plt.title("Mean Decoded Users vs Number of Users (per seed, with CI)\n(one-phase in front, two-phase behind)")
    plt.xticks(x)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(*xlim)    
    if ylim is not None:
        plt.ylim(*ylim)
        #plt.ylim(10,12)
    if fig_file_name is not None:
        plt.savefig(fig_file_name)
    plt.show()


#%%

plot_throughtput_vs_users_slots(normalize=False, prefix="-load", nb_slots=20, xlim=(3.5,30.5), ylim=(9.5,12), fig_file_name="throughput-vs-users-20slots.pdf")
plot_throughtput_vs_users_slots(normalize=True, prefix="", nb_slots=None, ylim=(0.5,3/4), fig_file_name="throughput-vs-users-and-slots.pdf")


#%%

def get_load_data(nb_users, nb_slots=20, one_phase=False, seed=1, prefix="-load"):
    if one_phase:
        dir_name = f"res-long/res{prefix}-1p-u{nb_users}-s{nb_slots}-e1000-b1000-s{seed}"
    else:
        dir_name = f"res-long/res{prefix}-u{nb_users}-s{nb_slots//2}-e1000-b1000-s{seed}"
    log_file_name = find_jsonl_file(dir_name)
    all_data = load_jsonl(log_file_name)

    NB_PARTS = 10 # we take statistics on the last 1/NB_PARTS

    K = int(len(all_data)//NB_PARTS)
    assert (len(all_data) % K) == 0
    #K = 250  # Set the window size for statistics
    ARRAY_KEY = "decoded_array"  # Change this to the desired key if needed
    LABEL = "Mean"  # Change this to a more descriptive label if needed

    epoch_centers, means, cis_lower, cis_upper = compute_array_stats(
        all_data, K=K, confidence=CONFIDENCE, array_key=ARRAY_KEY, label=LABEL
    )
    return epoch_centers[-1], means[-1], cis_lower[-1], cis_upper[-1], all_data

u1, u2, u3, u4, all_data = get_load_data(8,8, prefix="")
entropy_r1 = [data["entropy_r1_mean"] for data in all_data]
entropy_r1_std = [data["entropy_r1_std_dev"] for data in all_data]
entropy_r2 = [data["entropy_r2_mean"] for data in all_data]
entropy_r2_std = [data["entropy_r2_std_dev"] for data in all_data]

epochs = list(range(len(all_data)))

plt.figure(figsize=(8,4))
plt.plot(epochs, entropy_r1, label="Entropy R1", color='C0')
plt.fill_between(
    epochs,
    [m - s for m, s in zip(entropy_r1, entropy_r1_std)],
    [m + s for m, s in zip(entropy_r1, entropy_r1_std)],
    color='C0', alpha=0.3, label="R1 Mean ± Stddev"
)
plt.plot(epochs, entropy_r2, label="Entropy R2", color='C1')
plt.fill_between(
    epochs,
    [m - s for m, s in zip(entropy_r2, entropy_r2_std)],
    [m + s for m, s in zip(entropy_r2, entropy_r2_std)],
    color='C1', alpha=0.3, label="R2 Mean ± Stddev"
)
plt.xlabel("Epoch")
plt.ylabel("Entropy")
plt.title("Entropy R1 and R2 with Mean ± Stddev")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#get_load_data(20, nb_slots=20, one_phase=False, seed=1, prefix="-load")


#%%

# import matplotlib.pyplot as plt

# # epoch_centers = epoch_centers[-100:]
# # means = means[-100:]
# # cis_lower = cis_lower[-100:]
# # cis_upper = cis_upper[-100:]

# plt.figure(figsize=(8,4))
# plt.plot(epoch_centers, means, '-', label=f"{LABEL}")
# plt.fill_between(epoch_centers, cis_lower, cis_upper, color='C0', alpha=0.3, label="Confidence Interval")
# plt.xlabel("Epoch")
# plt.ylabel(LABEL)
# plt.title(f"{LABEL} with Confidence Interval")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


