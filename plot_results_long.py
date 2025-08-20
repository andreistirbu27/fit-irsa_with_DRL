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


def get_load_data(nb_users, nb_slots=20, one_phase=False, seed=1):
    if one_phase:
        dir_name = f"res-long/res-load-1p-u{nb_users}-s{nb_slots}-e1000-b1000-s{seed}"
    else:
        dir_name = f"res-long/res-load-u{nb_users}-s{nb_slots//2}-e1000-b1000-s{seed}"
    log_file_name = find_jsonl_file(dir_name)
    all_data = load_jsonl(log_file_name)

    NB_PARTS = 4 # we take statistics on the last 1/NB_PARTS

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

user_range = range(2, 31)  # Users from 2 to 30
seeds = range(1, 5)        # Seeds 1 to 4

# Store results per seed
decoded_avgs_per_seed = {seed: [] for seed in seeds}
decoded_lowers_per_seed = {seed: [] for seed in seeds}
decoded_uppers_per_seed = {seed: [] for seed in seeds}

for nb_users in user_range:
    for seed in seeds:
        try:
            _, avg, lower, upper = get_load_data(nb_users, seed=seed, one_phase=True)
            decoded_avgs_per_seed[seed].append(avg)
            decoded_lowers_per_seed[seed].append(lower)
            decoded_uppers_per_seed[seed].append(upper)
        except Exception as e:
            print(f"Skipping users={nb_users}, seed={seed} due to error: {e}")
            decoded_avgs_per_seed[seed].append(np.nan)
            decoded_lowers_per_seed[seed].append(np.nan)
            decoded_uppers_per_seed[seed].append(np.nan)

x = np.array(list(user_range))
bar_width = 0.18
offsets = np.linspace(-bar_width*1.5, bar_width*1.5, len(seeds))

plt.figure(figsize=(12,6))
colors = plt.cm.tab10.colors

for i, seed in enumerate(seeds):
    y = np.array(decoded_avgs_per_seed[seed])
    yerr_lower = y - np.array(decoded_lowers_per_seed[seed])
    yerr_upper = np.array(decoded_uppers_per_seed[seed]) - y
    yerr = np.vstack([yerr_lower, yerr_upper])
    plt.bar(x + offsets[i], y, width=bar_width, yerr=[yerr_lower, yerr_upper], 
            align='center', alpha=0.7, ecolor='black', capsize=4, 
            label=f"Seed {seed}", color=colors[i % len(colors)])

plt.xlabel("Number of Users")
plt.ylabel("Mean Decoded")
plt.title("Mean Decoded Users vs Number of Users (per seed, with CI)")
plt.xticks(x)
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()



#%%

import matplotlib.pyplot as plt

# epoch_centers = epoch_centers[-100:]
# means = means[-100:]
# cis_lower = cis_lower[-100:]
# cis_upper = cis_upper[-100:]

plt.figure(figsize=(8,4))
plt.plot(epoch_centers, means, '-', label=f"{LABEL}")
plt.fill_between(epoch_centers, cis_lower, cis_upper, color='C0', alpha=0.3, label="Confidence Interval")
plt.xlabel("Epoch")
plt.ylabel(LABEL)
plt.title(f"{LABEL} with Confidence Interval")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


