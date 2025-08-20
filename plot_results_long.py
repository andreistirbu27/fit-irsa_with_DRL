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

#print(os.listdir("res-long"))

dir_name = "res-long/res-1p-u10-s10-e1000-b1000-s1"
dir_name = "res-long/res-u10-s5-e1000-b1000-s1"
dir_name = "res-long/res-load-1p-u20-s20-e1000-b1000-s1"
#dir_name = "res-long/res-load-u20-s10-e1000-b1000-s1"
log_file_name = find_jsonl_file(dir_name)
all_data = load_jsonl(log_file_name)

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

K = 250  # Set the window size for statistics
ARRAY_KEY = "decoded_array"  # Change this to the desired key if needed
LABEL = "Mean"  # Change this to a more descriptive label if needed

epoch_centers, means, cis_lower, cis_upper = compute_array_stats(
    all_data, K=K, confidence=CONFIDENCE, array_key=ARRAY_KEY, label=LABEL
)


#%%

dir_name = "res-long/res-load-1p-u20-s20-e1000-b1000-s1"
#dir_name = "res-1p-u10-s10-e10000-b1000-seed200"
dir_name = "res-long/res-load-u20-s10-e1000-b1000-s1"
log_file_name = find_jsonl_file(dir_name)
all_data = load_jsonl(log_file_name)

K = 250  # Set the window size for statistics
ARRAY_KEY = "decoded_array"  # Change this to the desired key if needed
LABEL = "Mean"  # Change this to a more descriptive label if needed

epoch_centers, means, cis_lower, cis_upper = compute_array_stats(
    all_data, K=K, confidence=CONFIDENCE, array_key=ARRAY_KEY, label=LABEL
)



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


