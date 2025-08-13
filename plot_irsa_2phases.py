
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



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

frac_s = smooth_sma_nan(frac_decR1_txR2, k=51)  # tune k


plt.figure(figsize=(8,4))
plt.plot(frac_decR1_txR2, alpha=0.25, linewidth=1, label="Frac (raw)")
plt.plot(frac_s,              linewidth=2, label="Frac (SMA, k=51)")
plt.ylim(0, 1)
plt.xlabel("Epoch"); plt.ylabel("Fraction")
plt.title("Do R1-decoded users keep transmitting in R2?")
plt.grid(True); plt.legend(); plt.show()


#plt.figure(figsize=(8,4))
#plt.plot(frac_decR1_txR2, label="Frac: R1-decoded who transmit in R2")
#plt.xlabel("Epoch"); plt.ylabel("Fraction")
#plt.title("Do R1-decoded users keep transmitting in R2?")
#plt.grid(True); plt.legend(); plt.show()

def smooth_sma(x, k=31):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

# smooth
rewards_s = smooth_sma(rewards, k=31)
unique_s  = smooth_sma(avg_unique, k=31)

# plot (show raw faint + smoothed bold)
plt.figure(figsize=(8,4))
plt.plot(rewards,    alpha=0.25, linewidth=1, label="Avg Reward (raw)")
plt.plot(rewards_s,  linewidth=2,             label="Avg Reward (smoothed)")
plt.plot(avg_unique, alpha=0.25, linewidth=1, label="Avg unique decoded (raw)")
plt.plot(unique_s,   linewidth=2,             label="Avg unique decoded (smoothed)")
plt.xlabel("Epoch"); plt.ylabel("Value")
plt.title("Training Progress (smoothed)")
plt.grid(True); plt.legend(); plt.show()

#plt.figure(figsize=(8,4))
#plt.plot(rewards, label="Avg Reward")
#plt.plot(avg_unique, label="Avg unique decoded")
#plt.xlabel("Epoch"); plt.ylabel("Value")
#plt.title("Training Progress")
#plt.grid(True); plt.legend(); plt.show()

