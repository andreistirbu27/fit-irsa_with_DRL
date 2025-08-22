import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import json
import sys
from tqdm import tqdm
import shutil
import glob
import math
import time  # <-- Added for timing

try:
    import gzip
except ImportError:
    gzip = None

DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_KEEP_LAST_MODELS = 2
DEFAULT_SEED = 1000

def parse_args():
    parser = argparse.ArgumentParser(description="IRSA 2-Phases Training")
    parser.add_argument('--users', type=int, default=5, help='Number of users')
    parser.add_argument('--slots', type=int, default=3, help='Number of slots')
    parser.add_argument('--torch-single-core', default=False, action="store_true") 
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true', help='Compress log file with gzip')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--poisson', action='store_true')
    parser.add_argument('--log-action', action='store_true', help='Log all actions of all users in each sample of the batch (r1 and r2); implies --log')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS, help='Keep only the last X saved models (default=2)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Random seed (default={DEFAULT_SEED})')
    parser.add_argument('--gpu', action='store_true', help='Use GPU (cuda or mps) if available')
    parser.add_argument('--epoch-half-lr-interval', type=int, default=None, help='If set, halve learning rate every N epochs')
    parser.add_argument('--one-phase', action='store_true', help='Run only Round 1 (Round 2 disabled; equivalent to 0 slots in phase 2)')
    parser.add_argument('--energy-feedback', action='store_true', help='Use scalar energy feedback per-slot occupancy counts replace the undecoded one-hot.')
    parser.add_argument('--transmission-cost', type=float, default=None)

    args = parser.parse_args()
    # --log-action implies --log
    if getattr(args, "log_action", False):
        args.log = True
    return args

def make_result_dir(cfg):
    base = "res"
    if cfg["prefix"] is not None and cfg["prefix"] != "":
        base += "-"+cfg["prefix"]
    if cfg.get('one_phase', False):
        base += "-1p"
    if cfg.get('energy_feedback', False):
        base += "-ef"   # mark runs that use energy feedback

    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
        # Only include non-defaults for hidden-dim, epochs, batch-size, learning-rate
        parts = [
            f"{base}-u{cfg['num_users']}",
            f"s{cfg['num_slots']}"
        ]
        if cfg['poisson']:
            parts.append("poi")
        if cfg['hidden_dim'] != DEFAULT_HIDDEN_DIM:
            parts.append(f"h{cfg['hidden_dim']}")
        if cfg['epochs'] != DEFAULT_EPOCHS:
            parts.append(f"e{cfg['epochs']}")
        if cfg['batch_size'] != DEFAULT_BATCH_SIZE:
            parts.append(f"b{cfg['batch_size']}")
        if cfg['learning_rate'] != DEFAULT_LEARNING_RATE:
            parts.append(f"lr{cfg['learning_rate']}")
        # Only add seed if it is not the default
        if cfg.get('seed', DEFAULT_SEED) != DEFAULT_SEED:
            parts.append(f"seed{cfg['seed']}")
        result_dir = "-".join(parts)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

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

def cleanup_old_models(result_dir, keep_last=DEFAULT_KEEP_LAST_MODELS):
    """
    Keep only the last `keep_last` policy_epoch*.pt files in result_dir.
    Always keep policy_final.pt if present.
    """
    pattern = os.path.join(result_dir, "policy_epoch*.pt")
    files = glob.glob(pattern)
    # Extract epoch numbers
    def extract_epoch(f):
        base = os.path.basename(f)
        try:
            num = int(base.replace("policy_epoch", "").replace(".pt", ""))
            return num
        except Exception:
            return -1
    files_epochs = [(f, extract_epoch(f)) for f in files]
    # Sort by epoch number
    files_epochs = sorted([fe for fe in files_epochs if fe[1] >= 0], key=lambda x: x[1])
    # Keep only the last `keep_last`
    if keep_last > 0 and len(files_epochs) > keep_last:
        to_delete = [f for f, _ in files_epochs[:-keep_last]]
        for f in to_delete:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove old model {f}: {e}", file=sys.stderr)

# === Model loading function ===
def load_model_from_dir(result_dir, which="final", device=None):
    """
    Load a PolicyNetUser model from a directory.

    Args:
        result_dir (str): Directory containing the model and config.json.
        which (str or int): "final" for policy_final.pt, or an epoch number for policy_epoch{epoch}.pt.
        device: torch device to load to (default: None, uses torch default).

    Returns:
        model: PolicyNetUser instance with loaded weights.
        cfg: configuration dictionary loaded from config.json.
    """
    # Load config
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Derived dims
    prev_action_dim = cfg['num_slots']
    feedback_dim = 3 * cfg['num_slots']
    input_dim = cfg['input_obs_dim'] + feedback_dim + prev_action_dim

    # Determine model file
    if which == "final":
        model_path = os.path.join(result_dir, "policy_final.pt")
    elif isinstance(which, int):
        model_path = os.path.join(result_dir, f"policy_epoch{which}.pt")
    elif isinstance(which, str) and which.isdigit():
        model_path = os.path.join(result_dir, f"policy_epoch{which}.pt")
    else:
        raise ValueError(f"Invalid 'which' argument: {which}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Instantiate and load model
    # Determine map_location
    map_loc = device if device is not None else 'cpu'

    model = PolicyNetUser(input_dim, cfg['hidden_dim'], cfg['num_slots'])
    state_dict = torch.load(model_path, map_location=map_loc)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


# === Single-user policy ===
class PolicyNetUser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slots):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots)   # logits per slot for THIS user
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(-1.4)          # optional: start sparser (sigmoid ~0.2)

    def forward(self, x):          # x: [input_dim]
        return self.net(x)         # [num_slots] logits

def sample_actions_user(logits_user):
    """Bernoulli per slot for one user; allows 0,1,2,... slots."""
    d = torch.distributions.Bernoulli(logits=logits_user)
    a = d.sample()  # [num_slots] {0,1}
    lp = d.log_prob(a).sum()  # scalar log-prob for this user
    slots = torch.where(a == 1)[0].tolist()
    with torch.no_grad():
        ent_bits_mean = d.entropy().mean() / math.log(2.0)
    return (len(slots), slots), lp, a, ent_bits_mean  # (r,[slots]), logprob, tensor

# === SIC + feedback (decoded slots are those that became empty after SIC but were non-empty initially) ===
def run_sic_simulation(actions, num_slots, return_feedback_indices=False):
    slots_init = [[] for _ in range(num_slots)]
    for user_id, (r, slot_list) in enumerate(actions):
        for s in slot_list[:r]:
            slots_init[s].append(user_id)
    initial_empty = {i for i in range(num_slots) if len(slots_init[i]) == 0}

    slots = [lst.copy() for lst in slots_init]
    decoded_users = set()
    progress = True
    while progress:
        progress = False
        for s in range(num_slots):
            if len(slots[s]) == 1:
                u = slots[s][0]
                if u not in decoded_users:
                    decoded_users.add(u)
                    for t in range(num_slots):
                        if u in slots[t]:
                            slots[t].remove(u)
                    progress = True

    if return_feedback_indices:
        final_empty = {i for i in range(num_slots) if len(slots[i]) == 0}
        decoded_idx = sorted(list(final_empty - initial_empty))  # became empty due to SIC
        empty_idx = sorted(list(initial_empty))  # empty from start
        undec_idx = sorted([i for i in range(num_slots)
                            if len(slots_init[i]) > 0 and len(slots[i]) > 0])  # still occupied
        energy_per_slot = [len(slots[s]) for s in range(num_slots)]
        return decoded_users, [decoded_idx, empty_idx, undec_idx], energy_per_slot

    return decoded_users

def feedback_indices_to_vector(feedback_indices, num_slots, energy_per_slot=None, use_energy=False):
    decoded_idx, empty_idx, undec_idx = feedback_indices
    d = torch.zeros(num_slots)
    e = torch.zeros(num_slots)
    u = torch.zeros(num_slots)
    if decoded_idx: d[decoded_idx] = 1
    if empty_idx:   e[empty_idx] = 1
    if undec_idx:   u[undec_idx] = 1

    if use_energy and energy_per_slot is not None:
        # per-slot energy (counts), length == num_slots
        u = torch.tensor(energy_per_slot, dtype=d.dtype)

    return torch.cat([d, e, u], dim=0)  # length = 3*num_slots

# --- small local SIC for arbitrary slot count (no feedback needed here) ---
def sic_decode(actions, total_slots):
    # actions: list of (r, [slot_indices in 0..total_slots-1])
    slots = [[] for _ in range(total_slots)]
    for user_id, (r, slot_list) in enumerate(actions):
        for s in slot_list[:r]:
            slots[s].append(user_id)

    decoded_users = set()
    progress = True
    while progress:
        progress = False
        for s in range(total_slots):
            if len(slots[s]) == 1:
                u = slots[s][0]
                if u not in decoded_users:
                    decoded_users.add(u)
                    # cancel user from all slots
                    for t in range(total_slots):
                        if u in slots[t]:
                            slots[t].remove(u)
                    progress = True
    return decoded_users

def round1_actions(policy, obs_all, num_users, input_dim, feedback_dim, prev_action_dim, device):
    actions_r1, acts_bin_r1, entropy_r1 = [], [], []
    lp_r1_total = 0.0
    num_slots = prev_action_dim  # num_slots is equal to prev_action_dim
    for u in range(num_users):
        x1 = torch.cat([
            obs_all[u],
            torch.zeros(feedback_dim, device=device),  # no feedback yet
            torch.zeros(prev_action_dim, device=device)  # no previous action yet
        ], dim=0)
        assert x1.numel() == input_dim
        logits_u = policy(x1)  # [num_slots]
        cw_u, lp_u, a_u, e_u = sample_actions_user(logits_u)  # a_u: [num_slots] in {0,1}
        actions_r1.append(cw_u)
        acts_bin_r1.append(a_u)
        entropy_r1.append(e_u)
        lp_r1_total = lp_r1_total + lp_u
    acts_bin_r1 = torch.stack(acts_bin_r1, dim=0)  # [num_users, num_slots]
    return actions_r1, acts_bin_r1, entropy_r1, lp_r1_total

def round2_actions(policy, obs_all, acts_bin_r1, fb_vec, num_users,  input_dim, device):
    actions_r2, acts_bin_r2, entropy_r2 = [], [], []
    lp_r2_total = 0.0  
    num_slots = acts_bin_r1.shape[1]  # Fix: get number of slots from acts_bin_r1
    for u in range(num_users):
        prev_act_u = acts_bin_r1[u].float()  # THIS user's R1 action (0/1 per slot)
        x2 = torch.cat([
            obs_all[u],
            fb_vec,
            prev_act_u
        ], dim=0)
        assert x2.numel() == input_dim
        logits_u = policy(x2)
        cw_u, lp_u, a_u, e_u = sample_actions_user(logits_u)       
        actions_r2.append(cw_u)
        acts_bin_r2.append(a_u)
        entropy_r2.append(e_u)
        lp_r2_total = lp_r2_total + lp_u
    acts_bin_r2 = torch.stack(acts_bin_r2, dim=0)  # [num_users, num_slots]
    return actions_r2, acts_bin_r2, entropy_r2, lp_r2_total

def compute_reinforce_loss(batch_rewards, batch_log_probs, optimizer):
    baseline = np.mean(batch_rewards)
    total_loss = sum([-(r - baseline) * lp for r, lp in zip(batch_rewards, batch_log_probs)])
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return baseline

def log_epoch(epoch, batch_entropy_r1, batch_entropy_r2, batch_uniques, batch_actions_r1, batch_actions_r2, batch_num_users, log_action, log_file, timing_info=None):
    all_entropy_r1 = torch.cat(batch_entropy_r1)
    all_entropy_r2 = torch.cat(batch_entropy_r2)
    rec = {
        "epoch": epoch, 
        "decoded_array": batch_uniques, 
        "num_users": batch_num_users,
        "entropy_r1_mean": all_entropy_r1.mean().item(),
        "entropy_r2_mean": all_entropy_r2.mean().item(),
        "entropy_r1_std_dev": all_entropy_r1.std().item(),
        "entropy_r2_std_dev": all_entropy_r2.std().item(),
    }
    nb_slots1 = np.array([len(slots) for frame in batch_actions_r1 for slots in frame])
    nb_slots2 = np.array([len(slots) for frame in batch_actions_r2 for slots in frame])
    total_slots = nb_slots1 + nb_slots2

    # Histogram for total slots (R1 + R2)
    unique_total, counts_total = np.unique(total_slots, return_counts=True)
    histogram_total = dict(zip([int(u) for u in unique_total], [int(c) for c in counts_total]))
    rec["total_slots_histogram"] = histogram_total

    # Histogram for R1 only
    unique_r1, counts_r1 = np.unique(nb_slots1, return_counts=True)
    histogram_r1 = dict(zip([int(u) for u in unique_r1], [int(c) for c in counts_r1]))
    rec["r1_slots_histogram"] = histogram_r1

    # Histogram for R2 only
    unique_r2, counts_r2 = np.unique(nb_slots2, return_counts=True)
    histogram_r2 = dict(zip([int(u) for u in unique_r2], [int(c) for c in counts_r2]))
    rec["r2_slots_histogram"] = histogram_r2

    rec["num_users_mean"] = float(np.mean(batch_num_users))
    rec["num_users_std"]  = float(np.std(batch_num_users))

    if log_action:
        rec["actions_r1"] = batch_actions_r1
        rec["actions_r2"] = batch_actions_r2

    # Add timing info if provided
    if timing_info is not None:
        rec.update(timing_info)

    excluded_keys = ["decoded_array", "actions_r1", "actions_r2", "r1_slots_histogram", "r2_slots_histogram", "total_slots_histogram"]
    info = {k: v for k, v in rec.items() if k not in excluded_keys}
    info["decoded_mean"] = np.array(rec["decoded_array"]).mean()

    # Format floats in info to 3 decimal digits
    def format_floats(d):
        if isinstance(d, dict):
            return {k: format_floats(v) for k, v in d.items()}
        elif isinstance(d, float):
            return float(f"{d:.3f}")
        elif isinstance(d, list):
            return [format_floats(x) for x in d]
        else:
            return d
    status = str(format_floats(info))
    tqdm.write(status)
    print(json.dumps(rec), file=log_file, flush=True)

def store_sample_actions(actions_r1, actions_r2, num_users):
    sample_actions_r1 = []
    sample_actions_r2 = []
    for u in range(num_users):
        r1, s1 = actions_r1[u]
        r2, s2 = actions_r2[u]
        sample_actions_r1.append(s1)
        sample_actions_r2.append(s2)
    return sample_actions_r1, sample_actions_r2


# === Training (apply policy per user) ===
def train(policy, optimizer, cfg,
          sparsity_r1_max=0.02, sparsity_r2_max=0.01,
          warmup_r1=400, warmup_r2=800, log_file=None, device=None):
    reward_history = []
    avg_unique_history = []
    frac_decR1_txR2_hist = []  # fraction: R1-decoded who still tx in R2

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']
    feedback_dim = 3 * num_slots
    prev_action_dim = num_slots
    input_dim = input_obs_dim + feedback_dim + prev_action_dim

    log_action = cfg.get('log_action', False)
    one_phase = cfg.get('one_phase', False)
    transmission_cost = cfg["transmission_cost"]

    # Learning rate halving support
    epoch_half_lr_interval = cfg.get('epoch_half_lr_interval', None)
    if epoch_half_lr_interval is not None and epoch_half_lr_interval <= 0:
        epoch_half_lr_interval = None  # ignore non-positive values

    original_num_users = num_users
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        epoch_start_time = time.time()

        # Halve learning rate if needed
        if epoch_half_lr_interval is not None and epoch > 0 and (epoch % epoch_half_lr_interval == 0):
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = old_lr / 2.0
            tqdm.write(f"Learning rate halved at epoch {epoch}. New lr: {optimizer.param_groups[0]['lr']:.6f}")

        # ramped sparsity weights
        lam_r1 = sparsity_r1_max * min(1.0, epoch / float(warmup_r1))
        if epoch <= warmup_r1:
            lam_r2 = 0.0
        else:
            denom = max(1, warmup_r2 - warmup_r1)
            lam_r2 = sparsity_r2_max * min(1.0, (epoch - warmup_r1) / float(denom))

        batch_rewards, batch_log_probs, batch_uniques = [], [], []
        batch_num_users = []
        batch_frac_decR1_txR2 = []
        last_r1_act, last_r2_act = 0.0, 0.0
        batch_entropy_r1 = []
        batch_entropy_r2 = []

        # For --log-action: collect all actions for this batch
        batch_actions_r1 = []
        batch_actions_r2 = []

        # --- Timing: batch episodes generation ---
        batch_gen_start = time.time()


        for _ in range(batch_size):

            num_users = original_num_users
            # If using Poisson arrivals, sample the actual number of users for this batch
            if cfg.get('poisson', False):
                # The mean is num_users, sample from Poisson and ensure at least 1 user
                # Redraw Poisson until > 0 (no bias), warn if >10, error if >1000
                loop_count = 0
                while True:
                    actual_num_users = np.random.poisson(original_num_users)
                    loop_count += 1
                    if actual_num_users > 0:
                        break
                    if loop_count > 1000:
                        raise RuntimeError("Poisson sampling for actual_num_users exceeded 1000 attempts (Pr(0) too high?)")
                    if loop_count > 10 and loop_count % 10 == 0:
                        print(f"Warning: Poisson sampling for actual_num_users took {loop_count} attempts (Pr(0) may be high)", file=sys.stderr)
                num_users = actual_num_users
                actual_num_users = None


            # per-user noise (keep or share, your call)
            obs_all = [torch.rand(input_obs_dim, device=device) for _ in range(num_users)]                

            # -------- Round 1: per-user policy (feedback=0, prev_action=zeros) --------
            actions_r1, acts_bin_r1, entropy_r1, lp_r1_total = round1_actions(
                policy, obs_all, num_users,  input_dim, feedback_dim, prev_action_dim, device
            )

            # feedback from Round 1
            decoded_r1, fb_idx, energy_per_slot = run_sic_simulation(actions_r1, num_slots, return_feedback_indices=True)

            # -------- Round 2 (optional): per-user policy --------
            if not one_phase:
                fb_vec = feedback_indices_to_vector(fb_idx, num_slots,
                    energy_per_slot=energy_per_slot,
                    use_energy=cfg.get('energy_feedback', False)).to(device)  # len = 3*num_slots
                actions_r2, acts_bin_r2, entropy_r2, lp_r2_total = round2_actions(
                    policy, obs_all, acts_bin_r1, fb_vec, num_users,  input_dim, device
                )
            else:
                # Phase 2 disabled: no transmissions in R2
                actions_r2 = [(0, []) for _ in range(num_users)]
                acts_bin_r2 = torch.zeros(num_users, num_slots, device=device)
                entropy_r2 = [torch.tensor(0.0, device=device) for _ in range(num_users)]
                lp_r2_total = torch.tensor(0.0, device=device)

            # -------- CONCATENATE schedules and decode ONCE --------
            # Map Round-2 slots to [num_slots .. 2*num_slots-1], keep R1 as [0 .. num_slots-1]
            actions_concat = []
            for u in range(num_users):
                r1, s1 = actions_r1[u]
                #r2, s2 = actions_r2[u]
                #s2_off = [s + num_slots for s in s2]
                #combined = s1 + s2_off
                r2, s2 = actions_r2[u]
                if one_phase:
                    combined = s1
                else:
                    s2_off = [s + num_slots for s in s2]
                    combined = s1 + s2_off
                actions_concat.append((len(combined), combined))

            total_slots_concat = num_slots if one_phase else (2 * num_slots)
            decoded_concat = sic_decode(actions_concat, total_slots=total_slots_concat)
            num_decoded_concat = len(decoded_concat)
            reward = num_decoded_concat / max(num_users,1)

            # -------------- Ramped sparsity in BOTH rounds  --------------
            r1_activity = acts_bin_r1.float().mean().item()
            r2_activity = acts_bin_r2.float().mean().item() if not one_phase else 0.0
            if transmission_cost is None:
                reward -= lam_r1 * r1_activity
                reward -= lam_r2 * r2_activity
            else:
                reward -= transmission_cost*(r1_activity+r2_activity)*num_slots
            last_r1_act, last_r2_act = r1_activity, r2_activity

            # -------- Metric: frac(R1-decoded who transmit in R2) --------
            if (len(decoded_r1) > 0) and (not one_phase):
                mask_dec = torch.zeros(num_users, dtype=torch.bool, device=acts_bin_r2.device)
                mask_dec[list(decoded_r1)] = True
                r2_any = (acts_bin_r2.sum(dim=1) > 0)  # per-user bool
                frac_decR1_txR2 = r2_any[mask_dec].float().mean().item()
            else:
                frac_decR1_txR2 = float('nan')

            # accumulate
            batch_rewards.append(reward)
            batch_num_users.append(num_users)
            batch_log_probs.append(lp_r1_total + lp_r2_total)
            batch_uniques.append(num_decoded_concat)
            batch_frac_decR1_txR2.append(frac_decR1_txR2)
            batch_entropy_r1.append(torch.stack(entropy_r1))
            batch_entropy_r2.append(torch.stack(entropy_r2))

            # For --log-action: store actions for this sample
            if log_action or True:
                sample_actions_r1, sample_actions_r2 = store_sample_actions(actions_r1, actions_r2, num_users)
                batch_actions_r1.append(sample_actions_r1)
                batch_actions_r2.append(sample_actions_r2)

        batch_gen_end = time.time()
        batch_gen_duration = batch_gen_end - batch_gen_start

        # --- Timing: training phase (REINFORCE update) ---
        train_phase_start = time.time()
        # REINFORCE with baseline
        baseline = compute_reinforce_loss(batch_rewards, batch_log_probs, optimizer)
        train_phase_end = time.time()
        train_phase_duration = train_phase_end - train_phase_start

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # record epoch stats
        reward_history.append(baseline)
        avg_unique_history.append(np.mean(batch_uniques))
        frac_decR1_txR2_hist.append(np.nanmean(batch_frac_decR1_txR2))

        # optional logging
        if log_file is not None:
            timing_info = {
                "epoch_duration_sec": epoch_duration,
                "batch_generation_sec": batch_gen_duration,
                "train_phase_sec": train_phase_duration
            }
            log_epoch(epoch, batch_entropy_r1, batch_entropy_r2, batch_uniques, batch_actions_r1, batch_actions_r2, batch_num_users, log_action, log_file, timing_info=timing_info)

        # after logging/printing per-epoch stats
        if cfg.get('epoch_save_interval') and (epoch + 1) % cfg['epoch_save_interval'] == 0:
            save_model(policy, cfg['result_dir'], epoch=epoch + 1)
            cleanup_old_models(cfg['result_dir'], keep_last=cfg.get('keep_last_models', DEFAULT_KEEP_LAST_MODELS))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: "
                  f"Avg Reward={baseline:.3f}, "
                  f"Avg decoded (concat)={avg_unique_history[-1]:.2f}/~{np.mean(batch_num_users):.1f}, "
                  f"frac(R1-decoded tx in R2)={frac_decR1_txR2_hist[-1]:.3f}, "
                  f"R1 act={last_r1_act:.3f} (λ1={lam_r1:.4f}), "
                  f"R2 act={last_r2_act:.3f} (λ2={lam_r2:.4f}), "
                  f"Epoch time={epoch_duration:.3f}s, Batch gen={batch_gen_duration:.3f}s, Train={train_phase_duration:.3f}s")

    return reward_history, avg_unique_history, frac_decR1_txR2_hist


def main():
    args = parse_args()

    # Enforce --users and --slots (should be handled by argparse 'required', but double-check)
    if args.users is None:
        print("Error: --users is required.", file=sys.stderr)
        sys.exit(1)
    if args.slots is None:
        print("Error: --slots is required.", file=sys.stderr)
        sys.exit(1)

    if args.torch_single_core:
        torch.set_num_threads(1)
        #torch.set_num_interop_threads(1)

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        try:
            import random
            random.seed(args.seed)
        except ImportError:
            pass

    # Device selection
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU.")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS GPU.")
        else:
            device = torch.device("cpu")
            print("GPU requested but not available, using CPU.")
    else:
        device = torch.device("cpu")

    # Put all config in a dictionary
    cfg = {
        'num_users': args.users,
        'num_slots': args.slots,
        'input_obs_dim': args.input_obs_dim,
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'compress': args.compress,
        'epoch_save_interval': args.epoch_save_interval,
        'result_dir': args.result_dir,
        'keep_last_models': args.keep_last_models,
        'seed': args.seed,
        'prefix': args.prefix,
        'poisson': args.poisson,
        'log_action': getattr(args, "log_action", False),
        'epoch_half_lr_interval': getattr(args, "epoch_half_lr_interval", None),
        'one_phase': getattr(args, 'one_phase', False),
        'energy_feedback': getattr(args, 'energy_feedback', False),
        'transmission_cost': args.transmission_cost
    }

    # Derived dims
    prev_action_dim = cfg['num_slots']
    feedback_dim = 3 * cfg['num_slots']
    input_dim = cfg['input_obs_dim'] + feedback_dim + prev_action_dim

    # Make result dir
    result_dir = make_result_dir(cfg)
    cfg['result_dir'] = result_dir

    # Save config for reproducibility
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Open log file
    if args.log:
        log_f, log_path = get_log_file(result_dir, cfg['compress'])
    else:
        log_f, log_path = None, None

    # === Run ===
    policy = PolicyNetUser(input_dim, cfg['hidden_dim'], cfg['num_slots']).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
    try:
        rewards, avg_unique, frac_decR1_txR2 = train(policy, optimizer, cfg, log_file=log_f, device=device)
    finally:
        if log_f is not None:
            log_f.close()

    # Save final model
    save_model(policy, result_dir, epoch=None)

if __name__ == "__main__":
    main()
