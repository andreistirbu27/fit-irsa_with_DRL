import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Make the repo root importable so `python src/train/foo.py` works as well as `python -m src.train.foo`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.irsa_common.io import get_log_file, save_model, cleanup_old_models
from src.irsa_common.results import under_results
from src.irsa_common.seed import set_seed, set_torch_single_core
from src.irsa_common.sic import (
    feedback_indices_to_vector,
    run_sic_simulation,
    sample_actions_user,
    sic_decode,
)

DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_KEEP_LAST_MODELS = 2
DEFAULT_SEED = 1000

def parse_args():
    parser = argparse.ArgumentParser(description="IRSA 2-Phases Training")
    parser.add_argument('--users', type=int, default=5, help='Number of users (required)')
    parser.add_argument('--slots', type=int, default=3, help='Number of slots (required)')
    parser.add_argument('--torch-single-core', default=False, action="store_true") 
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true', help='Compress log file with gzip')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS, help='Keep only the last X saved models (default=2)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed (default=1)')
    return parser.parse_args()

def make_result_dir(cfg):
    base = "res"
    if cfg["prefix"] is not None and cfg["prefix"] != "":
        base += "-"+cfg["prefix"]
    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
        # Only include non-defaults for hidden-dim, epochs, batch-size, learning-rate
        parts = [
            f"{base}-u{cfg['num_users']}",
            f"s{cfg['num_slots']}"
        ]
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
            parts.append(f"s{cfg['seed']}")
        result_dir = "-".join(parts)
        result_dir = under_results(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

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

    # Define PolicyNetUser class (must match training definition)
    class PolicyNetUser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, cfg['hidden_dim']),
                nn.ReLU(),
                nn.Linear(cfg['hidden_dim'], cfg['num_slots'])
            )
            with torch.no_grad():
                last = self.net[-1]
                if isinstance(last, nn.Linear):
                    last.bias.fill_(-1.4)

        def forward(self, x):
            return self.net(x)

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
    model = PolicyNetUser()
    state_dict = torch.load(model_path, map_location=device)
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

# === Training (apply policy per user) ===
def train(policy, optimizer, cfg,
          sparsity_r1_max=0.02, sparsity_r2_max=0.01,
          warmup_r1=400, warmup_r2=800, log_file=None):
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

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        # ramped sparsity weights
        lam_r1 = sparsity_r1_max * min(1.0, epoch / float(warmup_r1))
        if epoch <= warmup_r1:
            lam_r2 = 0.0
        else:
            denom = max(1, warmup_r2 - warmup_r1)
            lam_r2 = sparsity_r2_max * min(1.0, (epoch - warmup_r1) / float(denom))

        batch_rewards, batch_log_probs, batch_uniques = [], [], []
        batch_frac_decR1_txR2 = []
        last_r1_act, last_r2_act = 0.0, 0.0

        for _ in range(batch_size):
            # per-user noise (keep or share, your call)
            obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

            # -------- Round 1: per-user policy (feedback=0, prev_action=zeros) --------
            actions_r1, acts_bin_r1 = [], []
            lp_r1_total = 0.0
            for u in range(num_users):
                x1 = torch.cat([
                    obs_all[u],
                    torch.zeros(feedback_dim),  # no feedback yet
                    torch.zeros(prev_action_dim)  # no previous action yet
                ], dim=0)
                assert x1.numel() == input_dim
                logits_u = policy(x1)  # [num_slots]
                cw_u, lp_u, a_u = sample_actions_user(logits_u)  # a_u: [num_slots] in {0,1}
                actions_r1.append(cw_u)
                acts_bin_r1.append(a_u)
                lp_r1_total = lp_r1_total + lp_u
            acts_bin_r1 = torch.stack(acts_bin_r1, dim=0)  # [num_users, num_slots]

            # feedback from Round 1
            decoded_r1, fb_idx = run_sic_simulation(actions_r1, num_slots, return_feedback_indices=True)

            # -------- Round 2: per-user policy (feedback + prev_action from R1) --------
            fb_vec = feedback_indices_to_vector(fb_idx, num_slots)  # len = 3*num_slots
            actions_r2, acts_bin_r2 = [], []
            lp_r2_total = 0.0
            for u in range(num_users):
                prev_act_u = acts_bin_r1[u].float()  # THIS user's R1 action (0/1 per slot)
                x2 = torch.cat([
                    obs_all[u],
                    fb_vec,
                    prev_act_u
                ], dim=0)
                assert x2.numel() == input_dim
                logits_u = policy(x2)
                cw_u, lp_u, a_u = sample_actions_user(logits_u)
                actions_r2.append(cw_u)
                acts_bin_r2.append(a_u)
                lp_r2_total = lp_r2_total + lp_u
            acts_bin_r2 = torch.stack(acts_bin_r2, dim=0)  # [num_users, num_slots]

            # -------- CONCATENATE schedules and decode ONCE --------
            # Map Round-2 slots to [num_slots .. 2*num_slots-1], keep R1 as [0 .. num_slots-1]
            actions_concat = []
            for u in range(num_users):
                r1, s1 = actions_r1[u]
                r2, s2 = actions_r2[u]
                s2_off = [s + num_slots for s in s2]
                combined = s1 + s2_off
                actions_concat.append((len(combined), combined))

            decoded_concat = sic_decode(actions_concat, total_slots=2 * num_slots)
            num_decoded_concat = len(decoded_concat)
            reward = num_decoded_concat / num_users

            # -------------- Ramped sparsity in BOTH rounds  --------------
            r1_activity = acts_bin_r1.float().mean().item()
            r2_activity = acts_bin_r2.float().mean().item()
            reward -= lam_r1 * r1_activity
            reward -= lam_r2 * r2_activity
            last_r1_act, last_r2_act = r1_activity, r2_activity

            # -------- Metric: frac(R1-decoded who transmit in R2) --------
            if len(decoded_r1) > 0:
                mask_dec = torch.zeros(num_users, dtype=torch.bool)
                mask_dec[list(decoded_r1)] = True
                r2_any = (acts_bin_r2.sum(dim=1) > 0)  # per-user bool
                frac_decR1_txR2 = r2_any[mask_dec].float().mean().item()
            else:
                frac_decR1_txR2 = float('nan')

            # accumulate
            batch_rewards.append(reward)
            batch_log_probs.append(lp_r1_total + lp_r2_total)
            batch_uniques.append(num_decoded_concat)
            batch_frac_decR1_txR2.append(frac_decR1_txR2)

        # REINFORCE with baseline
        baseline = np.mean(batch_rewards)
        total_loss = sum([-(r - baseline) * lp for r, lp in zip(batch_rewards, batch_log_probs)])
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # record epoch stats
        reward_history.append(baseline)
        avg_unique_history.append(np.mean(batch_uniques))
        frac_decR1_txR2_hist.append(np.nanmean(batch_frac_decR1_txR2))

        # optional logging
        if log_file is not None:
            rec = {"epoch": epoch, "decoded_array": batch_uniques}
            print(json.dumps(rec), file=log_file, flush=True)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: "
                  f"Avg Reward={baseline:.3f}, "
                  f"Avg decoded (concat)={avg_unique_history[-1]:.2f}/{num_users}, "
                  f"frac(R1-decoded tx in R2)={frac_decR1_txR2_hist[-1]:.3f}, "
                  f"R1 act={last_r1_act:.3f} (λ1={lam_r1:.4f}), "
                  f"R2 act={last_r2_act:.3f} (λ2={lam_r2:.4f})")

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
        set_torch_single_core()

    set_seed(args.seed)

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
        'prefix': args.prefix
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
    policy = PolicyNetUser(input_dim, cfg['hidden_dim'], cfg['num_slots'])
    optimizer = optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
    try:
        rewards, avg_unique, frac_decR1_txR2 = train(policy, optimizer, cfg, log_file=log_f)
    finally:
        if log_f is not None:
            log_f.close()

    # Save final model
    save_model(policy, result_dir, epoch=None)

if __name__ == "__main__":
    main()
