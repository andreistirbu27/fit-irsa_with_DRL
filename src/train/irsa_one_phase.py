import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.irsa_common.io import get_log_file, save_model, cleanup_old_models
from src.irsa_common.results import under_results
from src.irsa_common.seed import set_seed, set_torch_single_core
from src.irsa_common.sic import run_sic_simulation, sample_actions_user

DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_KEEP_LAST_MODELS = 2

def parse_args():
    parser = argparse.ArgumentParser(description="IRSA Single-Round Training")
    parser.add_argument('--users', type=int, required=True, help='Number of users')
    parser.add_argument('--slots', type=int, required=True, help='Number of slots')
    parser.add_argument('--torch-single-core', default=False, action="store_true")
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true', help='Compress log file with gzip')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS, help='Keep only the last X saved models (default=2)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    return parser.parse_args()


def make_result_dir(cfg):
    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
        parts = [
            f"res-plain-u{cfg['num_users']}",
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
        parts.append(f"seed{cfg['seed']}")
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
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Derived dims for single-round: input is only random noise
    input_dim = cfg['input_obs_dim']

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

    model = PolicyNetUser()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


def main():
    args = parse_args()

    if args.users is None:
        print("Error: --users is required.", file=sys.stderr)
        sys.exit(1)
    if args.slots is None:
        print("Error: --slots is required.", file=sys.stderr)
        sys.exit(1)

    if args.torch_single_core:
        set_torch_single_core()

    set_seed(args.seed)

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
    }

    # --- Derived dims (single round: input is only random noise) ---
    input_dim = cfg['input_obs_dim']

    result_dir = make_result_dir(cfg)
    cfg['result_dir'] = result_dir

    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log_f, log_path = get_log_file(result_dir, cfg['compress'])

    # === Single-user policy ===
    class PolicyNetUser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, cfg['hidden_dim']),
                nn.ReLU(),
                nn.Linear(cfg['hidden_dim'], cfg['num_slots'])   # logits per slot for THIS user
            )
            with torch.no_grad():
                last = self.net[-1]
                if isinstance(last, nn.Linear):
                    last.bias.fill_(-1.4)  # optional: start sparser (sigmoid ~0.2)

        def forward(self, x):          # x: [input_dim] = [input_obs_dim]
            return self.net(x)         # [num_slots] logits

    # ---- expose config values as locals for train() closure ----
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']

    # === Training (single round, no feedback) ===
    def train(policy, optimizer, sparsity_max=0.02, warmup=400):
        reward_history = []
        avg_decoded_history = []
        dummy_hist = []  # keep third return value to match prior call signature

        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            lam = sparsity_max * min(1.0, epoch / float(warmup))

            batch_rewards, batch_log_probs, batch_decoded = [], [], []
            last_activity = 0.0

            for _ in range(batch_size):
                # per-user random noise
                obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

                # ----- Single round: per-user policy on noise only -----
                actions, acts_bin = [], []
                lp_total = 0.0
                for u in range(num_users):
                    x = obs_all[u]                       # shape = [input_obs_dim]
                    assert x.numel() == input_dim
                    logits_u = policy(x)                 # [num_slots] logits
                    cw_u, lp_u, a_u = sample_actions_user(logits_u)
                    actions.append(cw_u)                 # (r, [slots])
                    acts_bin.append(a_u)                 # [num_slots] {0,1}
                    lp_total = lp_total + lp_u
                acts_bin = torch.stack(acts_bin, dim=0)  # [num_users, num_slots]

                # ----- Decode once -----
                decoded = run_sic_simulation(actions, num_slots)
                num_decoded = len(decoded)
                reward = num_decoded / num_users

                # ----- Sparsity encouragement -----
                activity = acts_bin.float().mean().item()  # mean 0/1 across all users/slots
                reward -= lam * activity
                last_activity = activity

                # accumulate
                batch_rewards.append(reward)
                batch_log_probs.append(lp_total)
                batch_decoded.append(num_decoded)

            # REINFORCE with baseline
            baseline = np.mean(batch_rewards)
            total_loss = sum([-(r - baseline) * lp for r, lp in zip(batch_rewards, batch_log_probs)])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # record epoch stats
            reward_history.append(baseline)
            avg_decoded_history.append(np.mean(batch_decoded))

            # optional logging
            #rec = {"epoch": epoch, "reward": float(baseline), "avg_decoded": float(avg_decoded_history[-1]), "activity": last_activity, "lambda": lam}
            if args.log:
                rec = {"epoch": epoch, "decoded_array": batch_decoded}            
                print(json.dumps(rec), file=log_f, flush=True)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: "
                      f"Avg Reward={baseline:.3f}, "
                      f"Avg decoded={avg_decoded_history[-1]:.2f}/{num_users}, "
                      f"Activity={last_activity:.3f} (λ={lam:.4f})")

            # optional checkpointing
            if (epoch + 1) % cfg['epoch_save_interval'] == 0:
                save_model(policy, result_dir, epoch=epoch + 1)
                cleanup_old_models(result_dir, keep_last=cfg['keep_last_models'])

        return reward_history, avg_decoded_history, dummy_hist

    # === Run ===
    policy = PolicyNetUser()
    optimizer = optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
    try:
        rewards, avg_decoded, _ = train(policy, optimizer)
    finally:
        log_f.close()

    save_model(policy, result_dir, epoch=None)


if __name__ == "__main__":
    main()
