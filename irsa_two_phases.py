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



try:
    import gzip
except ImportError:
    gzip = None

DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 0.02
DEFAULT_KEEP_LAST_MODELS = 2

def parse_args():
    parser = argparse.ArgumentParser(description="IRSA 2-Phases Training")
    parser.add_argument('--users', type=int, default=4, help='Number of users (required)')
    parser.add_argument('--slots', type=int, default=2, help='Number of slots (required)')
    parser.add_argument('--torch-single-core', default=False, action="store_true") 
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true', help='Compress log file with gzip')
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS, help='Keep only the last X saved models (default=2)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
    return parser.parse_args()

def make_result_dir(cfg):
    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
        # Only include non-defaults for hidden-dim, epochs, batch-size, learning-rate
        parts = [
            f"result-u{cfg['num_users']}",
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
        if cfg.get('seed', None) is not None:
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
                    last.bias.fill_(-1.4)          # optional: start sparser (sigmoid ~0.2)

        def forward(self, x):          # x: [input_dim]
            return self.net(x)         # [num_slots] logits

    def sample_actions_user(logits_user):
        """Bernoulli per slot for one user; allows 0,1,2,... slots."""
        d = torch.distributions.Bernoulli(logits=logits_user)
        a = d.sample()  # [num_slots] {0,1}
        lp = d.log_prob(a).sum()  # scalar log-prob for this user
        slots = torch.where(a == 1)[0].tolist()
        return (len(slots), slots), lp, a  # (r,[slots]), logprob, tensor

    #import torch.nn.functional as F

    #def sample_actions_user_fast(logits_user: torch.Tensor):
    #    # logits_user: 1D float tensor [num_slots]
    #    probs = torch.sigmoid(logits_user)             # σ(logits)
    #    a = torch.bernoulli(probs)                     # float {0.,1.}
    #    # log p(a | logits) = -BCEWithLogits(a, logits)
    #    lp = -F.binary_cross_entropy_with_logits(logits_user, a, reduction='sum')
    #    idx = a.nonzero(as_tuple=True)[0]              # indices where a==1
    #
    #    # Prefer returning tensors (avoids Python list conversion cost):
    #    return (idx.numel(), idx), lp, a


    # === SIC + feedback (decoded slots are those that became empty after SIC but were non-empty initially) ===
    def run_sic_simulation(actions, return_feedback_indices=False):
        slots_init = [[] for _ in range(cfg['num_slots'])]
        for user_id, (r, slot_list) in enumerate(actions):
            for s in slot_list[:r]:
                slots_init[s].append(user_id)
        initial_empty = {i for i in range(cfg['num_slots']) if len(slots_init[i]) == 0}

        slots = [lst.copy() for lst in slots_init]
        decoded_users = set()
        progress = True
        while progress:
            progress = False
            for s in range(cfg['num_slots']):
                if len(slots[s]) == 1:
                    u = slots[s][0]
                    if u not in decoded_users:
                        decoded_users.add(u)
                        for t in range(cfg['num_slots']):
                            if u in slots[t]:
                                slots[t].remove(u)
                        progress = True

        if return_feedback_indices:
            final_empty = {i for i in range(cfg['num_slots']) if len(slots[i]) == 0}
            decoded_idx = sorted(list(final_empty - initial_empty))  # became empty due to SIC
            empty_idx = sorted(list(initial_empty))  # empty from start
            undec_idx = sorted([i for i in range(cfg['num_slots'])
                                if len(slots_init[i]) > 0 and len(slots[i]) > 0])  # still occupied
            return decoded_users, [decoded_idx, empty_idx, undec_idx]

        return decoded_users

    def feedback_indices_to_vector(feedback_indices):
        decoded_idx, empty_idx, undec_idx = feedback_indices
        d = torch.zeros(cfg['num_slots'])
        e = torch.zeros(cfg['num_slots'])
        u = torch.zeros(cfg['num_slots'])
        if decoded_idx: d[decoded_idx] = 1
        if empty_idx:   e[empty_idx] = 1
        if undec_idx:   u[undec_idx] = 1
        return torch.cat([d, e, u], dim=0)  # length = 3*num_slots

    # === Training (apply policy per user) ===
    def train(policy, optimizer,
              sparsity_r1_max=0.02, sparsity_r2_max=0.01,
              warmup_r1=400, warmup_r2=800):
        reward_history = []
        avg_unique_history = []
        frac_decR1_txR2_hist = []   # fraction: R1-decoded who still tx in R2

        for epoch in tqdm(range(cfg['epochs']), desc="Training", unit="epoch"):
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

            for _ in range(cfg['batch_size']):
                # per-user noise (keep or share, your call)
                obs_all = [torch.rand(cfg['input_obs_dim']) for _ in range(cfg['num_users'])]

                # -------- Round 1: per-user policy (feedback=0, prev_action=zeros) --------
                actions_r1, acts_bin_r1 = [], []
                lp_r1_total = 0.0
                for u in range(cfg['num_users']):
                    x1 = torch.cat([
                        obs_all[u],
                        torch.zeros(feedback_dim),       # no feedback yet
                        torch.zeros(prev_action_dim)     # no previous action yet
                    ], dim=0)
                    assert x1.numel() == input_dim
                    logits_u = policy(x1)                                # [num_slots]
                    cw_u, lp_u, a_u = sample_actions_user(logits_u)     # a_u: [num_slots] in {0,1}
                    actions_r1.append(cw_u)
                    acts_bin_r1.append(a_u)
                    lp_r1_total = lp_r1_total + lp_u
                acts_bin_r1 = torch.stack(acts_bin_r1, dim=0)            # [num_users, num_slots]

                decoded_r1, fb_idx = run_sic_simulation(actions_r1, return_feedback_indices=True)

                # -------- Round 2: per-user policy (feedback + prev_action from R1) --------
                fb_vec = feedback_indices_to_vector(fb_idx)              # len = 3*num_slots
                actions_r2, acts_bin_r2 = [], []
                lp_r2_total = 0.0
                for u in range(cfg['num_users']):
                    prev_act_u = acts_bin_r1[u].float()                  # THIS user's R1 action (0/1 per slot)
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
                acts_bin_r2 = torch.stack(acts_bin_r2, dim=0)            # [num_users, num_slots]

                # -------- Reward: union of decodes across rounds --------
                decoded_r2 = run_sic_simulation(actions_r2)
                num_unique = len(decoded_r1.union(decoded_r2))
                reward = num_unique / cfg['num_users']

                # -------- Ramped sparsity in BOTH rounds --------
                r1_activity = acts_bin_r1.float().mean().item()
                r2_activity = acts_bin_r2.float().mean().item()
                reward -= lam_r1 * r1_activity
                reward -= lam_r2 * r2_activity
                last_r1_act, last_r2_act = r1_activity, r2_activity

                # -------- Metric: frac(R1-decoded who transmit in R2) --------
                if len(decoded_r1) > 0:
                    mask_dec = torch.zeros(cfg['num_users'], dtype=torch.bool)
                    mask_dec[list(decoded_r1)] = True
                    r2_any = (acts_bin_r2.sum(dim=1) > 0)               # per-user bool
                    frac_decR1_txR2 = r2_any[mask_dec].float().mean().item()
                else:
                    frac_decR1_txR2 = float('nan')

                # accumulate
                batch_rewards.append(reward)
                batch_log_probs.append(lp_r1_total + lp_r2_total)
                batch_uniques.append(num_unique)
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

            # Save stats as JSON line
            log_line = {
                "epoch": epoch,
                "avg_reward": float(baseline),
                "avg_unique": float(avg_unique_history[-1]),
                "frac_decR1_txR2": float(frac_decR1_txR2_hist[-1]) if not np.isnan(frac_decR1_txR2_hist[-1]) else None,
                "R1_activity": float(last_r1_act),
                "R2_activity": float(last_r2_act),
                "lam_r1": float(lam_r1),
                "lam_r2": float(lam_r2),
            }
            log_f.write(json.dumps(log_line) + "\n")
            log_f.flush()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: "
                      f"Avg Reward={baseline:.3f}, "
                      f"Avg unique={avg_unique_history[-1]:.2f}/{cfg['num_users']}, "
                      f"frac(R1-decoded tx in R2)={frac_decR1_txR2_hist[-1]:.3f}, "
                      f"R1 act={last_r1_act:.3f} (λ1={lam_r1:.4f}), "
                      f"R2 act={last_r2_act:.3f} (λ2={lam_r2:.4f})")

            # Save model at interval
            if (epoch > 0) and (epoch % cfg['epoch_save_interval'] == 0):
                save_model(policy, result_dir, epoch=epoch)
                cleanup_old_models(result_dir, keep_last=cfg.get('keep_last_models', DEFAULT_KEEP_LAST_MODELS))

        return reward_history, avg_unique_history, frac_decR1_txR2_hist

    # === Run ===
    policy = PolicyNetUser()
    optimizer = optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
    try:
        rewards, avg_unique, frac_decR1_txR2 = train(policy, optimizer)
    finally:
        log_f.close()

    # Save final model
    save_model(policy, result_dir, epoch=None)

if __name__ == "__main__":
    main()

