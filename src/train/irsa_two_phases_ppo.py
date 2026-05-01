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
from src.irsa_common.sic import (
    feedback_indices_to_vector,
    run_sic_simulation,
    sample_actions_user,
    sic_decode,
)

# =========================
# Defaults / Hyperparameters
# =========================
DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_KEEP_LAST_MODELS = 2

# PPO defaults
DEFAULT_CLIP_EPS = 0.2
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_PPO_EPOCHS = 4
DEFAULT_MINIBATCH = 512
DEFAULT_MAX_GRAD_NORM = 0.5


# =========================
# Argparse
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="IRSA 2-Phases Training (PPO)")
    parser.add_argument('--users', type=int, required=True, help='Number of users')
    parser.add_argument('--slots', type=int, required=True, help='Number of slots')
    parser.add_argument('--torch-single-core', default=False, action="store_true")
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of simulated episodes per PPO update')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true', help='Compress log file with gzip')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS,
                        help='Keep only the last X saved models (default=2)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')

    # PPO-specific
    parser.add_argument('--clip-eps', type=float, default=DEFAULT_CLIP_EPS)
    parser.add_argument('--value-coef', type=float, default=DEFAULT_VALUE_COEF)
    parser.add_argument('--entropy-coef', type=float, default=DEFAULT_ENTROPY_COEF)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--gae-lambda', type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument('--ppo-epochs', type=int, default=DEFAULT_PPO_EPOCHS)
    parser.add_argument('--minibatch-size', type=int, default=DEFAULT_MINIBATCH)
    parser.add_argument('--max-grad-norm', type=float, default=DEFAULT_MAX_GRAD_NORM)
    return parser.parse_args()


# =========================
# Result dir / logging utils
# =========================
def make_result_dir(cfg):
    base = "res"
    if cfg["prefix"] is not None and cfg["prefix"] != "":
        base += "-" + cfg["prefix"]
    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
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
        parts.append(f"seed{cfg['seed']}")
        result_dir = "-".join(parts)
        result_dir = under_results(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


# =========================
# Loader (updated to Actor-Critic)
# =========================
def load_model_from_dir(result_dir, which="final", device=None):
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    prev_action_dim = cfg['num_slots']
    feedback_dim = 3 * cfg['num_slots']
    input_dim = cfg['input_obs_dim'] + feedback_dim + prev_action_dim

    class ActorCriticUser(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(input_dim, cfg['hidden_dim']),
                nn.ReLU(),
                nn.Linear(cfg['hidden_dim'], cfg['hidden_dim']),
                nn.ReLU(),
            )
            self.pi = nn.Linear(cfg['hidden_dim'], cfg['num_slots'])  # policy logits per slot (Multi-Bernoulli)
            self.v = nn.Linear(cfg['hidden_dim'], 1)
            with torch.no_grad():
                self.pi.bias.fill_(-1.4)

        def forward(self, x):
            h = self.body(x)
            logits = self.pi(h)
            value = self.v(h).squeeze(-1)
            return logits, value

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

    model = ActorCriticUser()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


# =========================
# Model
# =========================
class ActorCriticUser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slots):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, num_slots)  # logits per slot (indep. Bernoulli)
        self.v = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.pi.bias.fill_(-1.4)  # start a bit sparse

    def forward(self, x):
        h = self.body(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


# =========================
# PPO-specific log-prob / entropy helpers (sample_actions_user lives in irsa_common.sic)
# =========================
def logprob_multibernoulli(logits_user, action_bin):
    d = torch.distributions.Bernoulli(logits=logits_user)
    return d.log_prob(action_bin).sum(dim=-1)


def entropy_multibernoulli(logits_user):
    d = torch.distributions.Bernoulli(logits=logits_user)
    return d.entropy().sum(dim=-1)


# =========================
# GAE / PPO update
# =========================
@torch.no_grad()
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones: 1D tensors of same length (concatenated across episodes).
    We assume episodes are demarcated by dones==1 at terminal step of each episode.
    """
    T = rewards.shape[0]
    adv = torch.zeros(T)
    lastgaelam = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
        next_value = values[t]
    returns = adv + values
    return adv, returns


def ppo_update(model, optimizer, batch, cfg):
    obs, acts_bin, old_logp, values, rewards, dones = batch
    # Compute advantages/returns
    with torch.no_grad():
        adv, ret = compute_gae(rewards, values, dones, cfg['gamma'], cfg['gae_lambda'])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.size(0)
    idx_all = torch.arange(N)
    for _ in range(cfg['ppo_epochs']):
        perm = torch.randperm(N)
        for start in range(0, N, cfg['minibatch_size']):
            mb = perm[start:start + cfg['minibatch_size']]
            logits, values_pred = model(obs[mb])
            logp = logprob_multibernoulli(logits, acts_bin[mb])
            ratio = torch.exp(logp - old_logp[mb])

            # Policy loss (clipped)
            unclipped = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1.0 - cfg['clip_eps'], 1.0 + cfg['clip_eps']) * adv[mb]
            pg_loss = -torch.min(unclipped, clipped).mean()

            # Value loss
            v_loss = ((values_pred - ret[mb]) ** 2).mean()

            # Entropy bonus
            ent = entropy_multibernoulli(logits).mean()

            loss = pg_loss + cfg['value_coef'] * v_loss - cfg['entropy_coef'] * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            optimizer.step()


# =========================
# Training (PPO)
# =========================
def train_ppo(model, optimizer, cfg,
              sparsity_r1_max=0.02, sparsity_r2_max=0.01,
              warmup_r1=400, warmup_r2=800, log_file=None):
    reward_history = []
    avg_unique_history = []
    frac_decR1_txR2_hist = []

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']
    feedback_dim = 3 * num_slots
    prev_action_dim = num_slots
    input_dim = input_obs_dim + feedback_dim + prev_action_dim

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        # Ramped sparsity
        lam_r1 = sparsity_r1_max * min(1.0, epoch / float(warmup_r1))
        if epoch <= warmup_r1:
            lam_r2 = 0.0
        else:
            denom = max(1, warmup_r2 - warmup_r1)
            lam_r2 = sparsity_r2_max * min(1.0, (epoch - warmup_r1) / float(denom))

        # Rollout storage
        obs_buf = []
        act_bin_buf = []
        old_logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        batch_uniques = []
        batch_frac_decR1_txR2 = []
        last_r1_act, last_r2_act = 0.0, 0.0

        # Collect batch_size episodes per PPO update
        for _ in range(batch_size):
            # per-user noise
            obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

            # ---- Round 1 ----
            actions_r1, acts_bin_r1 = [], []
            lp_r1_total = 0.0
            r1_vals = []

            for u in range(num_users):
                x1 = torch.cat([obs_all[u], torch.zeros(feedback_dim), torch.zeros(prev_action_dim)], dim=0)
                logits_u, v_u = model(x1)
                cw_u, lp_u, a_u = sample_actions_user(logits_u)
                actions_r1.append(cw_u)
                acts_bin_r1.append(a_u)
                lp_r1_total = lp_r1_total + lp_u
                r1_vals.append(v_u)

                # Per-user bandit: every step is its own one-step episode (dones=1),
                # reward filled in after the joint decode below.
                obs_buf.append(x1)
                act_bin_buf.append(a_u)
                old_logp_buf.append(lp_u.detach())
                val_buf.append(v_u.detach())
                rew_buf.append(torch.tensor(0.0))
                done_buf.append(torch.tensor(1.0))

            acts_bin_r1 = torch.stack(acts_bin_r1, dim=0)
            decoded_r1, fb_idx = run_sic_simulation(actions_r1, num_slots, return_feedback_indices=True)
            fb_vec = feedback_indices_to_vector(fb_idx, num_slots)

            # ---- Round 2 ----
            actions_r2, acts_bin_r2 = [], []
            lp_r2_total = 0.0
            r2_vals = []

            for u in range(num_users):
                prev_act_u = acts_bin_r1[u].float()
                x2 = torch.cat([obs_all[u], fb_vec, prev_act_u], dim=0)
                logits_u, v_u = model(x2)
                cw_u, lp_u, a_u = sample_actions_user(logits_u)
                actions_r2.append(cw_u)
                acts_bin_r2.append(a_u)
                lp_r2_total = lp_r2_total + lp_u
                r2_vals.append(v_u)

                obs_buf.append(x2)
                act_bin_buf.append(a_u)
                old_logp_buf.append(lp_u.detach())
                val_buf.append(v_u.detach())
                rew_buf.append(torch.tensor(0.0))
                done_buf.append(torch.tensor(1.0))

            acts_bin_r2 = torch.stack(acts_bin_r2, dim=0)

            # ---- Concatenate schedules and decode once ----
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

            # Sparsity penalties
            r1_activity = acts_bin_r1.float().mean().item()
            r2_activity = acts_bin_r2.float().mean().item()
            reward_adjusted = reward - lam_r1 * r1_activity - lam_r2 * r2_activity
            last_r1_act, last_r2_act = r1_activity, r2_activity

            # Metric: frac(R1-decoded who transmit in R2)
            if len(decoded_r1) > 0:
                mask_dec = torch.zeros(num_users, dtype=torch.bool)
                mask_dec[list(decoded_r1)] = True
                r2_any = (acts_bin_r2.sum(dim=1) > 0)
                frac_decR1_txR2 = r2_any[mask_dec].float().mean().item()
            else:
                frac_decR1_txR2 = float('nan')

            # Per-user bandit: write the joint global reward to each of the
            # 2*num_users one-step episodes just collected.
            episode_len = 2 * num_users
            for k in range(episode_len):
                rew_buf[-episode_len + k] = torch.tensor(reward_adjusted)

            batch_uniques.append(num_decoded_concat)
            batch_frac_decR1_txR2.append(frac_decR1_txR2)

        # Stack rollout
        obs_t = torch.stack(obs_buf, dim=0)
        act_bin_t = torch.stack(act_bin_buf, dim=0)
        old_logp_t = torch.stack(old_logp_buf, dim=0)
        val_t = torch.stack(val_buf, dim=0)
        rew_t = torch.stack(rew_buf, dim=0)
        done_t = torch.stack(done_buf, dim=0)

        # PPO update
        batch = (obs_t, act_bin_t, old_logp_t, val_t, rew_t, done_t)
        ppo_update(model, optimizer, batch, cfg)

        # Logging
        baseline = rew_t[done_t == 1].mean().item() if (done_t == 1).any() else 0.0
        reward_history.append(baseline)
        avg_unique_history.append(np.mean(batch_uniques))
        frac_decR1_txR2_hist.append(np.nanmean(batch_frac_decR1_txR2))

        if log_file is not None:
            rec = {"epoch": epoch, "decoded_array": batch_uniques}
            print(json.dumps(rec), file=log_file, flush=True)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: "
                  f"Avg Reward={baseline:.3f}, "
                  f"Avg decoded={avg_unique_history[-1]:.2f}/{num_users}, "
                  #   f"R1 act={last_r1_act:.3f} (λ1={lam_r1:.4f}), "
                  #   f"R2 act={last_r2_act:.3f} (λ2={lam_r2:.4f})")
                  )
        # Optional checkpointing
        if (epoch + 1) % cfg['epoch_save_interval'] == 0:
            save_model(model, cfg['result_dir'], epoch=epoch + 1)
            cleanup_old_models(cfg['result_dir'], cfg['keep_last_models'])

    return reward_history, avg_unique_history, frac_decR1_txR2_hist


# =========================
# Main
# =========================
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
        'prefix': args.prefix,

        # PPO
        'clip_eps': args.clip_eps,
        'value_coef': args.value_coef,
        'entropy_coef': args.entropy_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'ppo_epochs': args.ppo_epochs,
        'minibatch_size': args.minibatch_size,
        'max_grad_norm': args.max_grad_norm,
    }

    # Derived dims (for saving cfg completeness)
    prev_action_dim = cfg['num_slots']
    feedback_dim = 3 * cfg['num_slots']
    input_dim = cfg['input_obs_dim'] + feedback_dim + prev_action_dim

    result_dir = make_result_dir(cfg)
    cfg['result_dir'] = result_dir

    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if args.log:
        log_f, log_path = get_log_file(result_dir, cfg['compress'])
    else:
        log_f, log_path = None, None

    # === Run PPO ===
    model = ActorCriticUser(input_dim, cfg['hidden_dim'], cfg['num_slots'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    try:
        rewards, avg_unique, frac_decR1_txR2 = train_ppo(model, optimizer, cfg, log_file=log_f)
    finally:
        if log_f is not None:
            log_f.close()

    save_model(model, result_dir, epoch=None)

    # === Print avg decoded users over last 500 epochs ===
    window = 500
    if len(avg_unique) >= window:
        avg_last500_decoded = np.mean(avg_unique[-window:])
    else:
        avg_last500_decoded = np.mean(avg_unique)
    print(f"Average users decoded (concatenated) over last {window} epochs: "
          f"{avg_last500_decoded:.3f}")

    window = 1000
    if len(avg_unique) >= window:
        avg_last500_decoded = np.mean(avg_unique[-window:])
    else:
        avg_last500_decoded = np.mean(avg_unique)
    print(f"Average users decoded (concatenated) over last {window} epochs: "
          f"{avg_last500_decoded:.3f}")


if __name__ == "__main__":
    main()