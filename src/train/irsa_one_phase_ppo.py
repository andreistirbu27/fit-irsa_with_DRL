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
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_KEEP_LAST_MODELS = 2

# PPO defaults - mirrored from irsa_two_phases_ppo.py
DEFAULT_CLIP_EPS = 0.2
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_PPO_EPOCHS = 4
DEFAULT_MINIBATCH = 512
DEFAULT_MAX_GRAD_NORM = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="IRSA Single-Round Training (PPO)")
    parser.add_argument('--users', type=int, required=True)
    parser.add_argument('--slots', type=int, required=True)
    parser.add_argument('--torch-single-core', default=False, action="store_true")
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of simulated episodes per PPO update')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--epoch-save-interval', type=int, default=200)
    parser.add_argument('--result-dir', type=str, default=None)
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS)
    parser.add_argument('--seed', type=int, required=True)

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


def make_result_dir(cfg):
    base = "res-1p-ppo"
    if cfg["prefix"] is not None and cfg["prefix"] != "":
        base += "-" + cfg["prefix"]
    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
        parts = [f"{base}-u{cfg['num_users']}", f"s{cfg['num_slots']}"]
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


def load_model_from_dir(result_dir, which="final", device=None):
    from src.irsa_common.io import load_model

    def factory(cfg):
        return ActorCriticUser(cfg['input_obs_dim'], cfg['hidden_dim'], cfg['num_slots'])

    return load_model(result_dir, factory, which=which, device=device)


class ActorCriticUser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slots):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, num_slots)
        self.v = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.pi.bias.fill_(-1.4)

    def forward(self, x):
        h = self.body(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


def logprob_multibernoulli(logits_user, action_bin):
    d = torch.distributions.Bernoulli(logits=logits_user)
    return d.log_prob(action_bin).sum(dim=-1)


def entropy_multibernoulli(logits_user):
    d = torch.distributions.Bernoulli(logits=logits_user)
    return d.entropy().sum(dim=-1)


@torch.no_grad()
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
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
    with torch.no_grad():
        adv, ret = compute_gae(rewards, values, dones, cfg['gamma'], cfg['gae_lambda'])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.size(0)
    for _ in range(cfg['ppo_epochs']):
        perm = torch.randperm(N)
        for start in range(0, N, cfg['minibatch_size']):
            mb = perm[start:start + cfg['minibatch_size']]
            logits, values_pred = model(obs[mb])
            logp = logprob_multibernoulli(logits, acts_bin[mb])
            ratio = torch.exp(logp - old_logp[mb])

            unclipped = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1.0 - cfg['clip_eps'], 1.0 + cfg['clip_eps']) * adv[mb]
            pg_loss = -torch.min(unclipped, clipped).mean()

            v_loss = ((values_pred - ret[mb]) ** 2).mean()
            ent = entropy_multibernoulli(logits).mean()

            loss = pg_loss + cfg['value_coef'] * v_loss - cfg['entropy_coef'] * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            optimizer.step()


def train_ppo(model, optimizer, cfg, sparsity_max=0.02, warmup=400, log_file=None):
    reward_history = []
    avg_decoded_history = []

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']
    input_dim = input_obs_dim  # single-round: input is just per-user noise

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        lam = sparsity_max * min(1.0, epoch / float(warmup))

        obs_buf, act_bin_buf, old_logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        batch_decoded = []
        last_activity = 0.0

        for _ in range(batch_size):
            obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

            actions, acts_bin = [], []
            for u in range(num_users):
                x = obs_all[u]
                assert x.numel() == input_dim
                logits_u, v_u = model(x)
                cw_u, lp_u, a_u = sample_actions_user(logits_u)
                actions.append(cw_u)
                acts_bin.append(a_u)

                obs_buf.append(x)
                act_bin_buf.append(a_u)
                old_logp_buf.append(lp_u.detach())
                val_buf.append(v_u.detach())
                done_buf.append(torch.tensor(1.0))

            acts_bin = torch.stack(acts_bin, dim=0)
            decoded = run_sic_simulation(actions, num_slots)
            num_decoded = len(decoded)
            reward = num_decoded / num_users

            activity = acts_bin.float().mean().item()
            reward_adjusted = reward - lam * activity
            last_activity = activity

            # Per-user bandit: each user step is its own one-step episode (dones=1),
            # all sharing the joint global reward. No γ chain across users.
            rew_buf.extend([torch.tensor(reward_adjusted)] * num_users)

            batch_decoded.append(num_decoded)

        obs_t = torch.stack(obs_buf, dim=0)
        act_bin_t = torch.stack(act_bin_buf, dim=0)
        old_logp_t = torch.stack(old_logp_buf, dim=0)
        val_t = torch.stack(val_buf, dim=0)
        rew_t = torch.stack(rew_buf, dim=0)
        done_t = torch.stack(done_buf, dim=0)

        ppo_update(model, optimizer, (obs_t, act_bin_t, old_logp_t, val_t, rew_t, done_t), cfg)

        baseline = rew_t[done_t == 1].mean().item() if (done_t == 1).any() else 0.0
        reward_history.append(baseline)
        avg_decoded_history.append(np.mean(batch_decoded))

        if log_file is not None:
            rec = {
                "epoch": epoch,
                "decoded_array": batch_decoded,
                "reward": float(baseline),
                "avg_decoded": float(avg_decoded_history[-1]),
                "activity": float(last_activity),
                "lambda": float(lam),
            }
            print(json.dumps(rec), file=log_file, flush=True)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: "
                  f"Avg Reward={baseline:.3f}, "
                  f"Avg decoded={avg_decoded_history[-1]:.2f}/{num_users}, "
                  f"Activity={last_activity:.3f} (λ={lam:.4f})")

        if (epoch + 1) % cfg['epoch_save_interval'] == 0:
            save_model(model, cfg['result_dir'], epoch=epoch + 1)
            cleanup_old_models(cfg['result_dir'], keep_last=cfg['keep_last_models'])

    return reward_history, avg_decoded_history


def main():
    args = parse_args()

    if args.users is None or args.slots is None:
        print("Error: --users and --slots are required.", file=sys.stderr)
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

        'clip_eps': args.clip_eps,
        'value_coef': args.value_coef,
        'entropy_coef': args.entropy_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'ppo_epochs': args.ppo_epochs,
        'minibatch_size': args.minibatch_size,
        'max_grad_norm': args.max_grad_norm,
    }

    input_dim = cfg['input_obs_dim']

    result_dir = make_result_dir(cfg)
    cfg['result_dir'] = result_dir

    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log_f = None
    if args.log:
        log_f, _ = get_log_file(result_dir, cfg['compress'])

    model = ActorCriticUser(input_dim, cfg['hidden_dim'], cfg['num_slots'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    try:
        rewards, avg_decoded = train_ppo(model, optimizer, cfg, log_file=log_f)
    finally:
        if log_f is not None:
            log_f.close()

    save_model(model, result_dir, epoch=None)

    window = 500
    if len(avg_decoded) >= window:
        avg_last = np.mean(avg_decoded[-window:])
    else:
        avg_last = np.mean(avg_decoded)
    print(f"Average users decoded over last {window} epochs: {avg_last:.3f}")


if __name__ == "__main__":
    main()
