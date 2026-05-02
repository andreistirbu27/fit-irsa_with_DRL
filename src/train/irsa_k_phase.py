"""IRSA k-phase REINFORCE trainer.

Generalizes irsa_two_phases.py to k>=2 subframes. Frame is k subframes of
`--slots` slots each (total = k * slots). After each phase t<k, the BS runs
SIC restricted to the slots of phase t alone and broadcasts ternary feedback
(decoded/empty/undecoded). At the end, SIC is run on the concatenated frame.

Policy input layout (fixed size across all phases):
    concat(
        obs                                    [input_obs_dim],
        feedback_current_phase                 [3 * slots],            # zeros for phase 1
        prev_actions_padded                    [(k-1) * slots],        # zero-padded
        phase_one_hot                          [k]
    )

For k=2 this is bit-equivalent in shape to irsa_two_phases.py modulo the
phase one-hot tail (which is appended; see input_dim definition).
"""
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

DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_KEEP_LAST_MODELS = 2


def parse_args():
    parser = argparse.ArgumentParser(description="IRSA k-Phases Training (REINFORCE)")
    parser.add_argument('--users', type=int, required=True, help='Number of users')
    parser.add_argument('--slots', type=int, required=True, help='Slots per subframe')
    parser.add_argument('--num-phases', type=int, required=True, help='Number of phases k (>=2)')
    parser.add_argument('--torch-single-core', default=False, action="store_true")
    parser.add_argument('--input-obs-dim', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--epoch-save-interval', type=int, default=200)
    parser.add_argument('--result-dir', type=str, default=None)
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    if args.num_phases < 2:
        parser.error("--num-phases must be >= 2")
    return args


def make_result_dir(cfg):
    base = "res-kp"
    if cfg["prefix"]:
        base += "-" + cfg["prefix"]
    if cfg['result_dir'] is not None:
        result_dir = cfg['result_dir']
    else:
        parts = [f"{base}-u{cfg['num_users']}", f"s{cfg['num_slots']}"]
        if cfg['num_phases'] != 2:
            parts.append(f"k{cfg['num_phases']}")
        if cfg['hidden_dim'] != DEFAULT_HIDDEN_DIM:
            parts.append(f"h{cfg['hidden_dim']}")
        if cfg['epochs'] != DEFAULT_EPOCHS:
            parts.append(f"e{cfg['epochs']}")
        if cfg['batch_size'] != DEFAULT_BATCH_SIZE:
            parts.append(f"b{cfg['batch_size']}")
        if cfg['learning_rate'] != DEFAULT_LEARNING_RATE:
            parts.append(f"lr{cfg['learning_rate']}")
        parts.append(f"seed{cfg['seed']}")
        result_dir = under_results("-".join(parts))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def compute_input_dim(cfg):
    return (cfg['input_obs_dim']
            + 3 * cfg['num_slots']
            + (cfg['num_phases'] - 1) * cfg['num_slots']
            + cfg['num_phases'])


def load_model_from_dir(result_dir, which="final", device=None):
    from src.irsa_common.io import load_model

    def factory(cfg):
        return PolicyNetUser(compute_input_dim(cfg), cfg['hidden_dim'], cfg['num_slots'])

    return load_model(result_dir, factory, which=which, device=device)


class PolicyNetUser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_slots):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots),
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(-1.4)

    def forward(self, x):
        return self.net(x)


def build_policy_input(obs_u, fb_vec_current, prev_actions_list, phase_idx, cfg):
    """Pack per-user policy input for phase `phase_idx` (0-indexed).

    fb_vec_current: 3*num_slots vector for the most recent SIC feedback (zeros at phase 0).
    prev_actions_list: list of length phase_idx, each [num_slots] in {0,1}, in order.
    """
    num_slots = cfg['num_slots']
    k = cfg['num_phases']
    prev_pad = torch.zeros((k - 1) * num_slots)
    if prev_actions_list:
        flat = torch.cat(prev_actions_list, dim=0)
        prev_pad[:flat.numel()] = flat
    phase_oh = torch.zeros(k)
    phase_oh[phase_idx] = 1.0
    return torch.cat([obs_u, fb_vec_current, prev_pad, phase_oh], dim=0)


def train(policy, optimizer, cfg, log_file=None):
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    k = cfg['num_phases']
    input_obs_dim = cfg['input_obs_dim']
    expected_input_dim = compute_input_dim(cfg)

    # Per-phase ramped sparsity. Match irsa_two_phases.py for k=2:
    # phase 0: λ_max=0.02 ramps over [0, 400]; phase 1: λ_max=0.01 ramps over [400, 800].
    # General: phase i has λ_max = 0.02 / 2**i, ramp window [400*i, 400*(i+1)].
    sparsity_max = [0.02 / (2 ** i) for i in range(k)]
    warmup_starts = [400 * i for i in range(k)]
    warmup_ends = [400 * (i + 1) for i in range(k)]

    reward_history = []
    avg_unique_history = []

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        lams = []
        for i in range(k):
            if epoch <= warmup_starts[i]:
                lams.append(0.0)
            else:
                denom = max(1, warmup_ends[i] - warmup_starts[i])
                lams.append(sparsity_max[i] * min(1.0, (epoch - warmup_starts[i]) / float(denom)))

        batch_rewards, batch_log_probs, batch_uniques = [], [], []
        last_activities = [0.0] * k

        for _ in range(batch_size):
            obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

            # Per-phase storage
            actions_per_phase = []      # list[len=k] of list[num_users] of (r, slot_list) tuples
            acts_bin_per_phase = []     # list[len=k] of [num_users, num_slots] tensors
            lp_per_phase = []           # list[len=k] of scalar log-prob tensors

            fb_vec_current = torch.zeros(3 * num_slots)

            for phase_idx in range(k):
                actions_phase, acts_bin_phase = [], []
                lp_phase_total = 0.0
                for u in range(num_users):
                    prev_actions_u = [acts_bin_per_phase[p][u].float() for p in range(phase_idx)]
                    x = build_policy_input(obs_all[u], fb_vec_current, prev_actions_u, phase_idx, cfg)
                    assert x.numel() == expected_input_dim, (x.numel(), expected_input_dim)
                    logits_u = policy(x)
                    cw_u, lp_u, a_u = sample_actions_user(logits_u)
                    actions_phase.append(cw_u)
                    acts_bin_phase.append(a_u)
                    lp_phase_total = lp_phase_total + lp_u
                acts_bin_phase = torch.stack(acts_bin_phase, dim=0)  # [num_users, num_slots]

                actions_per_phase.append(actions_phase)
                acts_bin_per_phase.append(acts_bin_phase)
                lp_per_phase.append(lp_phase_total)

                # Feedback for the next phase: SIC restricted to this subframe alone.
                if phase_idx < k - 1:
                    _, fb_idx = run_sic_simulation(actions_phase, num_slots, return_feedback_indices=True)
                    fb_vec_current = feedback_indices_to_vector(fb_idx, num_slots)

            # Concatenated SIC over the full frame (k * num_slots slots).
            actions_concat = []
            for u in range(num_users):
                combined = []
                for phase_idx in range(k):
                    _, slot_list = actions_per_phase[phase_idx][u]
                    offset = phase_idx * num_slots
                    combined.extend(s + offset for s in slot_list)
                actions_concat.append((len(combined), combined))

            decoded_concat = sic_decode(actions_concat, total_slots=k * num_slots)
            num_decoded_concat = len(decoded_concat)
            reward = num_decoded_concat / num_users

            phase_activities = [acts_bin_per_phase[i].float().mean().item() for i in range(k)]
            for i in range(k):
                reward -= lams[i] * phase_activities[i]
            last_activities = phase_activities

            batch_rewards.append(reward)
            batch_log_probs.append(sum(lp_per_phase))
            batch_uniques.append(num_decoded_concat)

        baseline = float(np.mean(batch_rewards))
        total_loss = sum([-(r - baseline) * lp for r, lp in zip(batch_rewards, batch_log_probs)])
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        reward_history.append(baseline)
        avg_unique_history.append(float(np.mean(batch_uniques)))

        if log_file is not None:
            rec = {
                "epoch": epoch,
                "decoded_array": batch_uniques,
                "avg_reward": baseline,
                "avg_unique": avg_unique_history[-1],
            }
            for i in range(k):
                rec[f"activity_phase{i}"] = float(last_activities[i])
                rec[f"lambda_phase{i}"] = float(lams[i])
            print(json.dumps(rec), file=log_file, flush=True)

        if epoch % 100 == 0:
            act_str = ", ".join(f"P{i} act={last_activities[i]:.3f} (λ={lams[i]:.4f})" for i in range(k))
            print(f"Epoch {epoch}: Avg Reward={baseline:.3f}, "
                  f"Avg decoded={avg_unique_history[-1]:.2f}/{num_users}, {act_str}")

        if (epoch + 1) % cfg['epoch_save_interval'] == 0:
            save_model(policy, cfg['result_dir'], epoch=epoch + 1)
            cleanup_old_models(cfg['result_dir'], keep_last=cfg['keep_last_models'])

    return reward_history, avg_unique_history


def main():
    args = parse_args()
    if args.torch_single_core:
        set_torch_single_core()
    set_seed(args.seed)

    cfg = {
        'num_users': args.users,
        'num_slots': args.slots,
        'num_phases': args.num_phases,
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
    }

    input_dim = compute_input_dim(cfg)
    result_dir = make_result_dir(cfg)
    cfg['result_dir'] = result_dir

    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if args.log:
        log_f, _ = get_log_file(result_dir, cfg['compress'])
    else:
        log_f = None

    policy = PolicyNetUser(input_dim, cfg['hidden_dim'], cfg['num_slots'])
    optimizer = optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
    try:
        train(policy, optimizer, cfg, log_file=log_f)
    finally:
        if log_f is not None:
            log_f.close()

    save_model(policy, result_dir, epoch=None)


if __name__ == "__main__":
    main()
