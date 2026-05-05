"""Policy statistics for trained IRSA policies.

Produces the data behind conference Figs 10/11 and new journal analysis
(symmetry-break score, conditional entropy, seed-to-seed TV distance).
Outputs are saved as .npz files; plotting is handled by plot_policy_stats.py.

Usage (single run):
    python -m src.analysis.policy_stats \\
        --result-dir results/new/res-2p-u15-s10-seed1 \\
        --phase 1 --out figs/policy_stats/res-2p-u15-s10-seed1_p1/ \\
        --metrics latent envelopes pooled symbreak fidelity

Usage (multi-seed):
    python -m src.analysis.policy_stats --multi-seed \\
        --result-dirs results/new/res-2p-u8-s4-seed{1..20} \\
        --out figs/policy_stats/seed_distance_u8_s4/

For k-phase models, --phase 1..k selects which subframe to analyse.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.irsa_common.sic import (
    feedback_indices_to_vector,
    run_sic_simulation,
    sample_actions_user,
)


# ============================================================
# Model loading
# ============================================================

def _load_model_any(result_dir, which='final', device=None):
    """Load any trained policy from result_dir → (policy, cfg)."""
    config_path = os.path.join(result_dir, 'config.json')
    with open(config_path) as f:
        cfg = json.load(f)

    name = os.path.basename(result_dir)
    if name.startswith('res-kp') or 'num_phases' in cfg:
        from src.train.irsa_k_phase import load_model_from_dir
    elif 'clip_eps' in cfg:
        from src.train.irsa_two_phases_ppo import load_model_from_dir
    elif 'num_layers' in cfg:
        from src.train.irsa_2phase_2x64 import load_model_from_dir
    elif name.startswith('res-1p'):
        from src.train.irsa_one_phase import load_model_from_dir
    else:
        from src.train.irsa_two_phases import load_model_from_dir

    return load_model_from_dir(result_dir, which=which, device=device)


def _is_one_phase(cfg):
    return 'num_phases' not in cfg and cfg.get('num_slots') == cfg.get('num_slots')  # always true
    # Real check: 1-phase models have no feedback_dim key and no prev_action concept.
    # We detect by checking the policy input_dim matches input_obs_dim only.


def _model_type(cfg, result_dir):
    name = os.path.basename(result_dir)
    if 'num_phases' in cfg or name.startswith('res-kp'):
        return 'kphase'
    if 'clip_eps' in cfg:
        return 'ppo'
    if name.startswith('res-1p'):
        return '1phase'
    return '2phase'


# ============================================================
# Policy output sampling
# ============================================================

def _build_2phase_input_p1(cfg, obs):
    """Phase 1 input for a 2-phase model (feedback=0, prev=0)."""
    num_slots = cfg['num_slots']
    return torch.cat([
        obs,
        torch.zeros(3 * num_slots),
        torch.zeros(num_slots),
    ], dim=0)


def _build_2phase_input_p2(cfg, obs, fb_vec, prev_action):
    """Phase 2 input for a 2-phase model."""
    return torch.cat([obs, fb_vec, prev_action], dim=0)


def _build_kphase_input(cfg, obs, fb_vec_current, prev_actions_list, phase_idx):
    """Build k-phase policy input using the same layout as irsa_k_phase.build_policy_input."""
    num_slots = cfg['num_slots']
    k = cfg['num_phases']
    prev_pad = torch.zeros((k - 1) * num_slots)
    if prev_actions_list:
        flat = torch.cat(prev_actions_list, dim=0)
        prev_pad[:flat.numel()] = flat
    phase_oh = torch.zeros(k)
    phase_oh[phase_idx] = 1.0
    return torch.cat([obs, fb_vec_current, prev_pad, phase_oh], dim=0)


@torch.no_grad()
def sample_policy_outputs(model, cfg, result_dir, n_samples=1_000_000, phase=1):
    """Sample raw sigmoid(logits) probability vectors from the policy.

    phase: 1-indexed phase number (1 = first subframe).
    Returns np.ndarray of shape [n_samples, num_slots].

    Phase 1: feedback=0, prev_actions=0, obs=rand.
    Phase 2+: simulate preceding phases with the policy, build actual feedback.
    """
    model.eval()
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']
    mtype = _model_type(cfg, result_dir)
    phase_idx = phase - 1  # 0-indexed

    batch_size = min(n_samples, 4096)
    n_batches = (n_samples + batch_size - 1) // batch_size
    results = []

    for _ in range(n_batches):
        bs = min(batch_size, n_samples - len(results) * batch_size)
        # correct for last batch
        bs = batch_size

        obs_batch = torch.rand(bs, input_obs_dim)  # [bs, input_obs_dim]

        if phase_idx == 0:
            # No feedback needed regardless of model type
            if mtype == 'kphase':
                k = cfg['num_phases']
                fb = torch.zeros(bs, 3 * num_slots)
                prev_pad = torch.zeros(bs, (k - 1) * num_slots)
                phase_oh = torch.zeros(bs, k)
                phase_oh[:, 0] = 1.0
                x = torch.cat([obs_batch, fb, prev_pad, phase_oh], dim=1)
            elif mtype == '1phase':
                x = obs_batch  # 1-phase: input is only obs
            else:  # 2phase, ppo
                fb = torch.zeros(bs, 3 * num_slots)
                prev = torch.zeros(bs, num_slots)
                x = torch.cat([obs_batch, fb, prev], dim=1)

            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            results.append(probs.cpu().numpy())

        else:
            # Must simulate preceding phases one sample at a time to get feedback
            probs_batch = []
            for i in range(bs):
                obs_u = obs_batch[i]  # [input_obs_dim] — shared noise for all users (simplified)
                num_users = cfg.get('num_users', 1)

                # Simulate from phase 0 up to (but not including) phase_idx
                obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]
                acts_bin_phases = []
                actions_phases = []
                fb_vec = torch.zeros(3 * num_slots)

                for p in range(phase_idx):
                    acts_bin_p, actions_p = [], []
                    for u in range(num_users):
                        if mtype == 'kphase':
                            prev_list = [acts_bin_phases[pp][u].float() for pp in range(p)]
                            xin = _build_kphase_input(cfg, obs_all[u], fb_vec, prev_list, p)
                        else:
                            xin = _build_2phase_input_p1(cfg, obs_all[u]) if p == 0 else \
                                  _build_2phase_input_p2(cfg, obs_all[u], fb_vec, acts_bin_phases[0][u].float())
                        logits_u = model(xin)
                        if isinstance(logits_u, tuple):
                            logits_u = logits_u[0]
                        cw_u, _, a_u = sample_actions_user(logits_u)
                        actions_p.append(cw_u)
                        acts_bin_p.append(a_u)

                    acts_bin_phases.append(acts_bin_p)
                    actions_phases.append(actions_p)

                    if p < phase_idx - 1 or mtype == 'kphase':
                        _, fb_idx = run_sic_simulation(actions_p, num_slots, return_feedback_indices=True)
                        fb_vec = feedback_indices_to_vector(fb_idx, num_slots)

                # Now build the input for the target phase for this one user (obs_u)
                if mtype == 'kphase':
                    prev_list = [acts_bin_phases[pp][0].float() for pp in range(phase_idx)]
                    xin = _build_kphase_input(cfg, obs_u, fb_vec, prev_list, phase_idx)
                else:
                    prev_act = acts_bin_phases[0][0].float() if acts_bin_phases else torch.zeros(num_slots)
                    xin = _build_2phase_input_p2(cfg, obs_u, fb_vec, prev_act)

                logits_u = model(xin)
                if isinstance(logits_u, tuple):
                    logits_u = logits_u[0]
                probs_u = torch.sigmoid(logits_u)
                probs_batch.append(probs_u.cpu().numpy())

            results.append(np.stack(probs_batch, axis=0))  # [bs, num_slots]

        if sum(r.shape[0] for r in results) >= n_samples:
            break

    probs = np.concatenate(results, axis=0)[:n_samples]
    return probs  # [n_samples, num_slots]


# ============================================================
# Statistics
# ============================================================

def estimate_latent_D(probs, bins=50):
    """Estimate per-sample mean (the latent D-hat distribution).

    Returns (per_sample_mean, bin_edges, counts).
    """
    per_sample_mean = probs.mean(axis=1)
    counts, bin_edges = np.histogram(per_sample_mean, bins=bins, range=(0.0, 1.0))
    return per_sample_mean, bin_edges, counts


def per_slot_envelopes(probs):
    """Compute per-slot normalized envelopes (reproduces conference Fig 10 bottom).

    For each sample, normalize slot probs by the sample mean.
    Sort slots by their average normalized value.
    Returns dict with keys: slot_order, median, q25, q75, q05, q95, q01, q99
    all of shape [num_slots].
    """
    per_sample_mean = probs.mean(axis=1, keepdims=True)
    # avoid division by zero for samples where all probs are 0
    safe_mean = np.where(per_sample_mean > 0, per_sample_mean, 1.0)
    normalized = probs / safe_mean  # [n_samples, num_slots]

    slot_avg_norm = normalized.mean(axis=0)
    slot_order = np.argsort(slot_avg_norm)[::-1]  # descending

    n_sorted = normalized[:, slot_order]
    return {
        'slot_order': slot_order,
        'median': np.median(n_sorted, axis=0),
        'q25': np.percentile(n_sorted, 25, axis=0),
        'q75': np.percentile(n_sorted, 75, axis=0),
        'q05': np.percentile(n_sorted, 5, axis=0),
        'q95': np.percentile(n_sorted, 95, axis=0),
        'q01': np.percentile(n_sorted, 1, axis=0),
        'q99': np.percentile(n_sorted, 99, axis=0),
    }


def pooled_histogram(probs, bins=50):
    """Histogram of all per-slot probabilities (reproduces conference Fig 11).

    Returns (bin_edges, counts).
    """
    flat = probs.flatten()
    counts, bin_edges = np.histogram(flat, bins=bins, range=(0.0, 1.0))
    return bin_edges, counts


def symmetry_break_score(probs):
    """Quantitative symmetry-breaking metric.

    Returns var(slot_means) / mean(within-sample variance).
    ~0 = permutation-symmetric; >0 = symmetry broken.
    """
    slot_means = probs.mean(axis=0)           # [num_slots]
    var_slot_means = slot_means.var()
    within_sample_var = probs.var(axis=1).mean()
    if within_sample_var < 1e-12:
        return 0.0
    return float(var_slot_means / within_sample_var)


@torch.no_grad()
def latent_D_fidelity_throughput(model, cfg, result_dir, n_frames=100_000, n_latent_samples=100_000):
    """Compare throughput under actual policy vs latent D-hat surrogate.

    Phase 1 only. Returns (throughput_policy, throughput_latent).
    """
    from src.irsa_common.sic import run_sic_simulation, sample_actions_user

    model.eval()
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']
    mtype = _model_type(cfg, result_dir)

    # Build D-hat histogram from n_latent_samples
    probs_latent = sample_policy_outputs(model, cfg, result_dir, n_samples=n_latent_samples, phase=1)
    per_sample_mean = probs_latent.mean(axis=1)
    # Estimate D_hat as histogram; we'll sample theta ~ D_hat per user
    d_hat_values = per_sample_mean  # use empirical distribution directly

    # (a) Actual policy throughput
    decoded_policy = []
    for _ in range(n_frames):
        obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]
        actions = []
        for u in range(num_users):
            obs = obs_all[u]
            if mtype == 'kphase':
                k = cfg['num_phases']
                fb = torch.zeros(3 * num_slots)
                prev_pad = torch.zeros((k - 1) * num_slots)
                phase_oh = torch.zeros(k); phase_oh[0] = 1.0
                x = torch.cat([obs, fb, prev_pad, phase_oh])
            elif mtype == '1phase':
                x = obs
            else:
                x = torch.cat([obs, torch.zeros(3 * num_slots), torch.zeros(num_slots)])
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            cw, _, _ = sample_actions_user(logits)
            actions.append(cw)
        decoded = run_sic_simulation(actions, num_slots)
        decoded_policy.append(len(decoded) / num_users)

    throughput_policy = float(np.mean(decoded_policy))

    # (b) Latent D-hat surrogate throughput
    decoded_latent = []
    for _ in range(n_frames):
        # Draw one theta ~ D_hat per user, then Bernoulli(theta) per slot
        thetas = d_hat_values[np.random.randint(0, len(d_hat_values), size=num_users)]
        actions = []
        for u in range(num_users):
            theta = thetas[u]
            # Bernoulli(theta) independently per slot
            slots = [s for s in range(num_slots) if np.random.rand() < theta]
            actions.append((len(slots), slots))
        decoded = run_sic_simulation(actions, num_slots)
        decoded_latent.append(len(decoded) / num_users)

    throughput_latent = float(np.mean(decoded_latent))
    return throughput_policy, throughput_latent


@torch.no_grad()
def phase2_conditional_entropy(model, cfg, result_dir, n_samples=100_000):
    """Weighted average entropy of Phase 2 distribution over feedback buckets.

    Buckets: (n_decoded, n_empty, n_undecoded) triples rather than full patterns.
    Returns dict: {'entropy_mean': float, 'entropy_by_bucket': dict, 'bucket_weights': dict}.
    """
    from scipy.stats import entropy as scipy_entropy
    from collections import defaultdict

    model.eval()
    num_users = cfg['num_users']
    num_slots = cfg['num_slots']
    input_obs_dim = cfg['input_obs_dim']
    mtype = _model_type(cfg, result_dir)

    bucket_probs = defaultdict(list)  # bucket → list of [num_slots] prob vectors

    for _ in range(n_samples):
        obs_u = torch.rand(input_obs_dim)
        obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

        # Phase 1 rollout
        if mtype == 'kphase':
            k = cfg['num_phases']
            actions_p1 = []
            for u in range(num_users):
                fb = torch.zeros(3 * num_slots)
                prev_pad = torch.zeros((k - 1) * num_slots)
                phase_oh = torch.zeros(k); phase_oh[0] = 1.0
                x = torch.cat([obs_all[u], fb, prev_pad, phase_oh])
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
                cw, _, _ = sample_actions_user(logits)
                actions_p1.append(cw)
        else:
            actions_p1 = []
            for u in range(num_users):
                x = torch.cat([obs_all[u], torch.zeros(3 * num_slots), torch.zeros(num_slots)])
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
                cw, _, a_u = sample_actions_user(logits)
                actions_p1.append(cw)

        # SIC and feedback
        decoded_r1, fb_idx = run_sic_simulation(actions_p1, num_slots, return_feedback_indices=True)
        fb_vec = feedback_indices_to_vector(fb_idx, num_slots)

        # Bucket by (n_decoded, n_empty, n_undecoded)
        n_dec = len(fb_idx[0])
        n_emp = len(fb_idx[1])
        n_und = len(fb_idx[2])
        bucket = (n_dec, n_emp, n_und)

        # Phase 2 probability vector for obs_u
        if mtype == 'kphase':
            # Use a placeholder prev_action (zeroed) — this is an approximation
            prev_list = [torch.zeros(num_slots)]  # phase 1 action unknown here; use zeros
            xin = _build_kphase_input(cfg, obs_u, fb_vec, prev_list, 1)
        else:
            xin = torch.cat([obs_u, fb_vec, torch.zeros(num_slots)])  # prev_action=0 for entropy estimate

        logits2 = model(xin)
        if isinstance(logits2, tuple): logits2 = logits2[0]
        p2 = torch.sigmoid(logits2).cpu().numpy()
        bucket_probs[bucket].append(p2)

    # Per-bucket entropy (mean entropy of Bernoulli distribution per slot, averaged over slots)
    total_samples = sum(len(v) for v in bucket_probs.values())
    entropy_by_bucket = {}
    bucket_weights = {}
    weighted_entropy = 0.0

    for bucket, plist in bucket_probs.items():
        pmat = np.stack(plist, axis=0)  # [n, num_slots]
        # Per-sample Bernoulli entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        H = -(pmat * np.log(pmat + eps) + (1 - pmat) * np.log(1 - pmat + eps))  # [n, num_slots]
        mean_H = float(H.mean())
        w = len(plist) / total_samples
        entropy_by_bucket[bucket] = mean_H
        bucket_weights[bucket] = w
        weighted_entropy += w * mean_H

    return {
        'entropy_mean': weighted_entropy,
        'entropy_by_bucket': entropy_by_bucket,
        'bucket_weights': bucket_weights,
        'n_samples': n_samples,
    }


@torch.no_grad()
def seed_to_seed_distance(model_paths_and_cfgs, n_samples=200_000, bins=100):
    """Compute pairwise total-variation distance between pooled slot-prob histograms.

    model_paths_and_cfgs: list of result_dir strings.
    Returns (tv_matrix [n_seeds, n_seeds], seed_labels list).
    """
    n = len(model_paths_and_cfgs)
    histograms = []

    for result_dir in model_paths_and_cfgs:
        model, cfg = _load_model_any(result_dir)
        probs = sample_policy_outputs(model, cfg, result_dir, n_samples=n_samples, phase=1)
        flat = probs.flatten()
        counts, _ = np.histogram(flat, bins=bins, range=(0.0, 1.0))
        hist = counts / counts.sum()
        histograms.append(hist)

    tv_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            tv = 0.5 * np.abs(histograms[i] - histograms[j]).sum()
            tv_matrix[i, j] = tv
            tv_matrix[j, i] = tv

    return tv_matrix, list(model_paths_and_cfgs)


# ============================================================
# Save helpers
# ============================================================

def save_metric(out_dir, name, **arrays):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}.npz')
    np.savez(path, **arrays)
    print(f"  Saved {path}")


# ============================================================
# CLI
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(description='Compute and save policy statistics')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--result-dir', default=None,
                      help='Single run directory to analyse')
    mode.add_argument('--multi-seed', action='store_true',
                      help='Multi-seed TV distance mode; use --result-dirs')
    p.add_argument('--result-dirs', nargs='+', default=None,
                   help='List of result dirs for --multi-seed mode')
    p.add_argument('--phase', type=int, default=1,
                   help='Which phase to analyse (1-indexed, default: 1)')
    p.add_argument('--out', required=True,
                   help='Output directory for .npz files')
    p.add_argument('--metrics', nargs='+',
                   choices=['latent', 'envelopes', 'pooled', 'symbreak', 'fidelity', 'entropy', 'all'],
                   default=['all'],
                   help='Which metrics to compute (default: all)')
    p.add_argument('--n-samples', type=int, default=1_000_000,
                   help='Samples for sample_policy_outputs (default: 1M)')
    p.add_argument('--n-frames', type=int, default=100_000,
                   help='Frames for fidelity throughput evaluation (default: 100k)')
    return p.parse_args()


def run_single(result_dir, phase, out_dir, metrics, n_samples, n_frames):
    print(f"Loading model from {result_dir}")
    model, cfg = _load_model_any(result_dir)
    model.eval()

    do_all = 'all' in metrics
    do = lambda m: do_all or m in metrics

    probs = None
    if do('latent') or do('envelopes') or do('pooled') or do('symbreak'):
        print(f"  Sampling {n_samples} policy outputs (phase {phase})...")
        probs = sample_policy_outputs(model, cfg, result_dir, n_samples=n_samples, phase=phase)

    if do('latent') and probs is not None:
        print("  Computing latent D...")
        per_mean, bin_edges, counts = estimate_latent_D(probs)
        save_metric(out_dir, 'latent_D',
                    per_sample_mean=per_mean, bin_edges=bin_edges, counts=counts)

    if do('envelopes') and probs is not None:
        print("  Computing per-slot envelopes...")
        env = per_slot_envelopes(probs)
        save_metric(out_dir, 'envelopes', **env)

    if do('pooled') and probs is not None:
        print("  Computing pooled histogram...")
        bin_edges, counts = pooled_histogram(probs)
        save_metric(out_dir, 'pooled_hist', bin_edges=bin_edges, counts=counts)

    if do('symbreak') and probs is not None:
        score = symmetry_break_score(probs)
        print(f"  Symmetry-break score: {score:.4f}")
        save_metric(out_dir, 'symbreak', score=np.array([score]),
                    num_slots=np.array([cfg['num_slots']]),
                    phase=np.array([phase]))

    if do('fidelity') and phase == 1:
        print(f"  Evaluating latent D fidelity throughput ({n_frames} frames)...")
        tp_pol, tp_lat = latent_D_fidelity_throughput(model, cfg, result_dir, n_frames=n_frames)
        print(f"  Throughput: policy={tp_pol:.4f}, latent={tp_lat:.4f}, delta={tp_pol-tp_lat:+.4f}")
        save_metric(out_dir, 'fidelity',
                    throughput_policy=np.array([tp_pol]),
                    throughput_latent=np.array([tp_lat]))

    if do('entropy') and phase >= 2:
        print(f"  Computing Phase {phase} conditional entropy...")
        result = phase2_conditional_entropy(model, cfg, result_dir)
        print(f"  Weighted entropy: {result['entropy_mean']:.4f}")
        buckets = list(result['entropy_by_bucket'].keys())
        bucket_arr = np.array(buckets)
        H_arr = np.array([result['entropy_by_bucket'][b] for b in buckets])
        w_arr = np.array([result['bucket_weights'][b] for b in buckets])
        save_metric(out_dir, 'conditional_entropy',
                    entropy_mean=np.array([result['entropy_mean']]),
                    buckets=bucket_arr, H_per_bucket=H_arr, weights=w_arr)


def run_multi_seed(result_dirs, out_dir, n_samples=200_000):
    print(f"Computing seed-to-seed TV distance for {len(result_dirs)} runs...")
    tv_matrix, labels = seed_to_seed_distance(result_dirs, n_samples=n_samples)
    print(f"  Mean pairwise TV: {tv_matrix[tv_matrix > 0].mean():.4f}")
    save_metric(out_dir, 'seed_tv_distance',
                tv_matrix=tv_matrix,
                labels=np.array(labels, dtype=object))


def main():
    args = _parse_args()
    metrics = args.metrics

    if args.multi_seed:
        if not args.result_dirs:
            print("Error: --result-dirs required with --multi-seed", file=sys.stderr)
            sys.exit(1)
        run_multi_seed(args.result_dirs, args.out, n_samples=min(args.n_samples, 200_000))
    else:
        if not args.result_dir:
            print("Error: --result-dir required", file=sys.stderr)
            sys.exit(1)
        run_single(args.result_dir, args.phase, args.out, metrics, args.n_samples, args.n_frames)


if __name__ == '__main__':
    main()
