"""Plot policy statistics from .npz files produced by policy_stats.py.

Produces:
  - Latent D-hat histogram + per-slot envelopes (2×2 panel, conf Fig 10 style)
  - Pooled per-slot probability histogram (conf Fig 11 style)
  - Symmetry-break score bar chart
  - Seed-to-seed TV distance heatmap
  - Phase 2 conditional entropy bar chart

All figures saved as PDF at 300 dpi with embedded fonts.

Usage:
    python -m src.plot.plot_policy_stats \\
        --in figs/journal/policy_stats/ \\
        --out figs/journal/

    # Single run dir with all available npz files:
    python -m src.plot.plot_policy_stats \\
        --in figs/journal/policy_stats/res-2p-u15-s10-seed1_p1/ \\
        --out figs/journal/ --prefix 2p_u15_s10_p1
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load(npz_dir, name):
    """Load a .npz file by name from npz_dir. Returns the NpzFile or None."""
    path = os.path.join(npz_dir, f'{name}.npz')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


# ============================================================
# Individual figure functions
# ============================================================

def plot_latent_and_envelopes(npz_dir, out_path=None, title_prefix=''):
    """2×2 panel: D-hat histogram (top) + per-slot envelope (bottom).
    Reproduces conference paper Fig 10 layout.
    """
    lat = _load(npz_dir, 'latent_D')
    env = _load(npz_dir, 'envelopes')

    if lat is None and env is None:
        print(f"  Skipping latent/envelopes: neither latent_D.npz nor envelopes.npz found in {npz_dir}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    if lat is not None:
        ax = axes[0]
        edges = lat['bin_edges']
        counts = lat['counts']
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts / counts.sum(), width=edges[1] - edges[0],
               color='C0', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Per-sample mean probability θ̂')
        ax.set_ylabel('Density')
        ax.set_title(f'{title_prefix}Latent D̂ distribution')
        ax.grid(True, alpha=0.3)
    else:
        axes[0].set_visible(False)

    if env is not None:
        ax = axes[1]
        num_slots = len(env['median'])
        x = np.arange(num_slots)
        ax.fill_between(x, env['q01'], env['q99'], color='C0', alpha=0.1, label='1–99%')
        ax.fill_between(x, env['q05'], env['q95'], color='C0', alpha=0.2, label='5–95%')
        ax.fill_between(x, env['q25'], env['q75'], color='C0', alpha=0.35, label='IQR')
        ax.plot(x, env['median'], color='C0', linewidth=1.5, label='Median')
        ax.set_xlabel('Slot rank (sorted by avg normalized prob)')
        ax.set_ylabel('Normalized slot probability')
        ax.set_title(f'{title_prefix}Per-slot envelopes')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    _save(fig, out_path)


def plot_pooled_histogram(npz_dir, out_path=None, title_prefix=''):
    """Pooled per-slot probability histogram (conference Fig 11 style)."""
    data = _load(npz_dir, 'pooled_hist')
    if data is None:
        print(f"  Skipping pooled_hist: pooled_hist.npz not found in {npz_dir}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    edges = data['bin_edges']
    counts = data['counts']
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts / counts.sum(), width=edges[1] - edges[0],
           color='C1', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Per-slot probability p')
    ax.set_ylabel('Density')
    ax.set_title(f'{title_prefix}Pooled slot-probability distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_symbreak_bars(npz_dirs_and_labels, out_path=None):
    """Bar chart of symmetry-break scores across phases/configs.

    npz_dirs_and_labels: list of (npz_dir, label) tuples.
    """
    scores, labels = [], []
    for npz_dir, label in npz_dirs_and_labels:
        data = _load(npz_dir, 'symbreak')
        if data is None:
            continue
        scores.append(float(data['score']))
        labels.append(label)

    if not scores:
        print("  Skipping symbreak: no symbreak.npz files found")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(scores) * 0.8), 4))
    x = np.arange(len(scores))
    ax.bar(x, scores, color='C2', alpha=0.8, edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Symmetry-break score\n(var(slot means) / mean within-sample var)')
    ax.set_title('Policy symmetry-breaking by phase and config')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_tv_heatmap(npz_dir, out_path=None, title='Seed-to-seed TV distance'):
    """Heatmap of pairwise total-variation distances."""
    data = _load(npz_dir, 'seed_tv_distance')
    if data is None:
        print(f"  Skipping TV heatmap: seed_tv_distance.npz not found in {npz_dir}")
        return

    tv = data['tv_matrix']
    labels_raw = data['labels']
    # Shorten labels to seed number if possible
    labels = []
    for lb in labels_raw:
        lb_str = str(lb)
        import re
        m = re.search(r'seed(\d+)', lb_str)
        labels.append(f"seed{m.group(1)}" if m else lb_str.split('/')[-1][:12])

    n = tv.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(tv, vmin=0, vmax=tv.max(), cmap='Blues')
    plt.colorbar(im, ax=ax, label='TV distance')
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    mean_tv = tv[tv > 0].mean() if (tv > 0).any() else 0
    ax.set_xlabel(f'Mean pairwise TV = {mean_tv:.4f}')
    plt.tight_layout()
    _save(fig, out_path)


def plot_conditional_entropy(npz_dir, out_path=None, title_prefix=''):
    """Bar chart of Phase 2 conditional entropy per feedback bucket."""
    data = _load(npz_dir, 'conditional_entropy')
    if data is None:
        print(f"  Skipping conditional entropy: conditional_entropy.npz not found in {npz_dir}")
        return

    buckets = data['buckets']
    H_vals = data['H_per_bucket']
    weights = data['weights']
    entropy_mean = float(data['entropy_mean'])

    # Sort by weight descending
    order = np.argsort(weights)[::-1]
    buckets = buckets[order]
    H_vals = H_vals[order]
    weights = weights[order]

    # Only show top-N buckets
    top_n = min(30, len(buckets))
    bucket_labels = [f"({int(b[0])},{int(b[1])},{int(b[2])})" for b in buckets[:top_n]]

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.4), 5))
    x = np.arange(top_n)
    bars = ax.bar(x, H_vals[:top_n], color='C3', alpha=0.7, edgecolor='k')

    # Color by weight (darker = more common bucket)
    norm_weights = weights[:top_n] / weights[:top_n].max()
    for bar, nw in zip(bars, norm_weights):
        bar.set_alpha(0.3 + 0.7 * nw)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Mean Bernoulli entropy (nats)')
    ax.set_title(f'{title_prefix}Phase 2 conditional entropy (weighted avg = {entropy_mean:.4f})\n'
                 f'Bucket = (n_decoded, n_empty, n_undecoded), sorted by frequency')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_fidelity_comparison(fidelity_data, labels, out_path=None):
    """Bar chart comparing policy throughput vs latent surrogate throughput.

    fidelity_data: list of (throughput_policy, throughput_latent) tuples.
    labels: matching list of label strings.
    """
    if not fidelity_data:
        return

    n = len(fidelity_data)
    x = np.arange(n)
    bar_w = 0.35
    tp_pol = [d[0] for d in fidelity_data]
    tp_lat = [d[1] for d in fidelity_data]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.0), 5))
    ax.bar(x - bar_w/2, tp_pol, width=bar_w, label='Policy', color='C0', alpha=0.85, edgecolor='k')
    ax.bar(x + bar_w/2, tp_lat, width=bar_w, label='Latent D̂', color='C1', alpha=0.7, edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Mean throughput (decoded / users)')
    ax.set_title('Policy vs latent D̂ surrogate throughput (Phase 1)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


# ============================================================
# Helpers
# ============================================================

def _save(fig, out_path):
    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {out_path}")
    plt.show()
    plt.close(fig)


def _discover_subdirs(root):
    """Return all immediate subdirectories of root."""
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]


# ============================================================
# CLI
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(description='Plot policy statistics from .npz files')
    p.add_argument('--in', dest='npz_in', required=True,
                   help='Directory containing .npz files (or parent directory of subdirs)')
    p.add_argument('--out', required=True,
                   help='Output directory for figures')
    p.add_argument('--prefix', default='',
                   help='Filename prefix for saved figures')
    p.add_argument('--no-show', action='store_true',
                   help='Do not call plt.show() (useful for batch runs)')
    return p.parse_args()


def main():
    args = _parse_args()
    npz_dir = args.npz_in
    out_dir = args.out
    prefix = (args.prefix + '_') if args.prefix else ''

    if args.no_show:
        matplotlib.use('Agg')

    os.makedirs(out_dir, exist_ok=True)

    def out(name):
        return os.path.join(out_dir, f'{prefix}{name}.pdf')

    # Single-dir mode: plot all available metrics from npz_dir
    print(f"Plotting from {npz_dir}")

    plot_latent_and_envelopes(npz_dir,
                              out_path=out('latent_envelopes'),
                              title_prefix=args.prefix + ' ' if args.prefix else '')

    plot_pooled_histogram(npz_dir,
                          out_path=out('pooled_hist'),
                          title_prefix=args.prefix + ' ' if args.prefix else '')

    plot_conditional_entropy(npz_dir,
                             out_path=out('conditional_entropy'),
                             title_prefix=args.prefix + ' ' if args.prefix else '')

    plot_tv_heatmap(npz_dir, out_path=out('tv_heatmap'))

    # For symbreak: check for files in subdirs too (multi-phase/multi-config usage)
    subdirs = _discover_subdirs(npz_dir)
    if subdirs:
        symbreak_items = [(sd, os.path.basename(sd)) for sd in sorted(subdirs)
                         if os.path.exists(os.path.join(sd, 'symbreak.npz'))]
        # also check the dir itself
        if os.path.exists(os.path.join(npz_dir, 'symbreak.npz')):
            symbreak_items = [(npz_dir, os.path.basename(npz_dir))] + symbreak_items
        if symbreak_items:
            plot_symbreak_bars(symbreak_items, out_path=out('symbreak'))

    elif os.path.exists(os.path.join(npz_dir, 'symbreak.npz')):
        plot_symbreak_bars([(npz_dir, args.prefix or os.path.basename(npz_dir))],
                           out_path=out('symbreak'))

    # Fidelity: collect from subdirs if available
    fidelity_data, fid_labels = [], []
    for sd in sorted(subdirs):
        fd = _load(sd, 'fidelity')
        if fd is not None:
            fidelity_data.append((float(fd['throughput_policy']), float(fd['throughput_latent'])))
            fid_labels.append(os.path.basename(sd))
    fd = _load(npz_dir, 'fidelity')
    if fd is not None and not fidelity_data:
        fidelity_data = [(float(fd['throughput_policy']), float(fd['throughput_latent']))]
        fid_labels = [args.prefix or os.path.basename(npz_dir)]
    if fidelity_data:
        plot_fidelity_comparison(fidelity_data, fid_labels, out_path=out('fidelity'))


if __name__ == '__main__':
    main()
