"""Convergence-across-seeds figure.

For one chosen config (e.g. --config u15-s10 --variant 2p), loads all available
seed logs and plots:
  - 20 thin grey training curves (avg_unique vs epoch)
  - Median across seeds in solid black
  - IQR (25th–75th percentile) as a shaded band

Usage:
    python -m src.plot.plot_convergence \\
        --config u15-s10 --variant 2p --seeds 1 20 \\
        --results-dir results/new \\
        --out figs/journal/fig_convergence.pdf
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.plot.plot_results_long import find_jsonl_file, load_jsonl, _throughput_key


def load_convergence_curve(run_dir):
    """Return array of per-epoch throughput values (avg_unique or avg_decoded)."""
    path = find_jsonl_file(run_dir)
    data = load_jsonl(path)
    return np.array([_throughput_key(r) for r in data])


def find_run_dir(results_dir, variant, config, prefix='', num_phases=None, seed=1):
    """Locate a run dir matching the given parameters."""
    # config is like 'u15-s10'
    # dir name pattern: res-{variant}[-prefix]-{config}[-k{k}]...-seed{seed}
    base = f"res-{variant}"
    if prefix:
        base += f"-{prefix}"
    base += f"-{config}"
    suffix = f"-seed{seed}"

    candidates = []
    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            if name.startswith(base) and name.endswith(suffix):
                full = os.path.join(results_dir, name)
                if os.path.isdir(full):
                    candidates.append(full)

    if not candidates:
        return None
    # prefer exact match (no extra hparams)
    exact = base + suffix
    for c in candidates:
        if os.path.basename(c) == exact:
            return c
    return candidates[0]


def plot_convergence(
    config,
    variant='2p',
    seeds=None,
    prefix='',
    results_dir='results/new',
    smooth_window=1,
    out=None,
    title=None,
):
    """Plot convergence curves across seeds for one config.

    config: string like 'u15-s10'
    seeds: list of ints; if None, auto-detect all available seeds 1..50
    """
    if seeds is None:
        seeds = list(range(1, 51))

    curves = {}
    for seed in seeds:
        run_dir = find_run_dir(results_dir, variant, config, prefix=prefix, seed=seed)
        if run_dir is None:
            continue
        try:
            curves[seed] = load_convergence_curve(run_dir)
        except Exception as e:
            print(f"  Skipping seed {seed}: {e}")

    if not curves:
        print(f"No runs found for variant={variant}, config={config} in {results_dir}")
        return

    # Align all curves to the same length (truncate to shortest)
    min_len = min(len(c) for c in curves.values())
    mat = np.stack([c[:min_len] for c in curves.values()], axis=0)  # [n_seeds, epochs]

    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        mat = np.apply_along_axis(lambda x: uniform_filter1d(x, size=smooth_window), axis=1, arr=mat)

    epochs = np.arange(min_len)
    median = np.median(mat, axis=0)
    q25 = np.percentile(mat, 25, axis=0)
    q75 = np.percentile(mat, 75, axis=0)

    n_seeds = mat.shape[0]
    fig, ax = plt.subplots(figsize=(10, 5))

    for row in mat:
        ax.plot(epochs, row, color='gray', linewidth=0.5, alpha=0.4)

    ax.fill_between(epochs, q25, q75, color='C0', alpha=0.25, label='IQR (25–75%)')
    ax.plot(epochs, median, color='black', linewidth=1.5, label='Median')

    if title is None:
        u_s = config.upper().replace('-', ', ').replace('U', 'N=').replace('S', 'M=')
        variant_label = {'2p': 'FIT-IRSA', '1p': 'P1-IRSA', 'kp': 'k-phase'}.get(variant, variant)
        title = f"Training stability across {n_seeds} seeds, {u_s}, {variant_label}"
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg decoded users")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
    plt.show()

    print(f"Loaded {n_seeds} seeds, {min_len} epochs each.")
    print(f"Final median: {median[-1]:.3f}, IQR: [{q25[-1]:.3f}, {q75[-1]:.3f}]")


def _parse_args():
    p = argparse.ArgumentParser(description="Plot convergence curves across seeds")
    p.add_argument('--config', required=True,
                   help='Config string, e.g. u15-s10')
    p.add_argument('--variant', default='2p',
                   help='Variant prefix: 1p, 2p, kp, etc. (default: 2p)')
    p.add_argument('--seeds', nargs=2, type=int, default=None, metavar=('FIRST', 'LAST'),
                   help='Seed range inclusive, e.g. --seeds 1 20')
    p.add_argument('--prefix', default='',
                   help='Run prefix, e.g. load (default: empty)')
    p.add_argument('--results-dir', default='results/new')
    p.add_argument('--smooth', type=int, default=1, metavar='W',
                   help='SMA smoothing window in epochs (default: 1 = no smoothing)')
    p.add_argument('--out', default=None,
                   help='Output figure path (PDF recommended)')
    p.add_argument('--title', default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    seeds = list(range(args.seeds[0], args.seeds[1] + 1)) if args.seeds else None
    plot_convergence(
        config=args.config,
        variant=args.variant,
        seeds=seeds,
        prefix=args.prefix,
        results_dir=args.results_dir,
        smooth_window=args.smooth,
        out=args.out,
        title=args.title,
    )


if __name__ == '__main__':
    main()
