#%%

import argparse
import os
import re
import sys
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def find_jsonl_file(run_dir):
    """
    Find a .jsonl or .jsonl.gz file in the given directory.
    Prefers .jsonl.gz over .jsonl when both exist.
    Returns the path to the file, or raises FileNotFoundError.
    """
    gz = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.jsonl.gz')), None)
    if gz is not None:
        return gz
    plain = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith('.jsonl')), None)
    if plain is not None:
        return plain
    raise FileNotFoundError("No .jsonl or .jsonl.gz file found in {}".format(run_dir))

def load_jsonl(path):
    """
    Load a jsonl or jsonl.gz file and return a list of dicts.
    """
    if path.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'
    with open_fn(path, mode) as f:
        return [json.loads(line) for line in f if line.strip()]

def smooth_sma(x, k=31):
    # mode="same" pads with zeros at boundaries; first/last k//2 samples droop toward zero.
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def smooth_sma_nan(x, k=51):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    w = np.ones(k, dtype=float)
    mask = ~np.isnan(x)
    x_filled = np.nan_to_num(x, nan=0.0)
    num = np.convolve(x_filled, w, mode="same")
    den = np.convolve(mask.astype(float), w, mode="same")
    y = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return y


#%%

CONFIDENCE = 0.99

def compute_array_stats(
    all_data,
    K=1,
    confidence=0.99,
    array_key="decoded_array",
    label="Mean"
):
    """
    Compute means and confidence intervals over batches of data for a specified array key.

    Returns:
        epoch_centers, means, cis_lower, cis_upper
    """
    means = []
    cis_lower = []
    cis_upper = []
    epoch_centers = []

    num_batches = len(all_data) // K
    for i in range(num_batches):
        batch = all_data[i*K:(i+1)*K]
        batch_array = np.concatenate([np.array(d[array_key]) for d in batch])
        mean = batch_array.mean()
        std = batch_array.std(ddof=1)
        n = len(batch_array)
        if n > 1:
            ci = t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
        else:
            ci = 0.0
        means.append(mean)
        cis_lower.append(mean - ci)
        cis_upper.append(mean + ci)
        batch_epochs = [d.get("epoch", idx) for idx, d in enumerate(batch, start=i*K)]
        epoch_centers.append(np.mean(batch_epochs))

    if len(all_data) % K != 0:
        batch = all_data[num_batches*K:]
        batch_array = np.concatenate([np.array(d[array_key]) for d in batch])
        mean = batch_array.mean()
        std = batch_array.std(ddof=1)
        n = len(batch_array)
        if n > 1:
            ci = t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
        else:
            ci = 0.0
        means.append(mean)
        cis_lower.append(mean - ci)
        cis_upper.append(mean + ci)
        batch_epochs = [d.get("epoch", idx) for idx, d in enumerate(batch, start=num_batches*K)]
        epoch_centers.append(np.mean(batch_epochs))

    return epoch_centers, means, cis_lower, cis_upper


#%%
# ============================================================
# New-format run directory parsing and data collection
# ============================================================

_KNOWN_VARIANTS = ['2p-ppo', '1p-ppo', '2p-2x64', 'kp', '2p', '1p']


def parse_run_dir(dirname):
    """Parse a run directory basename into a metadata dict.

    Returns a dict with keys: variant, prefix, users, slots, num_phases, seed
    or None if the name does not match the expected format.
    """
    name = os.path.basename(dirname)
    if not name.startswith('res-'):
        return None

    rest = name[4:]  # strip 'res-'

    variant = None
    for v in _KNOWN_VARIANTS:
        if rest.startswith(v + '-'):
            variant = v
            rest = rest[len(v) + 1:]
            break
    if variant is None:
        return None

    # Optional prefix: any lowercase segment that does NOT start with 'u\d'
    prefix = ''
    m = re.match(r'^([a-z][a-z0-9]*)-(u\d)', rest)
    if m and not m.group(1).startswith('u'):
        prefix = m.group(1)
        rest = rest[len(prefix) + 1:]

    m = re.match(r'^u(\d+)-s(\d+)(-.+)?$', rest)
    if not m:
        return None
    users = int(m.group(1))
    slots = int(m.group(2))
    tail = m.group(3) or ''

    num_phases = 2  # default for kp; irrelevant for others
    km = re.search(r'-k(\d+)', tail)
    if km:
        num_phases = int(km.group(1))

    sm = re.search(r'-seed(\d+)$', tail)
    if not sm:
        return None
    seed = int(sm.group(1))

    return {
        'variant': variant,
        'prefix': prefix,
        'users': users,
        'slots': slots,
        'num_phases': num_phases,
        'seed': seed,
        'path': dirname,
    }


def collect_runs(results_dir='results/new'):
    """Scan results_dir and return a list of metadata dicts for all parseable run dirs."""
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for name in os.listdir(results_dir):
        full = os.path.join(results_dir, name)
        if not os.path.isdir(full):
            continue
        meta = parse_run_dir(full)
        if meta is not None:
            runs.append(meta)
    return runs


def _throughput_key(record):
    """Return the scalar throughput value from one log record."""
    if 'avg_unique' in record:
        return float(record['avg_unique'])
    if 'avg_decoded' in record:
        return float(record['avg_decoded'])
    return float(np.mean(record['decoded_array']))


def final_throughput_seed(run_dir, last_frac=0.1):
    """Return mean throughput over the last last_frac fraction of training epochs."""
    path = find_jsonl_file(run_dir)
    data = load_jsonl(path)
    n = max(1, int(len(data) * last_frac))
    vals = [_throughput_key(r) for r in data[-n:]]
    return float(np.mean(vals))


#%%
# ============================================================
# Legacy per-seed plot (matches conference paper)
# ============================================================

def get_load_data(nb_users, nb_slots=20, one_phase=False, seed=1, prefix="-load"):
    if one_phase:
        dir_name = f"results/res-long/res{prefix}-1p-u{nb_users}-s{nb_slots}-e1000-b1000-s{seed}"
    else:
        dir_name = f"results/res-long/res{prefix}-u{nb_users}-s{nb_slots//2}-e1000-b1000-s{seed}"
    log_file_name = find_jsonl_file(dir_name)
    all_data = load_jsonl(log_file_name)

    NB_PARTS = 10

    K = int(len(all_data)//NB_PARTS)
    ARRAY_KEY = "decoded_array"
    LABEL = "Mean"

    epoch_centers, means, cis_lower, cis_upper = compute_array_stats(
        all_data, K=K, confidence=CONFIDENCE, array_key=ARRAY_KEY, label=LABEL
    )
    return epoch_centers[-1], means[-1], cis_lower[-1], cis_upper[-1]


def plot_throughtput_vs_users_slots(prefix="-load", normalize=False, nb_slots=20, xlim=None, ylim=None, fig_file_name=None):
    if nb_slots is not None:
        user_range = range(2, 31)
    else:
        user_range = range(2, 31, 2)
    seeds = range(1, 5)

    results = {
        True: {
            "avgs": {seed: [] for seed in seeds},
            "lowers": {seed: [] for seed in seeds},
            "uppers": {seed: [] for seed in seeds},
        },
        False: {
            "avgs": {seed: [] for seed in seeds},
            "lowers": {seed: [] for seed in seeds},
            "uppers": {seed: [] for seed in seeds},
        }
    }

    for nb_users in user_range:
        for one_phase in [False, True]:
            for seed in seeds:
                try:
                    actual_nb_slots = nb_users if (nb_slots is None) else nb_slots
                    _, avg, lower, upper = get_load_data(nb_users, nb_slots=actual_nb_slots, seed=seed, one_phase=one_phase, prefix=prefix)
                    if normalize:
                        avg = avg / nb_users
                        lower = lower / nb_users
                        upper = upper / nb_users
                    results[one_phase]["avgs"][seed].append(avg)
                    results[one_phase]["lowers"][seed].append(lower)
                    results[one_phase]["uppers"][seed].append(upper)
                except Exception as e:
                    print(f"Skipping users={nb_users}, seed={seed}, one_phase={one_phase} due to error: {e}")
                    results[one_phase]["avgs"][seed].append(np.nan)
                    results[one_phase]["lowers"][seed].append(np.nan)
                    results[one_phase]["uppers"][seed].append(np.nan)

    x = np.array(list(user_range))
    bar_width = 0.18
    offsets = np.linspace(-bar_width*1.5, bar_width*1.5, len(seeds))

    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, seed in enumerate(seeds):
        y = np.array(results[False]["avgs"][seed])
        yerr_lower = y - np.array(results[False]["lowers"][seed])
        yerr_upper = np.array(results[False]["uppers"][seed]) - y
        plt.bar(
            x + offsets[i], y, width=bar_width*1.15,
            yerr=[yerr_lower, yerr_upper],
            align='center', alpha=0.4, ecolor='black', capsize=4,
            label=f"Seed {seed}, 2 Phases", color=colors[i % len(colors)], zorder=1
        )

    for i, seed in enumerate(seeds):
        y = np.array(results[True]["avgs"][seed])
        yerr_lower = y - np.array(results[True]["lowers"][seed])
        yerr_upper = np.array(results[True]["uppers"][seed]) - y
        plt.bar(
            x + offsets[i], y, width=bar_width,
            yerr=[yerr_lower, yerr_upper],
            align='center', alpha=0.8, ecolor='black', capsize=4,
            label=f"Seed {seed}, 1 Phase", color=colors[i % len(colors)], zorder=2, edgecolor='k'
        )

    plt.xlabel("Number of Users")
    plt.ylabel("Mean Throughput")
    plt.title("")
    plt.xticks(x)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    if fig_file_name is not None:
        plt.savefig(fig_file_name)
    plt.show()


#%%
# ============================================================
# New across-seed CI bar chart
# ============================================================

def _group_key(meta, sweep):
    """Return the grouping key for a run metadata dict under the given sweep."""
    if sweep == 'var-users-kphase':
        return (meta['variant'], meta['users'], meta['slots'], meta['num_phases'])
    return (meta['variant'], meta['prefix'], meta['users'], meta['slots'])


def plot_throughput_across_seeds(
    sweep,
    results_dir='results/new',
    normalize=False,
    confidence=0.99,
    last_frac=0.1,
    xlim=None,
    ylim=None,
    out=None,
):
    """Bar chart of throughput vs users with across-seed t-distribution CIs.

    sweep: one of 'var-users', 'var-load', 'var-users-kphase'
    """
    all_runs = collect_runs(results_dir)

    # Filter by sweep type
    if sweep == 'var-users':
        runs = [r for r in all_runs if r['prefix'] == '' and r['variant'] in ('1p', '2p')]
    elif sweep == 'var-load':
        runs = [r for r in all_runs if r['prefix'] == 'load' and r['variant'] in ('1p', '2p')]
    elif sweep == 'var-users-kphase':
        runs = [r for r in all_runs if r['variant'] == 'kp']
    else:
        raise ValueError(f"Unknown sweep: {sweep!r}. Use var-users, var-load, or var-users-kphase.")

    if not runs:
        print(f"No runs found for sweep={sweep!r} in {results_dir}")
        return

    # Group runs by config (excluding seed)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        k = _group_key(r, sweep)
        groups[k].append(r)

    # Compute per-group across-seed CI
    group_stats = {}
    for k, group_runs in sorted(groups.items()):
        vals = []
        for r in group_runs:
            try:
                v = final_throughput_seed(r['path'], last_frac=last_frac)
                vals.append(v)
            except Exception as e:
                print(f"  Skipping {r['path']}: {e}")
        if not vals:
            continue
        vals = np.array(vals)
        mean = vals.mean()
        n = len(vals)
        if n > 1:
            ci = t.ppf((1 + confidence) / 2, n - 1) * vals.std(ddof=1) / np.sqrt(n)
        else:
            ci = 0.0
        group_stats[k] = (mean, ci, n)

    if not group_stats:
        print("No data available to plot.")
        return

    # Extract sorted user counts and variant display names
    if sweep == 'var-users-kphase':
        all_keys = sorted(group_stats.keys(), key=lambda k: (k[0], k[3], k[1]))
        phase_values = sorted(set(k[3] for k in all_keys))
        user_values = sorted(set(k[1] for k in all_keys))
        x = np.array(user_values)

        colors = plt.cm.tab10.colors
        bar_width = 0.8 / len(phase_values)
        offsets = np.linspace(-(len(phase_values)-1)*bar_width/2, (len(phase_values)-1)*bar_width/2, len(phase_values))

        plt.figure(figsize=(14, 6))
        for ci_idx, k_phases in enumerate(phase_values):
            avgs, errs, xs = [], [], []
            for u in user_values:
                key = ('kp', u, None, k_phases)
                # find matching key (slots may vary)
                match = [g for g in group_stats if g[0] == 'kp' and g[1] == u and g[3] == k_phases]
                if match:
                    mean, ci_val, _ = group_stats[match[0]]
                    if normalize:
                        mean /= u
                        ci_val /= u
                    avgs.append(mean)
                    errs.append(ci_val)
                    xs.append(u)
            if avgs:
                xs_arr = np.array(xs)
                plt.bar(
                    xs_arr + offsets[ci_idx], avgs, width=bar_width,
                    yerr=errs, align='center', capsize=3,
                    label=f"k={k_phases}", color=colors[ci_idx % len(colors)],
                    alpha=0.8, ecolor='black',
                )
    else:
        # var-users or var-load: 2 variants (1p = P1-IRSA, 2p = FIT-IRSA)
        all_users = sorted(set(k[2] for k in group_stats))
        x = np.array(all_users)
        bar_width = 0.35
        offsets_map = {'2p': -bar_width/2, '1p': bar_width/2}
        style_map = {
            '2p': dict(alpha=0.5, edgecolor='none', label='FIT-IRSA (2-phase)'),
            '1p': dict(alpha=0.9, edgecolor='k', label='P1-IRSA (1-phase)'),
        }
        colors_map = {'2p': 'C0', '1p': 'C1'}

        plt.figure(figsize=(14, 6))
        for variant in ('2p', '1p'):
            prefix = 'load' if sweep == 'var-load' else ''
            avgs, errs, xs = [], [], []
            for u in all_users:
                match = [g for g in group_stats if g[0] == variant and g[1] == prefix and g[2] == u]
                if match:
                    mean, ci_val, _ = group_stats[match[0]]
                    if normalize:
                        mean /= u
                        ci_val /= u
                    avgs.append(mean)
                    errs.append(ci_val)
                    xs.append(u)
            if avgs:
                xs_arr = np.array(xs)
                plt.bar(
                    xs_arr + offsets_map[variant], avgs, width=bar_width,
                    yerr=errs, align='center', capsize=3,
                    color=colors_map[variant], ecolor='black',
                    **style_map[variant],
                )

    plt.xlabel("Number of Users")
    ylabel = "Throughput / Users" if normalize else "Mean Decoded Users"
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
    plt.show()


#%%
# ============================================================
# Per-seed CI bar chart on new-format results
# ============================================================

def plot_throughput_per_seed(
    sweep,
    results_dir='results/new',
    normalize=False,
    confidence=0.99,
    last_frac=0.1,
    xlim=None,
    ylim=None,
    out=None,
):
    """Bar chart matching conference paper style: one bar cluster per seed."""
    all_runs = collect_runs(results_dir)

    if sweep == 'var-users':
        runs = [r for r in all_runs if r['prefix'] == '' and r['variant'] in ('1p', '2p')]
    elif sweep == 'var-load':
        runs = [r for r in all_runs if r['prefix'] == 'load' and r['variant'] in ('1p', '2p')]
    elif sweep == 'var-users-kphase':
        runs = [r for r in all_runs if r['variant'] == 'kp']
    else:
        raise ValueError(f"Unknown sweep: {sweep!r}")

    seeds = sorted(set(r['seed'] for r in runs))
    all_users = sorted(set(r['users'] for r in runs))
    prefix = 'load' if sweep == 'var-load' else ''

    bar_width = 0.18
    offsets = np.linspace(-bar_width * 1.5, bar_width * 1.5, len(seeds))
    x = np.array(all_users)
    colors = plt.cm.tab10.colors

    plt.figure(figsize=(14, 6))

    for variant, zorder, alpha, edge in [('2p', 1, 0.4, 'none'), ('1p', 2, 0.85, 'k')]:
        for s_idx, seed in enumerate(seeds):
            avgs, lowers, uppers, xs = [], [], [], []
            for u in all_users:
                match = [r for r in runs if r['variant'] == variant and r['seed'] == seed and r['users'] == u and r['prefix'] == prefix]
                if not match:
                    avgs.append(np.nan); lowers.append(np.nan); uppers.append(np.nan); xs.append(u)
                    continue
                try:
                    log_path = find_jsonl_file(match[0]['path'])
                    data = load_jsonl(log_path)
                    n_tail = max(1, int(len(data) * last_frac))
                    tail = data[-n_tail:]
                    all_vals = np.concatenate([np.array(r['decoded_array']) for r in tail])
                    mean = all_vals.mean()
                    std = all_vals.std(ddof=1)
                    n = len(all_vals)
                    ci = t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n) if n > 1 else 0.0
                    if normalize:
                        mean /= u; ci /= u
                    avgs.append(mean); lowers.append(mean - ci); uppers.append(mean + ci); xs.append(u)
                except Exception as e:
                    print(f"  Skipping {match[0]['path']}: {e}")
                    avgs.append(np.nan); lowers.append(np.nan); uppers.append(np.nan); xs.append(u)

            avgs = np.array(avgs)
            yerr_lo = avgs - np.array(lowers)
            yerr_hi = np.array(uppers) - avgs
            lbl = f"Seed {seed}, {'2P' if variant == '2p' else '1P'}"
            plt.bar(
                x + offsets[s_idx], avgs, width=bar_width * (1.15 if variant == '2p' else 1.0),
                yerr=[yerr_lo, yerr_hi], align='center', alpha=alpha, ecolor='black', capsize=4,
                label=lbl, color=colors[s_idx % len(colors)], zorder=zorder, edgecolor=edge,
            )

    plt.xlabel("Number of Users")
    plt.ylabel("Throughput / Users" if normalize else "Mean Decoded Users")
    plt.xticks(x)
    plt.grid(True, axis='y')
    plt.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
    plt.show()


#%%
# ============================================================
# CLI entry point
# ============================================================

def _parse_args_cli():
    p = argparse.ArgumentParser(description="Plot throughput vs users/load from sweep results")
    p.add_argument('--ci-mode', choices=['per-seed', 'across-seed'], default='across-seed',
                   help='CI aggregation mode (default: across-seed for journal version)')
    p.add_argument('--sweep', choices=['var-users', 'var-load', 'var-users-kphase'], default='var-users',
                   help='Which sweep to plot')
    p.add_argument('--results-dir', default='results/new',
                   help='Directory containing run subdirectories (default: results/new)')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize throughput by number of users')
    p.add_argument('--out', default=None,
                   help='Save figure to this path (PDF recommended)')
    p.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('XMIN', 'XMAX'))
    p.add_argument('--ylim', nargs=2, type=float, default=None, metavar=('YMIN', 'YMAX'))
    p.add_argument('--legacy', action='store_true',
                   help='Run legacy (conference paper) plots from results/res-long/')
    return p.parse_args()


def main():
    args = _parse_args_cli()

    if args.legacy:
        plot_throughtput_vs_users_slots(
            normalize=False, prefix="-load", nb_slots=20,
            xlim=(10.5, 31), ylim=(10, 12.5),
            fig_file_name="throughput-vs-users-20slots.pdf",
        )
        plot_throughtput_vs_users_slots(
            normalize=True, prefix="", nb_slots=None,
            xlim=None, ylim=(0.40, 0.75),
            fig_file_name="throughput-vs-users-and-slots.pdf",
        )
        return

    kwargs = dict(
        sweep=args.sweep,
        results_dir=args.results_dir,
        normalize=args.normalize,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        out=args.out,
    )

    if args.ci_mode == 'across-seed':
        plot_throughput_across_seeds(**kwargs)
    else:
        plot_throughput_per_seed(**kwargs)


if __name__ == '__main__':
    main()
