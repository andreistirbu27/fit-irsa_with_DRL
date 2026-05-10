"""Reorganise results/new/ into per-sweep group folders.

Only moves run dirs that contain policy_final.pt (i.e. training completed).
Running/queued jobs write to flat results/new/ and are left untouched;
re-run this script after all jobs finish to migrate the remainder.

For each detected sweep group it:
  - Creates results/new/<group>/
  - Writes results/new/<group>/sweep_config.json  (shared hyperparams)
  - Moves each finished run dir into the group folder

Sweep groups detected:
  res-1p-var-users   — res-1p-u*-s*-seed*
  res-2p-var-users   — res-2p-u*-s*-seed*  (no "load" prefix)
  res-2p-var-load    — res-2p-load-u*-s*-seed*
  res-1p-ppo-var-users
  res-2p-ppo-var-users
  res-2p-2x64-var-users
  res-kp-var-users   — res-kp-* (all k values together)

Usage (dry run):
    python scripts/migrate_results.py --dry-run

Usage (for real):
    python scripts/migrate_results.py
"""

import argparse
import json
import os
import re
import shutil

RESULTS_DIR = os.path.join("results", "new")

# Per-run config keys that vary across runs — strip these from sweep_config.json
_VARYING_KEYS = {"num_users", "num_slots", "seed", "result_dir", "prefix"}


def detect_group(name):
    """Map a run directory name to a sweep group name, or None if unrecognised."""
    if re.match(r"res-2p-load-u\d+-s\d+-seed\d+", name):
        return "res-2p-var-load"
    if re.match(r"res-1p-load-u\d+-s\d+-seed\d+", name):
        return "res-1p-var-load"
    if re.match(r"res-2p-ppo-u\d+-s\d+-seed\d+", name):
        return "res-2p-ppo-var-users"
    if re.match(r"res-2p-2x64-u\d+-s\d+-seed\d+", name):
        return "res-2p-2x64-var-users"
    if re.match(r"res-2p-u\d+-s\d+-seed\d+", name):
        return "res-2p-var-users"
    if re.match(r"res-1p-ppo-u\d+-s\d+-seed\d+", name):
        return "res-1p-ppo-var-users"
    if re.match(r"res-1p-u\d+-s\d+-seed\d+", name):
        return "res-1p-var-users"
    if re.match(r"res-kp-u\d+", name):
        return "res-kp-var-users"
    return None


def load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def shared_config(configs):
    """Return the intersection of hyperparams that are identical across all configs."""
    if not configs:
        return {}
    keys = set(configs[0].keys()) - _VARYING_KEYS
    for cfg in configs[1:]:
        keys &= set(cfg.keys()) - _VARYING_KEYS
    result = {}
    for k in sorted(keys):
        vals = {str(c.get(k)) for c in configs}
        if len(vals) == 1:
            result[k] = configs[0][k]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without moving anything")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir
    dry = args.dry_run

    if dry:
        print("DRY RUN — nothing will be moved\n")

    # Collect finished runs grouped by sweep
    groups: dict[str, list[str]] = {}
    skipped_running = []

    for name in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, name)
        if not os.path.isdir(full):
            continue
        group = detect_group(name)
        if group is None:
            continue
        if not os.path.exists(os.path.join(full, "policy_final.pt")):
            skipped_running.append(name)
            continue
        groups.setdefault(group, []).append(name)

    if skipped_running:
        print(f"Skipping {len(skipped_running)} dirs without policy_final.pt "
              f"(still running or queued):\n  {skipped_running[:5]}"
              f"{'...' if len(skipped_running) > 5 else ''}\n")

    for group, run_names in sorted(groups.items()):
        group_dir = os.path.join(results_dir, group)
        print(f"\n{'[DRY] ' if dry else ''}Group: {group}  ({len(run_names)} runs)")

        # Build sweep_config.json from shared hyperparams
        configs = [load_config(os.path.join(results_dir, n)) for n in run_names]
        sweep_cfg = shared_config(configs)
        sweep_cfg_path = os.path.join(group_dir, "sweep_config.json")

        if not dry:
            os.makedirs(group_dir, exist_ok=True)
            with open(sweep_cfg_path, "w") as f:
                json.dump(sweep_cfg, f, indent=2)
            print(f"  Wrote {sweep_cfg_path}")
        else:
            print(f"  Would write {sweep_cfg_path}: {sweep_cfg}")

        # Move run dirs and drop now-redundant per-run config.json
        # (everything in it is either in sweep_config.json or encoded in the dirname).
        n_removed = 0
        for name in run_names:
            src = os.path.join(results_dir, name)
            dst = os.path.join(group_dir, name)
            if not dry:
                shutil.move(src, dst)
                run_cfg = os.path.join(dst, "config.json")
                if os.path.exists(run_cfg):
                    os.remove(run_cfg)
                    n_removed += 1
            print(f"  {'[move]' if not dry else '[would move]'} {name} → {group}/{name}")
        if n_removed:
            print(f"  Removed {n_removed} per-run config.json (redundant with sweep_config.json + dirname)")

    print("\nDone." if not dry else "\nDry run complete.")


if __name__ == "__main__":
    main()
