"""Logging and model checkpoint helpers shared across training scripts."""
import glob
import json
import os
import re
import sys

import torch

try:
    import gzip
except ImportError:
    gzip = None


# Order matters: longer variant prefixes must come before their substrings
# (e.g. '2p-ppo' before '2p') so the parser doesn't mis-tag.
_KNOWN_VARIANTS = ['2p-ppo', '1p-ppo', '2p-2x64', 'kp', '2p', '1p']


def _parse_run_name(name):
    """Parse a run directory basename into a metadata dict.

    Recognises names of the form `res-<variant>[-<prefix>]-u<N>-s<M>[-k<K>]-seed<S>`.
    Returns the cfg fields encoded in the name, or None if unrecognised.
    """
    if not name.startswith('res-'):
        return None
    rest = name[4:]
    variant = None
    for v in _KNOWN_VARIANTS:
        if rest.startswith(v + '-'):
            variant = v
            rest = rest[len(v) + 1:]
            break
    if variant is None:
        return None
    prefix = ''
    m = re.match(r'^([a-z][a-z0-9]*)-(u\d)', rest)
    if m and not m.group(1).startswith('u'):
        prefix = m.group(1)
        rest = rest[len(prefix) + 1:]
    m = re.match(r'^u(\d+)-s(\d+)(-.+)?$', rest)
    if not m:
        return None
    num_users = int(m.group(1))
    num_slots = int(m.group(2))
    tail = m.group(3) or ''
    out = {
        'variant': variant,
        'prefix': prefix,
        'num_users': num_users,
        'num_slots': num_slots,
    }
    km = re.search(r'-k(\d+)', tail)
    if km:
        out['num_phases'] = int(km.group(1))
    elif variant == 'kp':
        out['num_phases'] = 2  # default for kp dirs without explicit -k suffix
    sm = re.search(r'-seed(\d+)$', tail)
    if sm:
        out['seed'] = int(sm.group(1))
    return out


def resolve_run_config(result_dir):
    """Return the full config for a run by combining dirname + sweep_config.json.

    Sources, in order of precedence (later overrides earlier):
      1. Local config.json (if present — for ungrouped runs that haven't been
         migrated yet, the trainer's full config sits here).
      2. Parent dir's sweep_config.json (shared hyperparams for the sweep group).
      3. Fields parsed from the run dir's name (variant, prefix, users, slots,
         num_phases, seed).

    Raises FileNotFoundError only if NONE of the three sources yield anything
    useful (a malformed run dir with no config and no parseable name).
    """
    cfg = {}
    local_cfg = os.path.join(result_dir, 'config.json')
    if os.path.exists(local_cfg):
        with open(local_cfg) as f:
            cfg.update(json.load(f))
    parent_cfg = os.path.join(os.path.dirname(os.path.abspath(result_dir)),
                              'sweep_config.json')
    if os.path.exists(parent_cfg):
        with open(parent_cfg) as f:
            cfg.update(json.load(f))
    name_fields = _parse_run_name(os.path.basename(os.path.abspath(result_dir)))
    if name_fields:
        cfg.update(name_fields)
    cfg.setdefault('result_dir', result_dir)
    if not cfg or 'num_users' not in cfg or 'num_slots' not in cfg:
        raise FileNotFoundError(
            f"Could not resolve config for {result_dir}: no parseable name, "
            f"no local config.json, no parent sweep_config.json with enough info."
        )
    return cfg


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


def load_model(result_dir, model_factory, which="final", device=None):
    """Load a model checkpoint from a run directory.

    model_factory(cfg) must return an nn.Module with the correct architecture.
    The loaded state dict is applied with strict=True.

    The cfg is resolved from dirname + parent sweep_config.json (falling back
    to a local config.json if present) — see resolve_run_config.
    """
    cfg = resolve_run_config(result_dir)

    model = model_factory(cfg)

    if which == "final":
        model_path = os.path.join(result_dir, "policy_final.pt")
    elif isinstance(which, (int, str)) and str(which).isdigit():
        model_path = os.path.join(result_dir, f"policy_epoch{which}.pt")
    else:
        raise ValueError(f"Invalid 'which' argument: {which}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return model, cfg


def cleanup_old_models(result_dir, keep_last=2):
    """Keep only the last `keep_last` policy_epoch*.pt files. policy_final.pt is untouched."""
    pattern = os.path.join(result_dir, "policy_epoch*.pt")
    files = glob.glob(pattern)

    def extract_epoch(f):
        base = os.path.basename(f)
        try:
            return int(base.replace("policy_epoch", "").replace(".pt", ""))
        except Exception:
            return -1

    files_epochs = [(f, extract_epoch(f)) for f in files]
    files_epochs = sorted([fe for fe in files_epochs if fe[1] >= 0], key=lambda x: x[1])
    if keep_last > 0 and len(files_epochs) > keep_last:
        for f, _ in files_epochs[:-keep_last]:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: could not remove old model {f}: {e}", file=sys.stderr)
