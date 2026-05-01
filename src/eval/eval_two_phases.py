#%%

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.irsa_common.sic import sample_actions_user


def _load_two_phase_model(result_dir, which="final", device=None):
    """Dispatch to the correct loader based on the run's config.json."""
    import json
    config_path = os.path.join(result_dir, "config.json")
    with open(config_path) as f:
        cfg_peek = json.load(f)
    if "clip_eps" in cfg_peek:
        from src.train.irsa_two_phases_ppo import load_model_from_dir
    elif "num_layers" in cfg_peek:
        from src.train.irsa_2phase_2x64 import load_model_from_dir
    else:
        from src.train.irsa_two_phases import load_model_from_dir
    return load_model_from_dir(result_dir, which=which, device=device)


def prepare_forward_args(cfg, obs, feedback, prev_action):
    """
    Prepare the input tensor for the policy's forward method.

    Args:
        cfg (dict): Configuration dictionary with keys like 'num_slots', 'input_obs_dim'.
        obs (np.ndarray or torch.Tensor): Observation vector of shape [input_obs_dim].
        feedback (np.ndarray or torch.Tensor): Feedback vector of shape [3 * num_slots].
        prev_action (np.ndarray or torch.Tensor): Previous action vector of shape [num_slots].

    Returns:
        torch.Tensor: Concatenated input tensor of shape [input_obs_dim + 3*num_slots + num_slots].
    """
    import torch
    # Convert to torch tensors if needed
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        else:
            return torch.tensor(x, dtype=torch.float32)
    obs = to_tensor(obs)
    feedback = to_tensor(feedback)
    prev_action = to_tensor(prev_action)
    # Concatenate along last dimension
    x = torch.cat([obs, feedback, prev_action], dim=-1)
    return x


result_dir = "results/new/<edit-me>"  # point this at the run you want to inspect

if "<edit-me>" in result_dir:
    raise ValueError("Edit `result_dir` to point at a real run before executing this script.")

policy, cfg = _load_two_phase_model(result_dir)

policy.eval()
print("Model loaded from", result_dir)

import torch

dummy_obs = [0.0] * cfg['input_obs_dim']
dummy_feedback = [0] * (3 * cfg['num_slots'])
dummy_prev_action = [0] * cfg['num_slots']
model_input = prepare_forward_args(cfg, dummy_obs, dummy_feedback, dummy_prev_action)
with torch.no_grad():
    output = policy.forward(model_input)
    logits = output[0] if isinstance(output, tuple) else output  # PPO returns (logits, value)
    probs = torch.sigmoid(logits)
    print(probs)
    _, _, action = sample_actions_user(logits)
print("Sampled action:", action.numpy())


#%%
