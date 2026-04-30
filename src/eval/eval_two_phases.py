#%%

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train import irsa_two_phases

# Load model using function from irsa_2phases


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


result_dir = "results/result-u4-s2"  # Change as needed


# Use the function from irsa_2phases to load the latest model
policy, cfg = irsa_two_phases.load_model_from_dir(result_dir)

policy.eval()
print("Model loaded from", result_dir)

import torch

model_input = prepare_forward_args(cfg, [0.1,1,0], 3*[0,0], [0,0])
with torch.no_grad():
    logits = policy.forward(model_input)
    probs = torch.sigmoid(logits)
    print(probs)
    # Sample Bernoulli decision for each slot
    action = torch.bernoulli(probs)
print("Sampled action (Bernoulli):", action.numpy())


#%%
