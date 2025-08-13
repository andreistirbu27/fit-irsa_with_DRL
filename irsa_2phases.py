import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Config ===
num_users = 15
num_slots = 9
input_obs_dim = 3
prev_action_dim = num_slots
feedback_dim = 3 * num_slots  # [decoded one-hot | empty one-hot | undecoded one-hot]
#user_status_dim = 1  # 1 = needs decoding, 0 = already decoded
input_dim = input_obs_dim + feedback_dim + prev_action_dim
hidden_dim = 128
epochs = 2000
batch_size = 50
learning_rate = 0.02


# === Single-user policy ===
class PolicyNetUser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots)   # logits per slot for THIS user
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(-1.4)          # optional: start sparser (sigmoid ~0.2)

    def forward(self, x):          # x: [input_dim]
        return self.net(x)         # [num_slots] logits


def sample_actions_user(logits_user):
    """Bernoulli per slot for one user; allows 0,1,2,... slots."""
    d = torch.distributions.Bernoulli(logits=logits_user)
    a = d.sample()  # [num_slots] {0,1}
    lp = d.log_prob(a).sum()  # scalar log-prob for this user
    slots = torch.where(a == 1)[0].tolist()
    return (len(slots), slots), lp, a  # (r,[slots]), logprob, tensor


# === SIC + feedback (decoded slots are those that became empty after SIC but were non-empty initially) ===
def run_sic_simulation(actions, return_feedback_indices=False):
    slots_init = [[] for _ in range(num_slots)]
    for user_id, (r, slot_list) in enumerate(actions):
        for s in slot_list[:r]:
            slots_init[s].append(user_id)
    initial_empty = {i for i in range(num_slots) if len(slots_init[i]) == 0}

    slots = [lst.copy() for lst in slots_init]
    decoded_users = set()
    progress = True
    while progress:
        progress = False
        for s in range(num_slots):
            if len(slots[s]) == 1:
                u = slots[s][0]
                if u not in decoded_users:
                    decoded_users.add(u)
                    for t in range(num_slots):
                        if u in slots[t]:
                            slots[t].remove(u)
                    progress = True

    if return_feedback_indices:
        final_empty = {i for i in range(num_slots) if len(slots[i]) == 0}
        decoded_idx = sorted(list(final_empty - initial_empty))  # became empty due to SIC
        empty_idx = sorted(list(initial_empty))  # empty from start
        undec_idx = sorted([i for i in range(num_slots)
                            if len(slots_init[i]) > 0 and len(slots[i]) > 0])  # still occupied
        return decoded_users, [decoded_idx, empty_idx, undec_idx]

    return decoded_users


def feedback_indices_to_vector(feedback_indices):
    decoded_idx, empty_idx, undec_idx = feedback_indices
    d = torch.zeros(num_slots);
    e = torch.zeros(num_slots);
    u = torch.zeros(num_slots)
    if decoded_idx: d[decoded_idx] = 1
    if empty_idx:   e[empty_idx] = 1
    if undec_idx:   u[undec_idx] = 1
    return torch.cat([d, e, u], dim=0)  # length = 3*num_slots


# === Training (apply policy per user) ===
def train(policy, optimizer,
          sparsity_r1_max=0.02, sparsity_r2_max=0.01,
          warmup_r1=400, warmup_r2=800):
    reward_history = []
    avg_unique_history = []
    frac_decR1_txR2_hist = []   # fraction: R1-decoded who still tx in R2

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        # ramped sparsity weights
        lam_r1 = sparsity_r1_max * min(1.0, epoch / float(warmup_r1))
        if epoch <= warmup_r1:
            lam_r2 = 0.0
        else:
            denom = max(1, warmup_r2 - warmup_r1)
            lam_r2 = sparsity_r2_max * min(1.0, (epoch - warmup_r1) / float(denom))

        batch_rewards, batch_log_probs, batch_uniques = [], [], []
        batch_frac_decR1_txR2 = []
        last_r1_act, last_r2_act = 0.0, 0.0

        for _ in range(batch_size):
            # per-user noise (keep or share, your call)
            obs_all = [torch.rand(input_obs_dim) for _ in range(num_users)]

            # -------- Round 1: per-user policy (feedback=0, prev_action=zeros) --------
            actions_r1, acts_bin_r1 = [], []
            lp_r1_total = 0.0
            for u in range(num_users):
                x1 = torch.cat([
                    obs_all[u],
                    torch.zeros(feedback_dim),       # no feedback yet
                    torch.zeros(prev_action_dim)     # no previous action yet
                ], dim=0)
                assert x1.numel() == input_dim
                logits_u = policy(x1)                                # [num_slots]
                cw_u, lp_u, a_u = sample_actions_user(logits_u)     # a_u: [num_slots] in {0,1}
                actions_r1.append(cw_u)
                acts_bin_r1.append(a_u)
                lp_r1_total = lp_r1_total + lp_u
            acts_bin_r1 = torch.stack(acts_bin_r1, dim=0)            # [num_users, num_slots]

            decoded_r1, fb_idx = run_sic_simulation(actions_r1, return_feedback_indices=True)

            # -------- Round 2: per-user policy (feedback + prev_action from R1) --------
            fb_vec = feedback_indices_to_vector(fb_idx)              # len = 3*num_slots
            actions_r2, acts_bin_r2 = [], []
            lp_r2_total = 0.0
            for u in range(num_users):
                prev_act_u = acts_bin_r1[u].float()                  # THIS user's R1 action (0/1 per slot)
                x2 = torch.cat([
                    obs_all[u],
                    fb_vec,
                    prev_act_u
                ], dim=0)
                assert x2.numel() == input_dim
                logits_u = policy(x2)
                cw_u, lp_u, a_u = sample_actions_user(logits_u)
                actions_r2.append(cw_u)
                acts_bin_r2.append(a_u)
                lp_r2_total = lp_r2_total + lp_u
            acts_bin_r2 = torch.stack(acts_bin_r2, dim=0)            # [num_users, num_slots]

            # -------- Reward: union of decodes across rounds --------
            decoded_r2 = run_sic_simulation(actions_r2)
            num_unique = len(decoded_r1.union(decoded_r2))
            reward = num_unique / num_users

            # -------- Ramped sparsity in BOTH rounds --------
            r1_activity = acts_bin_r1.float().mean().item()
            r2_activity = acts_bin_r2.float().mean().item()
            reward -= lam_r1 * r1_activity
            reward -= lam_r2 * r2_activity
            last_r1_act, last_r2_act = r1_activity, r2_activity

            # -------- Metric: frac(R1-decoded who transmit in R2) --------
            if len(decoded_r1) > 0:
                mask_dec = torch.zeros(num_users, dtype=torch.bool)
                mask_dec[list(decoded_r1)] = True
                r2_any = (acts_bin_r2.sum(dim=1) > 0)               # per-user bool
                frac_decR1_txR2 = r2_any[mask_dec].float().mean().item()
            else:
                frac_decR1_txR2 = float('nan')

            # accumulate
            batch_rewards.append(reward)
            batch_log_probs.append(lp_r1_total + lp_r2_total)
            batch_uniques.append(num_unique)
            batch_frac_decR1_txR2.append(frac_decR1_txR2)

        # REINFORCE with baseline
        baseline = np.mean(batch_rewards)
        total_loss = sum([-(r - baseline) * lp for r, lp in zip(batch_rewards, batch_log_probs)])
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # record epoch stats
        reward_history.append(baseline)
        avg_unique_history.append(np.mean(batch_uniques))
        frac_decR1_txR2_hist.append(np.nanmean(batch_frac_decR1_txR2))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: "
                  f"Avg Reward={baseline:.3f}, "
                  f"Avg unique={avg_unique_history[-1]:.2f}/{num_users}, "
                  f"frac(R1-decoded tx in R2)={frac_decR1_txR2_hist[-1]:.3f}, "
                  f"R1 act={last_r1_act:.3f} (λ1={lam_r1:.4f}), "
                  f"R2 act={last_r2_act:.3f} (λ2={lam_r2:.4f})")

    return reward_history, avg_unique_history, frac_decR1_txR2_hist


# === Run ===
policy = PolicyNetUser()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
rewards, avg_unique, frac_decR1_txR2 = train(policy, optimizer)
