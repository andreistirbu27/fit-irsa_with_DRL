"""IRSA simulation primitives: per-user sampling, SIC decoding, feedback encoding."""
import torch


def sample_actions_user(logits_user):
    """Bernoulli per slot for one user; returns ((r, [slots]), log_prob, action_tensor)."""
    d = torch.distributions.Bernoulli(logits=logits_user)
    a = d.sample()
    lp = d.log_prob(a).sum()
    slots = torch.where(a == 1)[0].tolist()
    return (len(slots), slots), lp, a


def run_sic_simulation(actions, num_slots, return_feedback_indices=False):
    """SIC over a single round; optionally returns (decoded, empty, undecoded) slot index lists."""
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
        decoded_idx = sorted(list(final_empty - initial_empty))
        empty_idx = sorted(list(initial_empty))
        undec_idx = sorted([i for i in range(num_slots)
                            if len(slots_init[i]) > 0 and len(slots[i]) > 0])
        return decoded_users, [decoded_idx, empty_idx, undec_idx]
    return decoded_users


def feedback_indices_to_vector(feedback_indices, num_slots):
    decoded_idx, empty_idx, undec_idx = feedback_indices
    d = torch.zeros(num_slots)
    e = torch.zeros(num_slots)
    u = torch.zeros(num_slots)
    if decoded_idx:
        d[decoded_idx] = 1
    if empty_idx:
        e[empty_idx] = 1
    if undec_idx:
        u[undec_idx] = 1
    return torch.cat([d, e, u], dim=0)


def sic_decode(actions, total_slots):
    """SIC over a flat slot space (used for two-phase concatenated decoding)."""
    slots = [[] for _ in range(total_slots)]
    for user_id, (r, slot_list) in enumerate(actions):
        for s in slot_list[:r]:
            slots[s].append(user_id)

    decoded_users = set()
    progress = True
    while progress:
        progress = False
        for s in range(total_slots):
            if len(slots[s]) == 1:
                u = slots[s][0]
                if u not in decoded_users:
                    decoded_users.add(u)
                    for t in range(total_slots):
                        if u in slots[t]:
                            slots[t].remove(u)
                    progress = True
    return decoded_users
