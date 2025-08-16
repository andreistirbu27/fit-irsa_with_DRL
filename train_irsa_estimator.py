#%%
import os
import json
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
import irsa_two_phases

# ---------------------------
# Utilities: load logs/config
# ---------------------------

def find_jsonl_file(run_dir: str) -> str:
    """
    Return the path to a .jsonl or .jsonl.gz file in run_dir.
    Raises FileNotFoundError if none is found.
    """
    for fname in os.listdir(run_dir):
        if fname.endswith('.jsonl') or fname.endswith('.jsonl.gz'):
            return os.path.join(run_dir, fname)
    raise FileNotFoundError(f"No .jsonl or .jsonl.gz file found in {run_dir}")

def load_jsonl(path: str):
    """
    Load a jsonl or jsonl.gz file and return a list of dicts.
    """
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]

def load_config(run_dir: str) -> dict:
    """
    Load config.json from run_dir.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json found in {run_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)



def build_actions_feedback_dataset(
    data,
    num_slots_cfg,
    num_users_cfg,
    use_energy_feedback=False,
    verbose=False
):
    """
    Build the dataset of (one_hot_actions, feedback_vectors) from log data.
    Returns (one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions)
    """
    all_one_hot_actions = []   # list of [U, S] float32 tensors
    all_feedback_vectors = []  # list of [3S] (or alternative) float32 tensors
    num_samples = 0
    num_records_with_actions = 0

    for rec_idx, rec in enumerate(data):
        if "actions_r1" not in rec:
            # This record doesn't include action logs (requires --log-action at train time).
            continue
        actions_r1 = rec["actions_r1"]  # List[simulation][user] -> List[int slots]
        if not actions_r1:
            continue

        num_records_with_actions += 1

        # Sanity: infer users from first simulation and compare to config
        sim0 = actions_r1[0]
        U = len(sim0)
        S = num_slots_cfg
        if U != num_users_cfg and verbose:
            print(f"[warn] record {rec_idx}: users in log={U} != config={num_users_cfg} (continuing with U={U})")

        # Iterate simulations in this record
        for sim_idx, actions_list in enumerate(actions_r1):
            # actions_list is List[List[int]] with length U
            # Build one-hot [U, S]
            one_hot = torch.zeros(U, S, dtype=torch.float32)
            # Build SIC input structure: List[(r, slot_list)] per user
            sic_actions = []
            for u, slots in enumerate(actions_list):
                # keep only valid, unique slots
                clean_slots = sorted({s for s in slots if 0 <= s < S})
                if clean_slots:
                    one_hot[u, torch.tensor(clean_slots, dtype=torch.long)] = 1.0
                sic_actions.append((len(clean_slots), clean_slots))

            # Run SIC and construct feedback vector
            decoded_users, feedback_indices, energy_per_slot = irsa_two_phases.run_sic_simulation(
                sic_actions, S, return_feedback_indices=True
            )
            fb_vec = irsa_two_phases.feedback_indices_to_vector(
                feedback_indices, S,
                energy_per_slot=energy_per_slot,
                use_energy=use_energy_feedback
            ).to(dtype=torch.float32)

            all_one_hot_actions.append(one_hot)
            all_feedback_vectors.append(fb_vec)
            num_samples += 1

    if num_samples == 0:
        raise RuntimeError(
            "No samples built. Did you train with --log-action so that actions_r1 "
            "are present in the log?"
        )

    if verbose:
        print(f"Records with actions: {num_records_with_actions}/{len(data)}")
        print(f"Total samples: {num_samples}")

    # Shape consistency checks before stacking
    U0, S0 = all_one_hot_actions[0].shape
    F0 = all_feedback_vectors[0].numel()
    for i, (a, f) in enumerate(zip(all_one_hot_actions, all_feedback_vectors)):
        if a.shape != (U0, S0):
            raise ValueError(f"one_hot[{i}] shape {a.shape} != {(U0, S0)}")
        if f.numel() != F0:
            raise ValueError(f"feedback[{i}] length {f.numel()} != {F0}")

    one_hot_actions = torch.stack(all_one_hot_actions)      # [N, U, S]
    feedback_vectors = torch.stack(all_feedback_vectors)    # [N, 3S] (or alternative)

    return one_hot_actions, feedback_vectors, U0, S0, num_samples, num_records_with_actions

# ----------
# Parameters
# ----------
RUN_DIR = "res-1p-u10-sl10-e1000-b1000-s100"  # adjust as needed
RUN_DIR = "res-1p-u10-s10-e1000-b1000-s100"
DATASET_OUT = os.path.join(RUN_DIR, "actions_feedback_dataset.pt")
USE_ENERGY_FEEDBACK = False   # if True, use per-slot counts instead of undecoded one-hot
VERBOSE = True

# -----------------
# Load config (always), but only load log if building dataset
config = load_config(RUN_DIR)
num_slots_cfg = int(config["num_slots"])
num_users_cfg = int(config["num_users"])

# -----------------------------------------------------------
# 1) Build dataset: actions (one-hot) -> feedback vectors
#    Only build and save if DATASET_OUT does not exist
# -----------------------------------------------------------
def load_or_build_actions_feedback_dataset(
    run_dir,
    dataset_out,
    num_slots_cfg,
    num_users_cfg,
    use_energy_feedback=False,
    verbose=True
):
    """
    Loads the actions-feedback dataset from disk if it exists, otherwise builds it from log data.
    Returns: one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions
    """
    if os.path.exists(dataset_out):
        if verbose:
            print(f"Loading existing dataset: {dataset_out}")
        saved = torch.load(dataset_out)
        one_hot_actions = saved["one_hot_actions"]
        feedback_vectors = saved["feedback_vectors"]
        U = one_hot_actions.shape[1]
        S = one_hot_actions.shape[2]
        num_samples = one_hot_actions.shape[0]
        num_records_with_actions = None  # Not tracked when loading
        if verbose:
            print(f"Loaded dataset: {dataset_out}")
            print(f"one_hot_actions: {tuple(one_hot_actions.shape)}  feedback_vectors: {tuple(feedback_vectors.shape)}")
    else:
        log_file = find_jsonl_file(run_dir)
        data = load_jsonl(log_file)
        if verbose:
            print(f"Loaded {len(data)} log records from {log_file}")
            print(f"Config: users={num_users_cfg}, slots={num_slots_cfg}")

        one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions = build_actions_feedback_dataset(
            data,
            num_slots_cfg,
            num_users_cfg,
            use_energy_feedback=use_energy_feedback,
            verbose=verbose
        )

        os.makedirs(run_dir, exist_ok=True)
        torch.save(
            {"one_hot_actions": one_hot_actions, "feedback_vectors": feedback_vectors},
            dataset_out
        )
        if verbose:
            print(f"Saved dataset: {dataset_out}")
            print(f"one_hot_actions: {tuple(one_hot_actions.shape)}  feedback_vectors: {tuple(feedback_vectors.shape)}")
    return one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions

# Use the function to load or build the dataset
one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions = load_or_build_actions_feedback_dataset(
    RUN_DIR,
    DATASET_OUT,
    num_slots_cfg,
    num_users_cfg,
    use_energy_feedback=USE_ENERGY_FEEDBACK,
    verbose=VERBOSE
)

# -------------------------------------------------------------------
# 3) Define PyTorch Dataset and DataLoader (no model defined here)
# -------------------------------------------------------------------
class ActionsFeedbackDataset(Dataset):
    def __init__(self, one_hot_actions: torch.Tensor, feedback_vectors: torch.Tensor):
        """
        one_hot_actions: [N, U, S] float tensor
        feedback_vectors: [N, F] float tensor (F = 3*S if using default feedback)
        """
        if one_hot_actions.shape[0] != feedback_vectors.shape[0]:
            raise ValueError("Mismatched N between actions and feedback.")
        self.one_hot_actions = one_hot_actions
        self.feedback_vectors = feedback_vectors

    def __len__(self):
        return self.one_hot_actions.shape[0]

    def __getitem__(self, idx):
        return self.one_hot_actions[idx], self.feedback_vectors[idx]

# Example construction (kept minimal; adjust batch_size as needed)
dataset = ActionsFeedbackDataset(one_hot_actions, feedback_vectors)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

if VERBOSE:
    first_batch = next(iter(dataloader))
    xb, yb = first_batch
    print(f"[dataloader] batch x: {tuple(xb.shape)}, y: {tuple(yb.shape)}")

#%%

# --- 4) Transformer model (encoder-only) and training loop: one-hot actions -> feedback ---

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

# Derive shapes from the dataset you built above
N, U, S = one_hot_actions.shape   # [N, num_users, num_slots]
F = feedback_vectors.shape[1]     # feedback_dim (typically 3*S)

# -----------------------------
# Model: Action -> Feedback (2D learned embeddings for (user, slot))
# -----------------------------
class ActionToFeedbackTransformer(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_slots: int,
        feedback_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even to split across user/slot embeddings"

        self.num_users = num_users
        self.num_slots = num_slots
        self.feedback_dim = feedback_dim
        self.d_model = d_model

        # Learned 2D embeddings: user and slot
        self.user_embed = nn.Embedding(num_users, d_model // 2)
        self.slot_embed = nn.Embedding(num_slots, d_model // 2)

        # Project scalar one-hot (0/1) into d_model channels
        self.input_proj = nn.Linear(1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: flatten [U*S, d_model] and map to feedback vector
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),                                # [B, U*S*d_model]
            nn.Linear(num_users * num_slots * d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feedback_dim)
        )

        # Register static (user,slot) index grids as buffers so they move with the model's device
        user_idx = torch.arange(num_users).view(1, num_users, 1).expand(1, num_users, num_slots)  # [1,U,S]
        slot_idx = torch.arange(num_slots).view(1, 1, num_slots).expand(1, num_users, num_slots)  # [1,U,S]
        self.register_buffer("user_idx_grid", user_idx, persistent=False)
        self.register_buffer("slot_idx_grid", slot_idx, persistent=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, U, S] one-hot (0/1) tensor.
        Returns: [B, F] predicted feedback vector.
        """
        B, U, S = x.shape
        assert U == self.num_users and S == self.num_slots, "Input shape does not match model's (U,S)."

        # Embeddings for (user, slot) positions
        user_emb = self.user_embed(self.user_idx_grid.expand(B, -1, -1))  # [B,U,S,d/2]
        slot_emb = self.slot_embed(self.slot_idx_grid.expand(B, -1, -1))  # [B,U,S,d/2]
        pos_emb = torch.cat([user_emb, slot_emb], dim=-1)                  # [B,U,S,d]

        # Project scalar inputs to d_model and add positional embeddings
        x_proj = self.input_proj(x.unsqueeze(-1))                          # [B,U,S,d]
        x_enc = x_proj + pos_emb

        # Flatten grid to sequence for the encoder: [B, U*S, d]
        x_seq = x_enc.view(B, U * S, self.d_model)

        # Encoder
        h = self.encoder(x_seq)                                            # [B, U*S, d]

        # Head -> feedback vector
        out = self.head(h)                                                 # [B, F]
        return out


# -----------------------------
# Train/val split + DataLoaders
# -----------------------------
if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")


val_frac = 0.1
val_size = max(1, int(len(dataset) * val_frac))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

# -----------------------------
# Instantiate model, loss, optim
# -----------------------------
model = ActionToFeedbackTransformer(
    num_users=U,
    num_slots=S,
    feedback_dim=F,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.1,
).to(device)

# Use MSE since feedback can be binary (0/1) or counts if energy feedback was used
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# -----------------------------
# Training loop
# -----------------------------
epochs = 50
grad_clip = 1.0
BATCH_LIMIT = 10  # Limit to this many batches per epoch

def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    n = 0
    batch_count = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device)            # [B,U,S]
            yb = yb.to(device)            # [B,F]
            pred = model(xb)              # [B,F]
            loss = criterion(pred, yb)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            batch_count += 1
            if batch_count >= BATCH_LIMIT:
                break
    return total_loss / max(1, n)

best_val = float("inf")
best_state = None

for epoch in range(1, epochs + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step()

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        print(f"[{epoch:03d}/{epochs}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

# (Optional) restore best weights
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)
    print(f"Restored best model with val_loss={best_val:.6f}")
