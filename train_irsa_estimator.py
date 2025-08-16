import os
import json
import gzip
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import irsa_two_phases
import math
import torch.nn as nn
import torch.optim as optim

DEFAULT_KEEP_LAST_MODELS = 2
DEFAULT_SEED = 42

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
# Device and seed utilities
# -----------------------------
def get_device():
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Model saving and management
# -----------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)

def manage_saved_models(model_dir, keep_last):
    """
    Remove all but the last N models (sorted by epoch in filename)
    """
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model-epoch") and f.endswith(".pt")]
    if len(model_files) <= keep_last:
        return
    # Extract epoch number
    def get_epoch(f):
        try:
            return int(f.split("model-epoch")[1].split(".pt")[0])
        except Exception:
            return -1
    model_files = sorted(model_files, key=get_epoch)
    for f in model_files[:-keep_last]:
        try:
            os.remove(os.path.join(model_dir, f))
        except Exception as e:
            print(f"Warning: could not remove {f}: {e}")

# -----------------------------
# Training/validation loop
# -----------------------------
def run_epoch(model, loader, criterion, optimizer, device, train, grad_clip, batch_limit, scheduler=None, step_per_batch=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    n = 0
    batch_count = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if step_per_batch and scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            batch_count += 1
            if batch_count >= batch_limit:
                break
    return total_loss / max(1, n)

def train(
    args,
    one_hot_actions,
    feedback_vectors,
    U,
    S,
    F,
    run_dir,
    dataset,
):
    """
    Main training loop, model setup, logging, and saving.
    """
    # -----------------------------
    # Train/val split + DataLoaders
    # -----------------------------
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    steps_per_epoch = len(train_loader)
    steps_per_epoch_effective = min(steps_per_epoch, args.batches_per_epoch)

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
    ).to(args.device)

    # If resume is requested and model-best.pt exists, load weights
    if getattr(args, "resume", False):
        model_best_path = os.path.join(run_dir, "model-best.pt")
        if os.path.exists(model_best_path):
            print(f"[resume] Loading model weights from {model_best_path}")
            state_dict = torch.load(model_best_path, map_location=args.device)
            model.load_state_dict(state_dict)
        else:
            print(f"[resume] model-best.pt not found in {run_dir}, starting from scratch.")


    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if not args.warmup_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    else:
        total_steps = args.epochs * steps_per_epoch_effective
        warmup_steps = max(3 * steps_per_epoch_effective, int(0.03 * total_steps))
        eta_min = 1e-6
        base_lr = args.lr

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min / base_lr + 0.5 * (1 - eta_min / base_lr) * (1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val = float("inf")
    best_state = None
    model_save_dir = run_dir
    train_log_path = os.path.join(run_dir, "training-generator.jsonl")

    # -----------------------------
    # Training loop
    # -----------------------------
    with open(train_log_path, "w") as train_log_f:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss = run_epoch(
                model, train_loader, criterion, optimizer, args.device, train=True,
                grad_clip=args.grad_clip, batch_limit=args.batches_per_epoch,
                scheduler=scheduler if args.warmup_cosine else None,
                step_per_batch=args.warmup_cosine
            )

            val_loss = run_epoch(
                model, val_loader, criterion, optimizer, args.device, train=False,
                grad_clip=args.grad_clip, batch_limit=args.batches_per_epoch
            )

            if not args.warmup_cosine:
                scheduler.step()
            epoch_end = time.time()
            duration = epoch_end - epoch_start

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            current_lr = optimizer.param_groups[0]['lr']
            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "duration_sec": duration,
                "best_val_loss": best_val
            }
            train_log_f.write(json.dumps(log_entry) + "\n")
            train_log_f.flush()

            print(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.2e}  duration={duration:.2f}s")

            # Save model at intervals
            if args.epoch_save_interval > 0 and (epoch % args.epoch_save_interval == 0 or epoch == args.epochs):
                model_path = os.path.join(model_save_dir, f"model-epoch{epoch}.pt")
                save_model(model, model_path)
                print(f"Saved model at {model_path}")
                manage_saved_models(model_save_dir, args.keep_last_models)

    # Restore best weights and save final model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(args.device)
        print(f"Restored best model with val_loss={best_val:.6f}")
        best_model_path = os.path.join(model_save_dir, "model-best.pt")
        save_model(model, best_model_path)
        print(f"Saved best model at {best_model_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS, help='Keep only the last X saved models (default=2)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Random seed (default={DEFAULT_SEED})')
    parser.add_argument('--run-dir', type=str, default="res-1p-u10-s10-e1000-b1000-s100", help='Result directory (default: res-1p-u10-s10-e1000-b1000-s100)')
    parser.add_argument('--use-energy-feedback', action='store_true', help='Use per-slot counts instead of undecoded one-hot')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Limit to this many batches per epoch')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--resume', action='store_true', help='Resume training from model-best.pt if it exists')
    parser.add_argument('--warmup-cosine', action='store_true', help='Per-batch linear warmup (~3% or 3 epochs) then cosine decay to eta_min=1e-6.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Determine run directory
    run_dir = args.result_dir if args.result_dir is not None else args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    DATASET_OUT = os.path.join(run_dir, "actions_feedback_dataset.pt")

    # -----------------
    # Load config (always), but only load log if building dataset
    config = load_config(run_dir)
    num_slots_cfg = int(config["num_slots"])
    num_users_cfg = int(config["num_users"])

    # -----------------------------------------------------------
    # 1) Build dataset: actions (one-hot) -> feedback vectors
    #    Only build and save if DATASET_OUT does not exist
    # -----------------------------------------------------------
    one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions = load_or_build_actions_feedback_dataset(
        run_dir,
        DATASET_OUT,
        num_slots_cfg,
        num_users_cfg,
        use_energy_feedback=args.use_energy_feedback,
        verbose=args.verbose
    )

    # -------------------------------------------------------------------
    # 3) Define PyTorch Dataset and DataLoader (no model defined here)
    # -------------------------------------------------------------------
    dataset = ActionsFeedbackDataset(one_hot_actions, feedback_vectors)
    if args.verbose:
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
        first_batch = next(iter(dataloader))
        xb, yb = first_batch
        print(f"[dataloader] batch x: {tuple(xb.shape)}, y: {tuple(yb.shape)}")

    N, U, S = one_hot_actions.shape   # [N, num_users, num_slots]
    F = feedback_vectors.shape[1]     # feedback_dim (typically 3*S)

    args.device = get_device()
    print(f"Using device: {args.device}")

    train(
        args=args,
        one_hot_actions=one_hot_actions,
        feedback_vectors=feedback_vectors,
        U=U,
        S=S,
        F=F,
        run_dir=run_dir,
        dataset=dataset,
    )

if __name__ == "__main__":
    main()
