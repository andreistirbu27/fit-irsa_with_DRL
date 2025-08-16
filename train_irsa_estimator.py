import os
import json
import gzip
import time
import argparse
import math
import datetime

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn

import irsa_two_phases

DEFAULT_KEEP_LAST_MODELS = 2
DEFAULT_SEED = 42

# ---------------------------
# Utilities: load logs/config
# ---------------------------

def find_jsonl_file(run_dir: str) -> str:
    for fname in os.listdir(run_dir):
        if fname.endswith('.jsonl') or fname.endswith('.jsonl.gz'):
            return os.path.join(run_dir, fname)
    raise FileNotFoundError(f"No .jsonl or .jsonl.gz file found in {run_dir}")

def load_jsonl(path: str):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]

def load_config(run_dir: str) -> dict:
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
    all_one_hot_actions = []
    all_feedback_vectors = []
    num_samples = 0
    num_records_with_actions = 0

    for rec_idx, rec in enumerate(data):
        if "actions_r1" not in rec:
            continue
        actions_r1 = rec["actions_r1"]
        if not actions_r1:
            continue

        num_records_with_actions += 1

        sim0 = actions_r1[0]
        U = len(sim0)
        S = num_slots_cfg
        if U != num_users_cfg and verbose:
            print(f"[warn] record {rec_idx}: users in log={U} != config={num_users_cfg} (continuing with U={U})")

        for sim_idx, actions_list in enumerate(actions_r1):
            one_hot = torch.zeros(U, S, dtype=torch.float32)
            sic_actions = []
            for u, slots in enumerate(actions_list):
                clean_slots = sorted({s for s in slots if 0 <= s < S})
                if clean_slots:
                    one_hot[u, torch.tensor(clean_slots, dtype=torch.long)] = 1.0
                sic_actions.append((len(clean_slots), clean_slots))

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
    if os.path.exists(dataset_out):
        if verbose:
            print(f"Loading existing dataset: {dataset_out}")
        saved = torch.load(dataset_out)
        one_hot_actions = saved["one_hot_actions"]
        feedback_vectors = saved["feedback_vectors"]
        U = one_hot_actions.shape[1]
        S = one_hot_actions.shape[2]
        num_samples = one_hot_actions.shape[0]
        num_records_with_actions = None
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
# Dataset
# -------------------------------------------------------------------
class ActionsFeedbackDataset(Dataset):
    def __init__(self, one_hot_actions: torch.Tensor, feedback_vectors: torch.Tensor):
        if one_hot_actions.shape[0] != feedback_vectors.shape[0]:
            raise ValueError("Mismatched N between actions and feedback.")
        self.one_hot_actions = one_hot_actions
        self.feedback_vectors = feedback_vectors

    def __len__(self):
        return self.one_hot_actions.shape[0]

    def __getitem__(self, idx):
        return self.one_hot_actions[idx], self.feedback_vectors[idx]

# -----------------------------
# Model: Action -> Feedback
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

        self.user_embed = nn.Embedding(num_users, d_model // 2)
        self.slot_embed = nn.Embedding(num_slots, d_model // 2)

        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_users * num_slots * d_model, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feedback_dim)
        )

        user_idx = torch.arange(num_users).view(1, num_users, 1).expand(1, num_users, num_slots)
        slot_idx = torch.arange(num_slots).view(1, 1, num_slots).expand(1, num_users, num_slots)
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
        B, U, S = x.shape
        assert U == self.num_users and S == self.num_slots, "Input shape does not match model's (U,S)."

        user_emb = self.user_embed(self.user_idx_grid.expand(B, -1, -1))
        slot_emb = self.slot_embed(self.slot_idx_grid.expand(B, -1, -1))
        pos_emb = torch.cat([user_emb, slot_emb], dim=-1)                  # [B,U,S,d]

        x_proj = self.input_proj(x.unsqueeze(-1))                          # [B,U,S,d]
        x_enc = x_proj + pos_emb

        x_seq = x_enc.view(B, U * S, self.d_model)
        h = self.encoder(x_seq)                                            # [B, U*S, d]
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
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model-epoch") and f.endswith(".pt")]
    if len(model_files) <= keep_last:
        return
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
# Permutation symmetrization for eval
# -----------------------------
def forward_with_symmetrization(model, xb, *, num_users, num_slots, out_dim, perms:int,
                                perm_users:bool, perm_slots:bool):
    if perms <= 0:
        return model(xb)

    device = xb.device
    B = xb.size(0)

    # If output can be reshaped per-slot, do it to unpermute slots
    per_slot_dim = None
    if perm_slots and out_dim % num_slots == 0:
        per_slot_dim = out_dim // num_slots

    preds = []
    with torch.no_grad():
        for _ in range(perms):
            x_pert = xb
            inv_slot = None

            if perm_users:
                p_u = torch.randperm(num_users, device=device)
                x_pert = x_pert[:, p_u, :]

            if perm_slots:
                p_s = torch.randperm(num_slots, device=device)
                x_pert = x_pert[:, :, p_s]
                if per_slot_dim is not None:
                    inv = torch.empty_like(p_s)
                    inv[p_s] = torch.arange(num_slots, device=device)
                    inv_slot = inv

            y = model(x_pert)  # [B, F]

            if perm_slots and per_slot_dim is not None:
                y = y.view(B, num_slots, per_slot_dim)
                if inv_slot is not None:
                    y = y[:, inv_slot, :]
                y = y.reshape(B, out_dim)

            preds.append(y)

    return torch.stack(preds, dim=0).mean(dim=0)

# -----------------------------
# Training/validation loop
# -----------------------------
def run_epoch(model, loader, criterion, optimizer, device, train, grad_clip, batch_limit,
              scheduler=None, step_per_batch=False, eval_sym=None):
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

            if (not train) and eval_sym is not None:
                pred = forward_with_symmetrization(
                    model, xb,
                    num_users=eval_sym["U"],
                    num_slots=eval_sym["S"],
                    out_dim=eval_sym["F"],
                    perms=eval_sym["perms"],
                    perm_users=eval_sym["perm_users"],
                    perm_slots=eval_sym["perm_slots"],
                )
            else:
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
    # Instantiate model
    # -----------------------------
    model_dropout = 0.0 if args.zero_reg else args.model_dropout
    model = ActionToFeedbackTransformer(
        num_users=U,
        num_slots=S,
        feedback_dim=F,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=model_dropout,
    ).to(args.device)

    # Resume weights if requested
    if getattr(args, "resume", False):
        model_best_path = os.path.join(run_dir, "model-best.pt")
        if os.path.exists(model_best_path):
            print(f"[resume] Loading model weights from {model_best_path}")
            state_dict = torch.load(model_best_path, map_location=args.device)
            model.load_state_dict(state_dict)
        else:
            print(f"[resume] model-best.pt not found in {run_dir}, starting from scratch.")

    # -----------------------------
    # Loss, Optim, Schedulers
    # -----------------------------
    criterion = nn.MSELoss()
    weight_decay = 0.0 if args.zero_reg else args.weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    if not args.warmup_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        step_per_batch = False
    else:
        total_steps = args.epochs * steps_per_epoch_effective
        warmup_steps = max(3 * steps_per_epoch_effective, int(0.03 * total_steps))
        eta_min = 1e-6
        base_lr = args.lr

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min / base_lr + 0.5 * (1 - eta_min / base_lr) * (1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        step_per_batch = True

    # -----------------------------
    # SWA support (epoch-based start)
    # -----------------------------
    use_swa = args.swa
    swa_model = None
    swa_start_epoch = None
    if use_swa:
        swa_model = AveragedModel(model)
        if args.swa_start_epoch is not None:
            swa_start_epoch = args.swa_start_epoch
        else:
            swa_start_epoch = max(1, int(args.swa_start_frac * args.epochs))
        print(f"[swa] Enabled. Start averaging at epoch {swa_start_epoch} "
              f"({'{:.0%}'.format(args.swa_start_frac)} of training)" if args.swa_start_epoch is None else "")

    # -----------------------------
    # Eval symmetrization config
    # -----------------------------
    eval_sym = None
    if args.eval_symmetrize > 0:
        perm_users = args.symm_users
        perm_slots = args.symm_slots or (not args.symm_users)  # default to slots if none specified
        eval_sym = dict(
            U=U, S=S, F=F,
            perms=args.eval_symmetrize,
            perm_users=perm_users,
            perm_slots=perm_slots
        )
        print(f"[sym] Eval symmetrization: perms={args.eval_symmetrize}, users={perm_users}, slots={perm_slots}")

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val = float("inf")
    best_state = None

    # Logging file (timestamped when resuming)
    if getattr(args, "resume", False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_path = os.path.join(run_dir, f"training-generator-resume-{timestamp}.jsonl")
    else:
        train_log_path = os.path.join(run_dir, "training-generator.jsonl")

    with open(train_log_path, "w") as train_log_f:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss = run_epoch(
                model, train_loader, criterion, optimizer, args.device, train=True,
                grad_clip=args.grad_clip, batch_limit=args.batches_per_epoch,
                scheduler=scheduler if step_per_batch else None,
                step_per_batch=step_per_batch
            )

            val_loss = run_epoch(
                model, val_loader, criterion, optimizer, args.device, train=False,
                grad_clip=args.grad_clip, batch_limit=args.batches_per_epoch,
                eval_sym=eval_sym
            )

            if not step_per_batch:
                scheduler.step()

            # SWA: update running average after this epoch if past start
            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

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

            print(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={current_lr:.2e}  duration={duration:.2f}s")

            # Save model at intervals
            if args.epoch_save_interval > 0 and (epoch % args.epoch_save_interval == 0 or epoch == args.epochs):
                model_path = os.path.join(run_dir, f"model-epoch{epoch}.pt")
                save_model(model, model_path)
                print(f"Saved model at {model_path}")
                manage_saved_models(run_dir, args.keep_last_models)

    # Restore best weights and save final model-best.pt
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(args.device)
        print(f"Restored best model with val_loss={best_val:.6f}")
        best_model_path = os.path.join(run_dir, "model-best.pt")
        save_model(model, best_model_path)
        print(f"Saved best model at {best_model_path}")

    # -----------------------------
    # SWA finalization (evaluate and save)
    # -----------------------------
    if use_swa and swa_model is not None:
        # If the model used BatchNorms, this updates BN stats; otherwise it's a no-op.
        try:
            update_bn(train_loader, swa_model, device=args.device)
        except Exception:
            pass

        swa_model.to(args.device)
        swa_val = run_epoch(
            swa_model, DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
            nn.MSELoss(), optimizer, args.device, train=False,
            grad_clip=None, batch_limit=args.batches_per_epoch,
            eval_sym=eval_sym
        )
        print(f"[swa] Validation loss: {swa_val:.6f}")

        swa_path = os.path.join(run_dir, "model-swa.pt")
        save_model(swa_model, swa_path)
        print(f"[swa] Saved SWA model at {swa_path}")

        # If SWA is better, promote it to best
        if swa_val < best_val:
            save_model(swa_model, os.path.join(run_dir, "model-best.pt"))
            print(f"[swa] SWA improved best model. New best val_loss={swa_val:.6f}")

def parse_args():
    parser = argparse.ArgumentParser()
    # Training & data
    parser.add_argument('--epoch-save-interval', type=int, default=200, help='Save model every N epochs')
    parser.add_argument('--result-dir', type=str, default=None, help='Override result dir')
    parser.add_argument('--keep-last-models', type=int, default=DEFAULT_KEEP_LAST_MODELS, help='Keep only the last X saved models')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--run-dir', type=str, default="res-1p-u10-s10-e1000-b1000-s100", help='Result directory')
    parser.add_argument('--use-energy-feedback', action='store_true', help='Use per-slot counts instead of undecoded one-hot')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Limit to this many batches per epoch')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--resume', action='store_true', help='Resume training from model-best.pt if it exists')

    # LR scheduling
    parser.add_argument('--warmup-cosine', action='store_true', help='Per-batch linear warmup then cosine decay to eta_min')
    parser.add_argument('--lr', type=float, default=3e-4, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (AdamW)')

    # Zero-reg finisher
    parser.add_argument('--zero-reg', action='store_true', help='Turn off dropout and weight decay for last-mile polishing')
    parser.add_argument('--model-dropout', type=float, default=0.1, help='Model dropout (ignored if --zero-reg)')

    # Model size
    parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=8, help='Transformer nhead')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Transformer FFN dim')

    # SWA
    parser.add_argument('--swa', action='store_true', help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa-start-epoch', type=int, default=None, help='Epoch to start SWA (default: use fraction)')
    parser.add_argument('--swa-start-frac', type=float, default=0.7, help='Fraction of total epochs after which to start SWA')

    # Eval-time permutation symmetrization
    parser.add_argument('--eval-symmetrize', type=int, default=0, help='Average over N random permutations at eval (0 disables)')
    parser.add_argument('--symm-users', action='store_true', help='Permute users during eval symmetrization')
    parser.add_argument('--symm-slots', action='store_true', help='Permute slots during eval symmetrization')

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = args.result_dir if args.result_dir is not None else args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    DATASET_OUT = os.path.join(run_dir, "actions_feedback_dataset.pt")

    config = load_config(run_dir)
    num_slots_cfg = int(config["num_slots"])
    num_users_cfg = int(config["num_users"])

    one_hot_actions, feedback_vectors, U, S, num_samples, num_records_with_actions = load_or_build_actions_feedback_dataset(
        run_dir,
        DATASET_OUT,
        num_slots_cfg,
        num_users_cfg,
        use_energy_feedback=args.use_energy_feedback,
        verbose=args.verbose
    )

    dataset = ActionsFeedbackDataset(one_hot_actions, feedback_vectors)
    if args.verbose:
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
        first_batch = next(iter(dataloader))
        xb, yb = first_batch
        print(f"[dataloader] batch x: {tuple(xb.shape)}, y: {tuple(yb.shape)}")

    N, U, S = one_hot_actions.shape
    F = feedback_vectors.shape[1]

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
