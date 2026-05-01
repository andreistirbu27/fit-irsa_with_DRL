# RL for IRSA

Reinforcement learning for **IRSA** — Irregular Repetition Slotted ALOHA — a random-access protocol where many users transmit at the same time across a small number of slots, and a receiver decodes them via successive interference cancellation (SIC).

The goal here is to learn a per-user policy that decides which slots to transmit on, so the receiver decodes as many users as possible. There's a single-round version and a two-round version (where users get feedback after the first round and can retransmit). Each variant comes in REINFORCE and PPO flavours.


## Setup

You need Python 3, PyTorch, NumPy, and tqdm.

```bash
pip install torch numpy tqdm matplotlib scipy
```

Everything runs on CPU and is intentionally single-threaded by default in sweeps (`--torch-single-core`).

## Running a single experiment

Pick a variant and call it as a module from the repo root:

```bash
python -m src.train.irsa_two_phases --users 5 --slots 3 --seed 1 --epochs 2000 --log --compress
```

That writes a directory under `results/new/` (e.g. `results/new/res-2p-u5-s3-seed1/`) containing:
- `config.json` — exact hyperparameters
- `train_log.jsonl.gz` — per-epoch metrics
- `policy_final.pt` plus a couple of checkpoints

The variants live in [src/train/](src/train/):

| File | Algorithm | Setting | Network | Result-dir prefix |
|------|-----------|---------|---------|-------------------|
| `irsa_one_phase.py` | REINFORCE | single round | 1 hidden × 128 | `res-1p` |
| `irsa_two_phases.py` | REINFORCE | two rounds + feedback | 1 hidden × 128 | `res-2p` |
| `irsa_2phase_2x64.py` | REINFORCE | two rounds + feedback | configurable depth (default 2 × 64) | `res-2p-2x64` |
| `irsa_one_phase_ppo.py` | PPO | single round | actor-critic, 2 × 128 | `res-1p-ppo` |
| `irsa_two_phases_ppo.py` | PPO | two rounds + feedback | actor-critic, 2 × 128 | `res-2p-ppo` |

All scripts share a common set of CLI flags: `--users --slots --seed --epochs --batch-size --learning-rate --hidden-dim --prefix --log --compress`. PPO scripts add `--clip-eps --value-coef --entropy-coef --gamma --gae-lambda --ppo-epochs --minibatch-size --max-grad-norm`. `irsa_2phase_2x64.py` adds `--num-layers`. Run any script with `--help` to see everything.

## Sweeps

[scripts/run-var-users.sh](scripts/run-var-users.sh) and [scripts/run-var-load.sh](scripts/run-var-load.sh) launch parameter sweeps via the `tsp` task spooler (32 parallel jobs). Both iterate over seeds 1–4. Edit the loop ranges as needed and run from the repo root:

```bash
bash scripts/run-var-users.sh
```

## Plotting

[src/plot/plot_results_long.py](src/plot/plot_results_long.py) is the canonical multi-run plotter — it reads several runs, computes confidence intervals, and produces throughput-vs-load curves. Single-run diagnostics are in `plot_one_phase.py` and `plot_two_phases.py`.

```bash
python -m src.plot.plot_results_long
```

## Evaluating a trained model

[src/eval/eval_two_phases.py](src/eval/eval_two_phases.py) loads a saved policy and runs a forward pass on a single example. Edit the `result_dir` near the top of the file to point at the run you want to inspect. The script auto-detects the variant (REINFORCE / 2×64 / PPO) from `config.json`, so it works with any of the three two-phase training scripts.

## Repo layout

```
src/
  irsa_common/    shared helpers (logging, model save/load, SIC simulation, sampling, seeding)
  train/          5 training scripts, one per variant
  eval/           model loading / inference
  plot/           training-log plotters
scripts/          shell-based parameter sweeps
archive/          old prototypes kept for reference, not run
results/
  new/            new training-run outputs land here (gitignored)
  legacy/         pre-refactor runs kept for reference (gitignored)
```

The training scripts are kept as **separate files** rather than collapsed into one parameterised CLI. New experiment variants are usually created by copying an existing `src/train/irsa_*.py` and editing it. Identical scaffolding (logging, SIC, sampling, seeding) is imported from `src/irsa_common/` so the diff between variants stays focused on the algorithm.

## A note on the PPO scripts

Both PPO scripts use **per-user bandit credit assignment**: each user's step is recorded as a one-step episode (`done=1`) that receives the joint global reward directly. There's no discount-factor chain across the (arbitrary) user ordering — GAE collapses to `advantage = R − V(obs)` per step, with the value head still acting as a learned baseline.
