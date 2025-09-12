#!/bin/bash
#SBATCH --job-name=varusers
#SBATCH --array=1-15                  # 2..30 users in steps of 2 (15 points)
#SBATCH --cpus-per-task=1             # set per Python proc (adjust as needed)
# #SBATCHNOT --time=02:00:00               # adjust
# #SBATCHNOT --mem=2G                      # adjust
#SBATCH --partition=normal-best
#SBATCH --output=logs/%x_%A_%a.out
# #SBATCH --account=...               # if needed

# Optional feature flags passed via environment (POISSON=1 ONE_PHASE=1), default 0
POISSON=${POISSON:-0}
ONE_PHASE=${ONE_PHASE:-0}

# Seed: use provided SEED env var or default to 1; you can also vary by task if you want
BASE_SEED=${SEED:-1}
seed=$BASE_SEED

# Map array index -> users = 2,4,...,30
users=$(( SLURM_ARRAY_TASK_ID * 2 ))
slots=$(( (users + 1) / 2 ))

# If one phase, double the slots (after rounding up)
if (( ONE_PHASE )); then
  slots=$(( slots * 2 ))
fi

# Build args
args=(
  python irsa_two_phases.py
  --slots "$slots"
  --users "$users"
  --torch-single-core
  --seed "$seed"
  --log
  --batch-size 1000
  --epochs 1000
  --epoch-half-lr-interval 100
  --epoch-save-interval 100
  --keep-last-models 100
  --transmission-cost 0
)

(( POISSON ))   && args+=(--poisson)
(( ONE_PHASE )) && args+=(--one-phase)

# Run one process for this array task
srun "${args[@]}"
