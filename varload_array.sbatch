#!/bin/bash
#SBATCH --job-name=varload
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal-best
#SBATCH --output=logs/%x_%A_%a.out
# #SBATCH --time=02:00:00
# #SBATCH --mem=2G
# (No --array here; set it at submit time)

set -euo pipefail
mkdir -p logs

# If you use Conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# Keep Python chatty and single-threaded
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Parameters from sbatch --export
ONE_PHASE=${ONE_PHASE:-0}
POISSON=${POISSON:-0}
SEED=${SEED:-1}
SLOTS=${SLOTS:-10}       # frame size per phase (do NOT double here)
PREFIX=${PREFIX:-load}

USERS=${SLURM_ARRAY_TASK_ID:-1}

# For logging: offered load per phase (users/slots)
G_NUM=$USERS
G_DEN=$SLOTS
printf -v G "%.4f" "$(python - <<'PY'
import os
u=int(os.environ["G_NUM"]); s=int(os.environ["G_DEN"])
print(u/s if s else 0.0)
PY
)"

echo "$(date -Is) job=$SLURM_JOB_NAME id=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=${SLURMD_NODENAME:-NA} ONE_PHASE=$ONE_PHASE POISSON=$POISSON seed=$SEED slots(per-phase)=$SLOTS users=$USERS G=users/slots=$G"

args=(
  python irsa_two_phases.py
  --prefix "$PREFIX"
  --slots "$SLOTS"        # per-phase frame size
  --users "$USERS"
  --torch-single-core
  --seed "$SEED"
  --log
  --batch-size 1000
  --epochs 1000
  --epoch-half-lr-interval 100
  --epoch-save-interval 100
  --keep-last-models 100
  --transmission-cost 0
)
(( ONE_PHASE )) && args+=(--one-phase)
(( POISSON ))   && args+=(--poisson)

srun --cpu-bind=cores "${args[@]}"
