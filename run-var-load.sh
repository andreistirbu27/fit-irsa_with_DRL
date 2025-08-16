#!/usr/bin/env bash
set -euo pipefail

# Optional flag at the front: --one-phase
ONE_PHASE=0
if [[ ${1-} == "--one-phase" ]]; then
  ONE_PHASE=1
  shift
fi

# Positional args (after optional flag)
seed="${1:-1}"
slots="${2:-10}"
max_users="${3:-$((slots*2))}"

echo "one_phase=${ONE_PHASE} seed=${seed} slots=${slots} max_users=${max_users}"

# tsp -S 32
for users in $(seq 1 "$max_users"); do
  args=(
    irsa_two_phases.py
    --prefix load
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
  )
  # Add the python flag only if requested
  if (( ONE_PHASE )); then
    args+=(--one-phase)
  fi

  tsp python "${args[@]}"
done

