#!/usr/bin/env bash
set -euo pipefail

# Optional flags at the front: --one-phase and/or --poisson
ONE_PHASE=0
POISSON=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --one-phase)
      ONE_PHASE=1
      shift
      ;;
    --poisson)
      POISSON=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Positional args (after optional flags)
seed="${1:-1}"
slots="${2:-10}"
max_users="${3:-$((slots*2))}"

echo "one_phase=${ONE_PHASE} poisson=${POISSON} seed=${seed} slots=${slots} max_users=${max_users}"

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
  # Add the python flags only if requested
  if (( ONE_PHASE )); then
    args+=(--one-phase)
  fi
  if (( POISSON )); then
    args+=(--poisson)
  fi

  tsp python "${args[@]}"
done
