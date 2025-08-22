#!/bin/bash

# Use the first argument as seed if provided, else default to 1
seed="${1:-1}"

# Parse optional flags --poisson and --one-phase
POISSON=0
ONE_PHASE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poisson)
      POISSON=1
      shift
      ;;
    --one-phase)
      ONE_PHASE=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

# After optional flags, the first positional argument is seed (if provided)
if [[ $# -gt 0 ]]; then
  seed="$1"
fi

tsp -S 64
for users in $(seq 2 2 30); do
    slots=$(( (users+1)/2 ))
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
    if (( POISSON )); then
        args+=(--poisson)
    fi
    if (( ONE_PHASE )); then
        args+=(--one-phase)
    fi
    tsp "${args[@]}"
done
