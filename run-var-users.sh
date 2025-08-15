#!/bin/bash

# Use the first argument as seed if provided, else default to 1
seed="${1:-1}"

tsp -S 32
for users in $(seq 1 20); do
    slots=$(( (users+1)/2 ))
    tsp python irsa_two_phases.py \
        --slots "$slots" \
        --users "$users" \
        --torch-single-core \
        --seed "$seed" \
        --log \
        --compress
done

