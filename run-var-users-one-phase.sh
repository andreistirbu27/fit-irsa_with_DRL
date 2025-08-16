#!/bin/bash

# Use the first argument as seed if provided, else default to 1
seed="${1:-1}"

tsp -S 64
for users in $(seq 2 2 30); do
    slots=$(( 2*((users+1)/2) ))
    tsp python irsa_two_phases.py \
	--one-phase \
        --slots "$slots" \
        --users "$users" \
        --torch-single-core \
        --seed "$seed" \
        --log \
	--batch-size 1000 \
	--epochs 1000 \
        --epoch-half-lr-interval 100 \
	--epoch-save-interval 100 \
	--keep-last-models 100
done

