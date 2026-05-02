#! /bin/bash

tsp -S 32
for seed in $(seq 1 20); do
    for k in 2 3 4; do
        for users in $(seq 1 20); do
            total_slots=$(( users + 1 ))
            slots=$(( (total_slots + k - 1) / k ))
            tsp python -m src.train.irsa_k_phase \
                --slots "$slots" --users "$users" --num-phases "$k" \
                --torch-single-core --seed "$seed" --log --compress
        done
    done
done
