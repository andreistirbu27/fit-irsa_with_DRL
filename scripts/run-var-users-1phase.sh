#! /bin/bash

tsp -S 32
for seed in $(seq 1 20); do
    for users in $(seq 1 20) ; do
        slots=$(( users + 1 ))
        tsp python -m src.train.irsa_one_phase --slots "$slots" --users "$users" --torch-single-core --seed "$seed" --log --compress
    done
done
