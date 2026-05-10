#! /bin/bash

tsp -S 10
for seed in $(seq 1 20); do
    for users in $(seq 10 30) ; do
        tsp python -m src.train.irsa_one_phase --slots 20 --users "$users" --torch-single-core --prefix load --seed "$seed" --log --compress
    done
done
