#! /bin/bash

tsp -S 10
for seed in $(seq 1 20); do
    for users in $(seq 1 20) ; do
        slots=$(( 2*(($users+2)/2) ))
        tsp python -m src.train.irsa_one_phase --slots "$slots" --users "$users" --torch-single-core --seed "$seed" --log --compress
    done
done
