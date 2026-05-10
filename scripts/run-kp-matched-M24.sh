#!/bin/bash

tsp -S 10
M=24
P=m${M}
for seed in $(seq 1 20); do
    for users in $(seq 10 30); do
        # k=1 — 1-phase
        tsp python -m src.train.irsa_one_phase \
            --slots ${M} --users ${users} --prefix ${P} \
            --torch-single-core --seed ${seed} --log --compress

        # k=2 — 2-phase REINFORCE
        tsp python -m src.train.irsa_two_phases \
            --slots $((M/2)) --users ${users} --prefix ${P} \
            --torch-single-core --seed ${seed} --log --compress

        # k=3 — k-phase
        tsp python -m src.train.irsa_k_phase \
            --slots $((M/3)) --users ${users} --num-phases 3 --prefix ${P} \
            --torch-single-core --seed ${seed} --log --compress

        # k=4 — k-phase
        tsp python -m src.train.irsa_k_phase \
            --slots $((M/4)) --users ${users} --num-phases 4 --prefix ${P} \
            --torch-single-core --seed ${seed} --log --compress
    done
done
