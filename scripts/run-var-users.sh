#! /bin/bash

tsp -S 32
for users in $(seq 1 20) ; do
    slots=$(( ($users+1)/2 ))
    tsp python -m src.train.irsa_two_phases --slots $slots --users $users --torch-single-core --seed 1 --log --compress
done
