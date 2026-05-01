#! /bin/bash


tsp -S 32
for users in $(seq 10 30) ; do
    slots=$(( ($users+1)/2 ))
    tsp python -m src.train.irsa_two_phases --slots 10 --users $users --torch-single-core --prefix load --seed 1 --log --compress
done
