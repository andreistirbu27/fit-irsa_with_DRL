#! /bin/bash


tsp -S 32
for users in $(seq 10 30) ; do
    slots=$(( ($users+1)/2 ))
    tsp python irsa_two_phases.py --slots 10 --users $users --torch-single-core
done
