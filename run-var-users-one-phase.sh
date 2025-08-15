#! /bin/bash

seed="${1:-1}"

tsp -S 32
for users in $(seq 1 20) ; do
    slots=$(( (($users+1)/2)*2 ))
    tsp python irsa_one_phase.py --slots $slots --users $users --torch-single-core --seed "$seed" --log --compress
done
