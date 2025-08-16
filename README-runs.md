
./run-var-users.sh 1
./run-var-users.sh 2
./run-var-users.sh 3
./run-var-users.sh 4

./run-var-users-one-phase.sh 1
./run-var-users-one-phase.sh 2
./run-var-users-one-phase.sh 3
./run-var-users-one-phase.sh 4

./run-var-load.sh 1 10 30
./run-var-load.sh 2 10 30
./run-var-load.sh 3 10 30
./run-var-load.sh 4 10 30


./run-var-load.sh --one-phase 1 20 30
./run-var-load.sh --one-phase 2 20 30
./run-var-load.sh --one-phase 3 20 30
./run-var-load.sh --one-phase 4 20 30
