
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



------

* 
python train_irsa_estimator.py --resume --resume --warmup-cosine --lr 5e-5 --epochs 60

* Zero-reg finisher
python train_irsa_estimator.py --resume --warmup-cosine --zero-reg --lr 2e-5 --epochs 40

* SWA
python train_irsa_estimator.py --resume --warmup-cosine --lr 1e-4 --epochs 120 --swa --swa-start-frac 0.7
or --swa --swa-start-epoch 80

* symmetrization
--eval-symmetrize 16 --symm-slots
--eval-symmetrize 16 --symm-users
--eval-symmetrize 16 --symm-users --symm-slots

