#!/usr/bin/env bash
# Experiment matrix answering the open questions of the SuperMimic case study on Push-T.
# Each run: ~2M environment steps. Run them one after another (or in parallel if you have the RAM).
set -e
STEPS=${STEPS:-2000000}

# Q1. Sparse vs dense reward, pathwise (SuperMimic) vs likelihood ratio (PPO)
python train.py --algo supermimic --reward dense  --total-steps $STEPS --run sm_dense
python train.py --algo supermimic --reward sparse --total-steps $STEPS --run sm_sparse
python train.py --algo ppo        --reward dense  --total-steps $STEPS --run ppo_dense
python train.py --algo ppo        --reward sparse --total-steps $STEPS --run ppo_sparse

# Q3. Does the critic help? (and the horizon trade-off from SuperMimic Fig. 4)
python train.py --algo supermimic --reward dense --no-critic  --total-steps $STEPS --run sm_dense_nocritic
python train.py --algo supermimic --reward dense --horizon 16 --total-steps $STEPS --run sm_dense_H16
python train.py --algo supermimic --reward dense --horizon 8  --total-steps $STEPS --run sm_dense_H8

# Physics: original Push-T without table friction (block keeps sliding after a push)
python train.py --algo supermimic --reward dense --damping 0 --total-steps $STEPS --run sm_dense_nofriction
python train.py --algo ppo        --reward dense --damping 0 --total-steps $STEPS --run ppo_dense_nofriction

# Ablation from the ESM paper: no reverse curriculum
python train.py --algo supermimic --reward dense --no-curriculum --total-steps $STEPS --run sm_dense_nocurr

python plot.py
