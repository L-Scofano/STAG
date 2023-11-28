#!/bin/bash

# *TRAIN CODE

# Stage 1 -> contact points
python exp_gta_stage1_cont.py --cfg gta_stage1_PVCNN2_DCT_CONT --is_amp --gpus 1

# Stage 2 -> pose
python exp_gta_stage2_pose.py --cfg gta_stage2_GRU_POSE --is_amp --gpus 1

# *TRAIN CODE GCN

# Stage 1 -> contact points
python exp_gta_stage1_cont.py --cfg gta_stage1_PVCNN2_DCT_CONT_GCN --is_amp --gpus 1

# Stage 2 -> pose
PYTHONPATH=. python exp_gta_stage2_pose.py --cfg gta_stage2_GCN_POSE --is_amp --gpus 1

# Trajectory
PYTHONPATH=. python exp_root_traj_train.py --cfg gta_stage2_TRAJ_GCN --is_amp --gpus 1


# *TEST CODE

# Using GT contact points
python eval_gta_stats.py --cfg_cont gta_stage1_PVCNN2_DCT_CONT_GCN --cfg gta_stage2_GCN_POSE --iter 50 --iter_cont 50

# Using estimated contact points
python eval_gta_stats.py --cfg_cont gta_stage1_PVCNN2_DCT_CONT --cfg gta_stage2_GRU_POSE --iter 50 --iter_cont 50 --w_est_cont
