# *TRAIN CODE

# Stage 1 -> contact points
PYTHONPATH=. python s1_cont_est_train.py --cfg gta_stage1_PVCNN2_DCT_CONT_GCN --is_amp --gpus 1

# Stage 2 -> Trajectory Forecasting
PYTHONPATH=. python s2_traj_train.py --cfg gta_stage2_TRAJ_GCN --is_amp --gpus 1

# Stage 3 -> Pose Forecasting
PYTHONPATH=. python s3_pose_train.py --cfg gta_stage2_GCN_POSE --is_amp --gpus 1


# *TEST CODE

# Using GT contact points
python s3_pose_test.py --cfg_cont gta_stage1_PVCNN2_DCT_CONT_GCN --cfg gta_stage2_GCN_POSE --iter 50 --iter_cont 50

# Using estimated contact points
python s3_pose_test.py --cfg_cont gta_stage1_PVCNN2_DCT_CONT --cfg gta_stage2_GRU_POSE --iter 50 --iter_cont 50 --w_est_cont
