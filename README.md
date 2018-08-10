## Important files 
rnet.py contain the network definition of representor, and train_r.ipynb is used to train it toward TD task.

## To run it?
python main.py --env Pong-v0 --workers 7 --gpu-ids 0 --amsgrad True --pre-rnet pong_human_1env_novae --actor-weight 0.66 --rl-r 0.0001 --max-step 800000 --log-target human_1env_novae_aw0.66_1e-4
