#!/bin/bash

# setup:
# conda create -n rlenv python=3.9
#
# activate conda:
# source conda_activate.sh
#
# after you activate the conda environment:
#
# pip install -r requirements.txt
# try 'python3 test_libs.py' and see if it runs
# 
# training: 
# then do: python scripts/train.py --config configs/rejoin/rejoin_default.yaml
#
# need to modify these commands based on the output path date:
# produces eval.log for plotting:
# python3 scripts/eval.py --dir /home/stan/Desktop/repositories/SafeRL/output/expr_20240522_143535/ --num_rollouts 1
#
# custom plotting code to make eval_plot.png file:
# python3 scripts/plot_stan.py output/expr_20240522_143535/PPO_DubinsRejoin_15bc3_00000_0_2024-05-22_14-35-38/eval/ckpt_200/eval.log
 
conda activate rlenv
