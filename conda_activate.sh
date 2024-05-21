#!/bin/bash

# conda create -n rlenv python=3.9
#
# after you make conda environment:
#
# pip install setuptools==65.5.1 pip==21.0 wheel==0.38.0
# pip install -r requirements.txt
# (you may still get a pip warning about versions)
# try 'python3 test_libs.py' and see if it runs
# then do: python scripts/train.py --config configs/rejoin/rejoin_default.yaml
 
conda activate rlenv
