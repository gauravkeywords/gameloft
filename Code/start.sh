#!/bin/bash

cd /home/ubuntu/projects/Code


source ~/miniconda3/etc/profile.d/conda.sh

conda activate base

nohup python scrap_gameloft.py > output.log 2>&1 &
nohup python supbase_fastmcp.py > fastmacp.log 2>&1 &