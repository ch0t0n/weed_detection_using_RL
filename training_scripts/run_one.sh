#!/bin/bash

# Run a single experiment with: sbatch training_scripts/run_one.sh

#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

algorithm="A2C"
set=1
steps=1000000

conda run --no-capture-output -n rl4pag python3 train.py --algorithm $algorithm --set $set --verbose 1 --steps $steps

wait