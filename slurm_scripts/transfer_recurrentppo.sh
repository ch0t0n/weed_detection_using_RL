#!/bin/bash

# Run RecurrentPPO experiments with: sbatch slurm_scripts/transfer_recurrentppo.sh

#SBATCH --array=0-8
#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# Modify these for other experiments
algorithms=("RecurrentPPO")
sets=(2 3 4 5 6 7 8 9 10)

# IMPORTANT: array job length = num_algorithms * num_sets - 1
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}

index=$((SLURM_ARRAY_TASK_ID))
algorithm_index=$((index / num_sets))
algorithm=${algorithms[$algorithm_index]}
set_index=$((index % num_sets))
set=${sets[$set_index]}

conda run --no-capture-output -n rl4pag python3 transfer.py --algorithm $algorithm --load_set 1 --train_set $set --steps 2000000 --verbose 1 --seed 33 --log_steps 5000 --device "cuda"

wait