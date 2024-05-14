#!/bin/bash
#SBATCH --time=12:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=ser
#SBATCH -n 1

module load mamba
module load cuda/11.8 

source activate pytorch-env

## CNN-Transformer
# /usr/bin/time -v python Parallel_is_All_You_Want.py
# 81849 Train done, test error with model device location
/usr/bin/time -v python teset.py
