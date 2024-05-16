#!/bin/bash
#SBATCH --time=00:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=ser
#SBATCH -n 1

module load mamba

source activate pytorch-env

## CNN-Transformer
# /usr/bin/time -v python Parallel*.py
# 186263 test on epoch 429 67.56
# 186280 test on epoch 499 68.67%
/usr/bin/time -v python prepare_data.py
/usr/bin/time -v python main.py 
/usr/bin/time -v python test.py
# 193718 3s 

## speaker-independant 1-fold w.r.t. MMEmoRec

## speaker-independant 5-fold
