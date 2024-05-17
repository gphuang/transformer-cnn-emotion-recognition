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

## CNN-Transformer speaker-dependant
# /usr/bin/time -v python Parallel_is_All_You_Want.py
# 186263 test on epoch 429 67.56
# 186280 test on epoch 499 68.67%
# 208622 augment train/val/test 71.11%
# 209874 augment train/val 72.00%
# 210251 no augment 72.67% * augmentation effect is not obvious 
# 210978 rename pkl  

## Modular
# python prepare_data_spkr_dep.py  
# python main.py --data_dir ./data/spkr-dep --model_dir ./models/checkpoints/spkr-dep --out_dir ./results/spkr-dep
# 210980 no augment  

## speaker-independant fold0 w.r.t. mmerr
python prepare_data_spkr_indep.py  
python main.py --data_dir ./data/spkr-indep/fold0 --model_dir ./models/checkpoints/spkr-indep/fold0 --out_dir ./results/spkr-indep/fold0
# 210981 no augment

## 5s audio segments, how does model parameter change?
# python test.py use full-lenght audio, padding and batch processing

## speaker-independant 5-fold
