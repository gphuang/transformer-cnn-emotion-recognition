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
# 209874 augment train/val
#   no augmentation

## Modular
python prepare_data_spkr_dep.py --agwn_augment
python main.py --data_dir ./data/spkr-dep --model_dir ./models/checkpoints/spkr-dep/
python test.py --data_dir ./data/spkr-dep --model_dir ./models/checkpoints/spkr-dep/ 
# 208631 augment train/val/test 11.56%
# 210103 augment train/val

## speaker-independant fold0 w.r.t. mmerr
# /usr/bin/time -v python prepare_data_spkr_indep.py --agwn_augment
# /usr/bin/time -v python main.py --data_dir ./data/spkr-indep/fold0 --model_dir ./models/checkpoints/spkr-indep/fold0
# /usr/bin/time -v python test.py --data_dir ./data/spkr-indep/fold0 --model_dir ./models/checkpoints/spkr-indep/fold0
# 208643 augment train/val/test

## 5s audio segments, how does model parameter change?
# use full-lenght audio, padding and batch processing

## speaker-independant 5-fold
