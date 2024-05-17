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
# python prepare_data_spkr_dep.py
/usr/bin/time -v python main.py --in_file ./features+labels-spkr-dep.npy --model_dir ./models/checkpoints-spkr-dep/
/usr/bin/time -v python test.py
# 202041

# /usr/bin/time -v python prepare_data.py
# /usr/bin/time -v python main.py 
# /usr/bin/time -v python test.py
# 
## speaker-independant 1-fold w.r.t. mmerr
# 193718 23.89%

## 5s audio segments, how does model parameter change?
# use full-lenght audio, padding and batch processing

## speaker-independant 5-fold
