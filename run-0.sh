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
python Parallel_is_All_You_Want.py
# python prepare_data_spkr_dep.py # --agwn_augment
# python main.py --num_epochs 500 --data_dir ./data/spkr-dep --model_dir ./models/checkpoints/spkr-dep/
# python test.py --epoch_num 499 --data_dir ./data/spkr-dep --model_dir ./models/checkpoints/spkr-dep/
