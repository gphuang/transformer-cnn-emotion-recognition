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

## speaker-independant fold0 w.r.t. mmerr
/usr/bin/time -v python prepare_data.py --agwn_augment
/usr/bin/time -v python main.py --data_dir ./data/spkr-indep/fold0 --model_dir ./models/checkpoints/spkr-indep/fold0
/usr/bin/time -v python test.py --data_dir ./data/spkr-indep/fold0 --model_dir ./models/checkpoints/spkr-indep/fold0
