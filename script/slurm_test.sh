#!/bin/bash
#SBATCH --account [Your GPU Project Name]
#SBATCH --partition ampere
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

source ~/.bashrc
conda activate [Your Conda Env]
cd [Your Path]/scotch_and_soda/
python test.py