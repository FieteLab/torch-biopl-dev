#!/usr/bin/bash
#SBATCH -J trainer
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --constraint=rocky8
#SBATCH --mem 40G
#SBATCH -o outputs/trainer_%j.out

source ~/.bashrc
conda activate pytorch

python -u src/bioplnn/trainers/ei.py data.mode=color data.holdout=[blue,green] model.modulation_type=ag optimizer.lr=0.0001 criterion.all_timesteps=True model.immediate_inhibition=False model.exc_rectify=pos