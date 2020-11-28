#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=Q2
#SBATCH --output=logs/job.out
#SBATCH --error=logs/job.err

module load anaconda
python train.py --epoch 70 --num_layers 5 --name model_5_cnn --lr 0.015 --word_mode CNN