#!/bin/bash
#SBATCH --job-name=final_train
#SBATCH --output=/scratch/user/dipanwita22rano/denoising_project/logs/final_%j.out
#SBATCH --error=/scratch/user/dipanwita22rano/denoising_project/logs/final_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dipanwita22rano@tamu.edu

source ~/.bashrc
conda activate denoise

cd /scratch/user/dipanwita22rano/denoising_project

python code/train_final.py \
--project_root /scratch/user/dipanwita22rano/denoising_project