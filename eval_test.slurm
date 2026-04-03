#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --output=/scratch/user/dipanwita22rano/denoising_project/logs/eval_%j.out
#SBATCH --error=/scratch/user/dipanwita22rano/denoising_project/logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dipanwita22rano@tamu.edu

source ~/.bashrc
conda activate denoise

cd /scratch/user/dipanwita22rano/denoising_project

python code/eval_on_test.py \
--project_root /scratch/user/dipanwita22rano/denoising_project \
--ckpt checkpoints/pruned.pth
