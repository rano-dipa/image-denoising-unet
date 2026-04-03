#!/bin/bash
#SBATCH --job-name=prune
#SBATCH --output=/scratch/user/dipanwita22rano/denoising_project/logs/prune_%j.out
#SBATCH --error=/scratch/user/dipanwita22rano/denoising_project/logs/prune_%j.err
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dipanwita22rano@tamu.edu

source ~/.bashrc
conda activate denoise

cd /scratch/user/dipanwita22rano/denoising_project

python code/prune_and_eval.py