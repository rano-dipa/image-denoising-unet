#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=/scratch/user/dipanwita22rano/denoising_project/logs/baseline_%j.out
#SBATCH --error=/scratch/user/dipanwita22rano/denoising_project/logs/baseline_%j.err
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

source ~/.bashrc
conda activate denoise

cd /scratch/user/dipanwita22rano/denoising_project

python code/baseline_filters.py \
--data_root data/SIDD \
--out_dir outputs/baseline
