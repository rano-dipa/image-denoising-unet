#!/bin/bash
#SBATCH --job-name=denoise_unet
#SBATCH --output=/scratch/user/dipanwita22rano/denoising_project/logs/denoise_%j.out
#SBATCH --error=/scratch/user/dipanwita22rano/denoising_project/logs/denoise_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dipanwita22ano@tamu.edu

# Activate conda correctly on Grace
source ~/.bashrc
conda activate denoise

cd /scratch/user/dipanwita22rano/denoising_project

echo "Running on node:"
hostname
echo "GPU check:"
python - <<'EOF'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
EOF

python code/train_unet.py \
  --data_root /scratch/user/dipanwita22rano/denoising_project/data/SIDD \
  --project_root /scratch/user/dipanwita22rano/denoising_project \
  --batch_size 16 \
  --val_batch_size 8 \
  --patch_size 128 \
  --patches_per_image 20 \
  --epochs 30 \
  --lr 1e-4 \
  --use_perceptual \
  --save_every_batches 500 \
  --num_workers 6
