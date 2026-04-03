#!/bin/bash
#SBATCH --job-name=denoise_test
#SBATCH --output=/scratch/user/dipanwita22rano/denoising_project/logs/test_%j.out
#SBATCH --error=/scratch/user/dipanwita22rano/denoising_project/logs/test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dipanwita22ano@tamu.edu

source ~/.bashrc
conda activate denoise

cd /scratch/user/dipanwita22rano/denoising_project

echo "GPU test"
python - <<'EOF'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
EOF

python code/train_unet.py \
  --data_root /scratch/user/dipanwita22rano/denoising_project/data/SIDD \
  --project_root /scratch/user/dipanwita22rano/denoising_project \
  --epochs 1 \
  --batch_size 4 \
  --patches_per_image 2 \
  --num_workers 2
