#!/bin/bash
#SBATCH --job-name=equibot_final
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/final_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/final_%j.err
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Load conda
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
# Set wandb to offline mode
export WANDB_MODE=offline

# Clean up previous files to start fresh
rm -rf /fs/nexus-projects/Sketch_VLM_RL/equibit/train.log
rm -rf /fs/nexus-projects/Sketch_VLM_RL/equibit/ckpt*.pth
rm -rf /fs/nexus-projects/Sketch_VLM_RL/equibit/eval*.mp4

# Create log directory with proper permissions
mkdir -p /fs/nexus-projects/Sketch_VLM_RL/equibit
chmod 775 /fs/nexus-projects/Sketch_VLM_RL/equibit

# Print debug info
echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: /fs/nexus-projects/Sketch_VLM_RL/equibit"

# Change to project directory
cd /fs/cml-scratch/amishab/equibot/

# Run very short training to test directory override
python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  prefix=sim_mobile_fold_7dof_final_test \
  data.dataset.path=/fs/cml-scratch/amishab/data/fold/pcs \
  training.num_epochs=2 \
  training.eval_interval=1 \
  training.save_interval=1 \
  training.vis_interval=1 \
  use_wandb=True

# Check results
echo "Job completed at $(date)"
echo "Checking files in the target directory:"
ls -la /fs/nexus-projects/Sketch_VLM_RL/equibit/
echo "Checking for checkpoint files:"
find /fs/nexus-projects/Sketch_VLM_RL/equibit/ -name "*.pth" -type f
echo "Checking for video files:"
find /fs/nexus-projects/Sketch_VLM_RL/equibit/ -name "*.mp4" -type f
echo "Checking if Hydra created its own directory:"
find /fs/cml-scratch/amishab/equibot -maxdepth 3 -path "*/sim_mobile_fold_7dof_final_test" -type d 