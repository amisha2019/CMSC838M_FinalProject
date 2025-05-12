#!/bin/bash
#SBATCH --job-name=equibot_prod
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/prod_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/prod_%j.err
#SBATCH --time=48:00:00
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
# Set wandb to offline mode (can be changed to online if desired)
export WANDB_MODE=offline

# Create log directory with proper permissions
mkdir -p /fs/nexus-projects/Sketch_VLM_RL/equibit
chmod 775 /fs/nexus-projects/Sketch_VLM_RL/equibit

# Print some debug info
echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: /fs/nexus-projects/Sketch_VLM_RL/equibit"
echo "User: $(whoami)"
echo "Groups: $(groups)"

# Change to project directory
cd /fs/cml-scratch/amishab/equibot/

# Run training with production settings
# This now saves all logs, checkpoints, and videos to /fs/nexus-projects/Sketch_VLM_RL/equibit
python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  prefix=sim_mobile_fold_7dof_equibot_prod \
  data.dataset.path=/fs/cml-scratch/amishab/data/fold/pcs \
  use_wandb=True

# Check results after completion
echo "Job completed at $(date)"
echo "Listing contents of the output directory:"
ls -la /fs/nexus-projects/Sketch_VLM_RL/equibit/
echo "Checking for checkpoint files:"
find /fs/nexus-projects/Sketch_VLM_RL/equibit/ -name "*.pth" -type f | head -10
echo "Checking for video files:"
find /fs/nexus-projects/Sketch_VLM_RL/equibit/ -name "*.mp4" -type f | head -10

# Print info for syncing wandb runs to cloud (if desired)
echo "To sync wandb runs to the cloud, run:"
find /fs/nexus-projects/Sketch_VLM_RL/equibit/wandb -name "offline-run-*" -type d | xargs -I{} echo "wandb sync {}" 