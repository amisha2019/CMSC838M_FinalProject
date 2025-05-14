#!/bin/bash
#SBATCH --job-name=equibot_fixed
#SBATCH --output=/fs/nexus-projects/Sketch_REBEL/equibot/anukriti/fixed_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_REBEL/equibot/anukriti/fixed_%j.err
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
mkdir -p /fs/nexus-projects/Sketch_REBEL/equibot/anukriti
chmod 775 /fs/nexus-projects/Sketch_REBEL/equibot/anukriti

# Print some debug info
echo "Starting fixed production job at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: /fs/nexus-projects/Sketch_REBEL/equibot/anukriti"
echo "User: $(whoami)"

# Change to project directory
cd /fs/cml-scratch/amishab/equibot/

# Run full training with production settings
python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  prefix=sim_mobile_fold_7dof_production \
  data.dataset.path=/fs/cml-scratch/amishab/data/fold/pcs \
  training.eval_interval=50 \
  training.save_interval=50 \
  training.vis_interval=100 \
  use_wandb=True

# Report completion
echo "Job completed at $(date)"
echo "Listing contents of the output directory:"
ls -la /fs/nexus-projects/Sketch_REBEL/equibot/anukriti/
echo "Summary of checkpoint files:"
find /fs/nexus-projects/Sketch_REBEL/equibot/anukriti/ -name "*.pth" -type f | wc -l
echo "Summary of video files:"
find /fs/nexus-projects/Sketch_REBEL/equibot/anukriti/ -name "*.mp4" -type f | wc -l

# Print info for syncing wandb runs to cloud (if desired)
echo "To sync wandb runs to the cloud, run:"
find /fs/nexus-projects/Sketch_REBEL/equibot/anukriti/wandb -name "offline-run-*" -type d | xargs -I{} echo "wandb sync {}" 