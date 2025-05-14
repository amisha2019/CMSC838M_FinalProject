#!/bin/bash
#SBATCH --job-name=equibot_fix
#SBATCH --output=/fs/nexus-projects/Sketch_REBEL/equibot/anukriti/fix_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_REBEL/equibot/anukriti/fix_%j.err
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

# Clean up previous files
rm -f /fs/nexus-projects/Sketch_REBEL/equibot/anukriti/train.log

# Create log directory with proper permissions
mkdir -p /fs/nexus-projects/Sketch_REBEL/equibot/anukriti
chmod 775 /fs/nexus-projects/Sketch_REBEL/equibot/anukriti

# Print debug info
echo "Starting quick fix test at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: /fs/nexus-projects/Sketch_REBEL/equibot/anukriti"

# Change to project directory
cd /fs/cml-scratch/amishab/equibot/

# Run very short test to verify environment loading works properly
python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  prefix=sim_mobile_fold_7dof_fix_test \
  data.dataset.path=/fs/cml-scratch/amishab/data/fold/pcs \
  training.num_epochs=2 \
  training.eval_interval=1 \
  training.save_interval=1 \
  training.vis_interval=1 \
  use_wandb=True

# Check results
echo "Test completed at $(date)"
echo "Listing logs directory:"
ls -la /fs/nexus-projects/Sketch_REBEL/equibot/anukriti/ 