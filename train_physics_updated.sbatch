#!/bin/bash
#SBATCH --job-name=phy_fold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:1
#SBATCH --output=/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/logs/%x_%j.out
#SBATCH --error=/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/logs/%x_%j.err

# Create log directory if it doesn't exist
mkdir -p /fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/logs

# Configure task and seed
TASK=$1    # fold, close, or cover
SEED=$2    # 42, 123, or 456

# Activate conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set critical environment variables
export LD_PRELOAD=/fs/cml-scratch/amishab/miniconda3/envs/lfd/lib/libstdc++.so.6
export WANDB_MODE=offline
export PYTHONPATH=/fs/cml-scratch/amishab/equibot:$PYTHONPATH

# Set output directory and config location
OUTPUT_DIR="/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/${TASK}_${SEED}"
CONFIG_DIR="/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/${TASK}_${SEED}"
CONFIG_FILE="pi_phys_${TASK}_${SEED}"
export CUSTOM_OUTPUT_DIR=$OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Print diagnostic information
echo "Task: $TASK"
echo "Seed: $SEED"
echo "Config directory: $CONFIG_DIR"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "Log directory: /fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/logs/"

# Check for available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run the training script
cd /fs/cml-scratch/amishab/equibot
python -m equibot.policies.train --config-path=$CONFIG_DIR --config-name=$CONFIG_FILE device=cuda seed=$SEED hydra.run.dir=$OUTPUT_DIR log_dir=$OUTPUT_DIR 