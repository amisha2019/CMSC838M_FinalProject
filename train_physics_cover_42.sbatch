#!/bin/bash
#SBATCH --job-name=phy_cover42
#SBATCH --output=logs/phy_cover_42_%j.out
#SBATCH --error=logs/phy_cover_42_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Create log directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set PYTHONPATH
export PYTHONPATH=/fs/cml-scratch/amishab/equibot:

# Set output directory
export CUSTOM_OUTPUT_DIR=/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision/cover_42

# Run the training script with specific output directory
python -m equibot.policies.train --config-name pi_phys_cover_42
