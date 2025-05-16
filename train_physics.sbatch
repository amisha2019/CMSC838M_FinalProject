#!/bin/bash
#SBATCH --job-name=physics_train
#SBATCH --output=logs/physics_train_%j.out
#SBATCH --error=logs/physics_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default

# Create log directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set PYTHONPATH
export PYTHONPATH=/fs/cml-scratch/amishab/equibot:$PYTHONPATH

# Run the training script
python -m equibot.policies.train --config-name pi_phys 