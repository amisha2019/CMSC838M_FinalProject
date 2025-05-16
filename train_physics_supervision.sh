#!/bin/bash
#SBATCH --job-name=phys_supervision
#SBATCH --output=logs/phys_supervision_%j.log
#SBATCH --error=logs/phys_supervision_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Source conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Set data path for the model to use
export DATA_PATH="/fs/nexus-projects/Sketch_REBEL/equibot/data/fold_phy/pcs"

# Add the current directory to the Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:/fs/cml-scratch/amishab/equibot

# Run training with physics supervision - using correct Hydra parameters
cd /fs/cml-scratch/amishab/equibot
python -m equibot.policies.train --config-name pi_phys

echo "Training completed!" 