#!/bin/bash
#SBATCH --job-name=eval_phys
#SBATCH --output=logs/eval_phys_%j.log
#SBATCH --error=logs/eval_phys_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=16G

# Source conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Set data path for the model to use
export DATA_PATH="/fs/nexus-projects/Sketch_REBEL/equibot/data/fold_phy/pcs"

# Add the current directory to the Python path to fix module imports
export PYTHONPATH=$PYTHONPATH:/fs/cml-scratch/amishab/equibot

# Run evaluation with physics supervision
cd /fs/cml-scratch/amishab/equibot
python -m equibot.policies.vec_eval --config-name pi_phys

echo "Evaluation completed!" 