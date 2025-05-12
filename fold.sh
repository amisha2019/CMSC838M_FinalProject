#!/bin/bash
#SBATCH --job-name=equibot_train
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/train_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/train_%j.err
#SBATCH --time=0:30:00
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

# Create log directory
mkdir -p /fs/nexus-projects/Sketch_VLM_RL/equibit

# Change to project directory
cd /fs/cml-scratch/amishab/equibot/

# Run training with shorter epochs for testing
python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  prefix=sim_mobile_fold_7dof_equibot \
  data.dataset.path=/fs/cml-scratch/amishab/data/fold/pcs \
  training.num_epochs=5 \
  training.eval_interval=2 \
  training.save_interval=2 \
  training.vis_interval=1
