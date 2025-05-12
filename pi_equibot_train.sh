#!/bin/bash
#SBATCH --job-name=pi_equibot_train
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_equibot_train_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_equibot_train_%j.err
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
# Set wandb to offline mode
export WANDB_MODE=offline

# Set working directory
cd /fs/cml-scratch/amishab/equibot

# Create log and checkpoint directories with proper permissions
OUTPUT_DIR=/fs/nexus-projects/Sketch_VLM_RL/equibit
mkdir -p ${OUTPUT_DIR}/checkpoints/original
mkdir -p ${OUTPUT_DIR}/checkpoints/pi_model
chmod -R 775 ${OUTPUT_DIR}

# Define data directory
DATA_DIR=/fs/cml-scratch/amishab/data/fold_physics
PCS_DIR=${DATA_DIR}/pcs

# Check if data exists
if [ ! -d "${PCS_DIR}" ] || [ $(find "${PCS_DIR}" -name "*.npz" | wc -l) -eq 0 ]; then
  echo "Error: Point cloud directory ${PCS_DIR} does not exist or is empty."
  echo "Please run pi_equibot_generate_data.sh first."
  exit 1
fi

echo "Found $(find ${PCS_DIR} -name "*.npz" | wc -l) data files in ${PCS_DIR}"

# Print timestamp
echo "Starting training at $(date)"

# Train the original EquiBot model (physics embedding disabled)
echo "================================================="
echo "Training original EquiBot (without physics embedding)..."
echo "================================================="

python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  model.use_physics_embed=false \
  prefix=original_equibot \
  data.dataset.path=${PCS_DIR} \
  training.eval_interval=50 \
  training.save_interval=50 \
  training.vis_interval=100 \
  training.num_epochs=100 \
  exp_out_dir=${OUTPUT_DIR}/checkpoints/original \
  use_wandb=true \
  wandb.name="original_equibot" \
  wandb.project="pi_equibot"

# Train the physics-informed EquiBot
echo "================================================="
echo "Training Physics-Informed EquiBot..."
echo "================================================="

python -m equibot.policies.train \
  --config-name fold_mobile_equibot \
  model.use_physics_embed=true \
  model.physics_embed_dim=4 \
  prefix=pi_equibot \
  data.dataset.path=${PCS_DIR} \
  training.eval_interval=50 \
  training.save_interval=50 \
  training.vis_interval=100 \
  training.num_epochs=100 \
  exp_out_dir=${OUTPUT_DIR}/checkpoints/pi_model \
  use_wandb=true \
  wandb.name="pi_equibot" \
  wandb.project="pi_equibot"

echo "================================================="
echo "Training completed at $(date)"
echo "================================================="

# List the output directories to verify
echo "Original EquiBot checkpoints:"
ls -la ${OUTPUT_DIR}/checkpoints/original

echo "PI-EquiBot checkpoints:"
ls -la ${OUTPUT_DIR}/checkpoints/pi_model

# Print info for syncing wandb runs to cloud (if desired)
echo "To sync wandb runs to the cloud, run:"
find /fs/nexus-projects/Sketch_VLM_RL/equibit/wandb -name "offline-run-*" -type d | xargs -I{} echo "wandb sync {}" 