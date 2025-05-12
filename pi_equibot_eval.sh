#!/bin/bash
#SBATCH --job-name=pi_equibot_eval
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_equibot_eval_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_equibot_eval_%j.err
#SBATCH --time=4:00:00
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

# Create output directory
OUTPUT_DIR=/fs/nexus-projects/Sketch_VLM_RL/equibit
EVAL_DIR=${OUTPUT_DIR}/eval
mkdir -p ${EVAL_DIR}/videos/original
mkdir -p ${EVAL_DIR}/videos/pi_model
chmod -R 775 ${EVAL_DIR}

# Define checkpoint paths
ORIGINAL_CKPT=${OUTPUT_DIR}/checkpoints/original/ckpt_latest.pth
PI_CKPT=${OUTPUT_DIR}/checkpoints/pi_model/ckpt_latest.pth

# Check if checkpoints exist
if [ ! -f "${ORIGINAL_CKPT}" ]; then
  echo "Error: Original model checkpoint not found at ${ORIGINAL_CKPT}"
  echo "Please run pi_equibot_train.sh first."
  exit 1
fi

if [ ! -f "${PI_CKPT}" ]; then
  echo "Error: PI model checkpoint not found at ${PI_CKPT}"
  echo "Please run pi_equibot_train.sh first."
  exit 1
fi

echo "Found checkpoints:"
echo "- Original: ${ORIGINAL_CKPT}"
echo "- PI-EquiBot: ${PI_CKPT}"

# Print timestamp
echo "Starting evaluation at $(date)"

# Physical parameters to test
# Format: "mass friction stiffness name"
PHYSICS_PARAMS=(
  "0.5 1.0 150.0 normal"           # Normal (in-distribution)
  "2.0 1.0 150.0 heavy"            # Heavy object
  "0.1 1.0 150.0 light"            # Light object
  "0.5 5.0 150.0 high_friction"    # High friction
  "0.5 0.5 150.0 low_friction"     # Low friction
  "0.5 1.0 300.0 high_stiffness"   # High stiffness
  "0.5 1.0 100.0 low_stiffness"    # Low stiffness
  "2.0 5.0 300.0 extreme"          # Out of distribution extreme
)

# Evaluate each model with different physical parameters
for model_type in "original" "pi_model"; do
  if [ "$model_type" == "original" ]; then
    CKPT=${ORIGINAL_CKPT}
    MODEL_NAME="Original EquiBot"
    MODEL_FLAG="--model.use_physics_embed=false"
  else
    CKPT=${PI_CKPT}
    MODEL_NAME="PI-EquiBot"
    MODEL_FLAG="--model.use_physics_embed=true --model.physics_embed_dim=4"
  fi
  
  echo "================================================="
  echo "Evaluating ${MODEL_NAME}..."
  echo "================================================="
  
  for params in "${PHYSICS_PARAMS[@]}"; do
    # Parse parameters
    read -r mass friction stiffness name <<< "$params"
    
    echo "Testing with parameters: mass=${mass}, friction=${friction}, stiffness=${stiffness} (${name})"
    
    # Run evaluation
    python -m equibot.policies.eval \
      --config-name fold_mobile_equibot \
      ${MODEL_FLAG} \
      --ckpt_path ${CKPT} \
      --record_video \
      --video_path ${EVAL_DIR}/videos/${model_type}/${name}.mp4 \
      --deform_object_mass ${mass} \
      --deform_friction_coeff ${friction} \
      --deform_elastic_stiffness ${stiffness} \
      --num_episodes 5 \
      --seed 42 \
      --use_wandb=false
      
    # Check if video was generated
    if [ -f "${EVAL_DIR}/videos/${model_type}/${name}.mp4" ]; then
      echo "✓ Video saved to ${EVAL_DIR}/videos/${model_type}/${name}.mp4"
    else
      echo "✗ Failed to generate video for ${name} parameters"
    fi
  done
done

echo "================================================="
echo "Evaluation completed at $(date)"
echo "================================================="

echo "Video files generated:"
find ${EVAL_DIR}/videos -name "*.mp4" | sort 