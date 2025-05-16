#!/bin/bash

# Training script for equibot
# This script includes fixes for the issues identified in the codebase

# Set up environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export PYTHONPATH=/fs/cml-scratch/amishab/equibot:$PYTHONPATH

# Default values
TASK="close"  # Task name (e.g., close, fold, etc.)
VARIANT="default"  # Variant name
SEED=0           # Random seed
DATA_ROOT="/fs/nexus-projects/Sketch_REBEL/equibot/data"  # Path to data
LOG_ROOT="./logs"  # Root directory for logs

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --task)
      TASK="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set up paths
DATA_PATH="${DATA_ROOT}/${TASK}_phy/pcs"
LOG_DIR="${LOG_ROOT}/${TASK}/${VARIANT}_seed${SEED}"

# Make sure log directory exists
mkdir -p $LOG_DIR

# Print configuration
echo "Running training with configuration:"
echo "  Task: $TASK"
echo "  Variant: $VARIANT"
echo "  Seed: $SEED"
echo "  Data path: $DATA_PATH"
echo "  Log directory: $LOG_DIR"

# Check if data directory exists
if [ ! -d "$DATA_PATH" ]; then
  echo "Error: Data directory $DATA_PATH does not exist!"
  echo "Please make sure the data is available before running training."
  exit 1
fi

# Run training with proper configuration
cd /fs/cml-scratch/amishab/equibot
python -m equibot.policies.train \
  mode=train \
  prefix=${VARIANT} \
  data.dataset.path=${DATA_PATH} \
  env.args.max_episode_length=100 \
  env.args.num_points=1024 \
  log_dir=${LOG_DIR} \
  seed=${SEED} 