#!/bin/bash
#SBATCH --job-name=equibot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# This script submits a SLURM job array to train models with different configurations
# Usage: sbatch submit_jobs.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Define task and variant combinations
# Format: task1 variant1 task2 variant2 etc.
TASKS=("close" "close" "close" "fold" "fold" "fold" "pour" "pour" "pour")
VARIANTS=("baseline" "physics" "curriculum" "baseline" "physics" "curriculum" "baseline" "physics" "curriculum")
SEEDS=(0 0 0 0 0 0 0 0 0)

# Get the current job array index
JOB_IDX=$SLURM_ARRAY_TASK_ID

if [ -z "$JOB_IDX" ]; then
    echo "Error: This script should be run with sbatch as a job array."
    exit 1
fi

# Extract the task, variant, and seed for this job
TASK=${TASKS[$JOB_IDX]}
VARIANT=${VARIANTS[$JOB_IDX]}
SEED=${SEEDS[$JOB_IDX]}

# Set data root - replace this with the path to your data
DATA_ROOT="/fs/nexus-projects/Sketch_REBEL/equibot/data"

# Set log root - this will create unique directories for each job
LOG_ROOT="/fs/cml-scratch/amishab/equibot/experiment_logs"

# Run the training script with the selected configuration
bash run_training.sh --task $TASK --variant $VARIANT --seed $SEED --data_root $DATA_ROOT --log_root $LOG_ROOT

exit 0 