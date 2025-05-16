#!/bin/bash

# Script to submit multiple EquiBot training jobs with different parameters
# This version uses the fixed training script that handles PyTorch3D GPU compatibility
# Usage: ./submit_training_jobs_fixed.sh [variants] [tasks] [seeds] [use_wandb]
# Example: ./submit_training_jobs_fixed.sh "A B C" "fold cover" "0 1" "true"

# Default parameters if not provided
VARIANTS=${1:-"A B C D E"}  # Default: all variants
TASKS=${2:-"fold cover close"}  # Default: all tasks
SEEDS=${3:-"0 1 2"}  # Default: seeds 0, 1, 2
USE_WANDB=${4:-"true"}  # Default: use W&B logging

# Set output directories
LOG_DIR="/fs/cml-scratch/amishab/equibot/logs"
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/jobs

# Create job tracking file
JOB_TRACKING_FILE="${LOG_DIR}/training_jobs_fixed_$(date +%Y%m%d_%H%M%S).txt"
echo "JobID,Variant,Task,Seed,Status,SubmitTime" > ${JOB_TRACKING_FILE}

# Print job submission summary
echo "===================================================="
echo "EquiBot Training Jobs Submission (with PyTorch3D Fix)"
echo "===================================================="
echo "Variants: ${VARIANTS}"
echo "Tasks: ${TASKS}"
echo "Seeds: ${SEEDS}"
echo "W&B logging: ${USE_WANDB}"
echo "Job tracking file: ${JOB_TRACKING_FILE}"
echo "===================================================="

# Make the training script executable
chmod +x train_equibot_gpu_fixed.sh

# Submit jobs for each combination
for variant in ${VARIANTS}; do
    for task in ${TASKS}; do
        for seed in ${SEEDS}; do
            job_name="equibot_fixed_${variant}_${task}_s${seed}"
            echo "Submitting job: ${job_name}"
            
            # Submit the job with environment variables
            JOB_ID=$(VARIANT=${variant} TASK=${task} SEED=${seed} USE_WANDB=${USE_WANDB} \
                    sbatch --job-name=${job_name} \
                    --output=${LOG_DIR}/jobs/${job_name}_%j.log \
                    --error=${LOG_DIR}/jobs/${job_name}_%j.err \
                    train_equibot_gpu_fixed.sh | awk '{print $4}')
            
            # Record job information
            submit_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "${JOB_ID},${variant},${task},${seed},Submitted,${submit_time}" >> ${JOB_TRACKING_FILE}
            
            echo "Submitted job ${job_name} with ID: ${JOB_ID}"
            
            # Add a small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

echo "===================================================="
echo "All training jobs have been submitted with PyTorch3D fix."
echo "Total jobs: $(grep -c "Submitted" ${JOB_TRACKING_FILE})"
echo "Monitor job status with: cat ${JOB_TRACKING_FILE}"
echo "Or check individual job status with: squeue -u $USER"
echo "====================================================" 