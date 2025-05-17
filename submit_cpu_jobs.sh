#!/bin/bash

# Array of tasks and seeds
TASKS=("fold" "close" "cover")
SEEDS=("42" "123" "456")

# Cancel any existing jobs for these tasks
echo "Cancelling any existing jobs..."
for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="phy_${TASK}_${SEED}"
        JOBID=$(squeue -h -n $JOB_NAME -o %A)
        if [ ! -z "$JOBID" ]; then
            echo "Cancelling job $JOBID ($JOB_NAME)"
            scancel $JOBID
        fi
    done
done

# Submit jobs for all combinations
for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting job for $TASK with seed $SEED"
        
        # Set job name based on task and seed
        JOB_NAME="phy_${TASK}_${SEED}"
        
        # Submit the job with task-specific job name - USING CPU VERSION
        sbatch --job-name=$JOB_NAME train_physics_fixed.sbatch $TASK $SEED
        
        # Wait a bit to avoid overwhelming the scheduler
        sleep 1
    done
done

echo "All jobs submitted. Check status with: squeue -u $(whoami)"
echo "Logs will be in: /fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new/logs/" 