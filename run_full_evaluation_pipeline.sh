#!/bin/bash
# Master script for running the entire PI-EquiBot evaluation pipeline

set -e  # Exit on error

# Configuration
BASE_DIR="/fs/nexus-projects/Sketch_REBEL/equibot"
CHECKPOINTS_DIR="${BASE_DIR}/checkpoints/evaluation"
RESULTS_DIR="${BASE_DIR}/results/evaluation"
ANALYSIS_DIR="${BASE_DIR}/analysis/evaluation"

# Create directories
mkdir -p ${CHECKPOINTS_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${ANALYSIS_DIR}

echo "=============================================="
echo "    PI-EquiBot Full Evaluation Pipeline      "
echo "=============================================="
echo "1. Submit all training jobs (parallel)"
echo "2. Submit evaluation grid job with dependency"
echo "3. Generate analysis and plots when done"
echo "=============================================="
echo "Checkpoint directory: ${CHECKPOINTS_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Analysis directory: ${ANALYSIS_DIR}"
echo "=============================================="

# Submit training jobs and collect their job IDs
echo "Submitting training jobs..."
TRAINING_MASTER_ID=$(sbatch submit_training_evaluation.sh | awk '{print $4}')
echo "Training master job submitted with ID ${TRAINING_MASTER_ID}"

# Function to check if any jobs from a specific submission are still running
check_jobs_from_tracking_file() {
    local tracking_file="$1"
    local running_jobs=0
    
    if [ -f "$tracking_file" ]; then
        # Skip header line and get job IDs
        job_ids=$(tail -n +2 "$tracking_file" | cut -d',' -f1)
        
        for job_id in $job_ids; do
            if squeue -j "$job_id" &>/dev/null; then
                running_jobs=$((running_jobs + 1))
            fi
        done
    fi
    
    return $running_jobs
}

# Wait for training jobs to complete before proceeding
wait_for_training_completion() {
    local tracking_file="${BASE_DIR}/logs/evaluation/submitted_jobs.txt"
    
    echo "Waiting for all training jobs to complete..."
    
    while true; do
        # First check if the master job is still running
        if squeue -j ${TRAINING_MASTER_ID} &>/dev/null; then
            echo "Training master job ${TRAINING_MASTER_ID} is still running. Waiting for 5 minutes..."
            sleep 300  # Wait for 5 minutes
            continue
        fi
        
        # Then check if the tracking file exists (should be created by the master job)
        if [ ! -f "$tracking_file" ]; then
            echo "Job tracking file not found yet. Waiting for 5 minutes..."
            sleep 300
            continue
        fi
        
        # Check if any jobs from the tracking file are still running
        check_jobs_from_tracking_file "$tracking_file"
        running_jobs=$?
        
        if [ $running_jobs -eq 0 ]; then
            echo "All training jobs have completed."
            break
        else
            echo "${running_jobs} training jobs are still running. Waiting for 5 minutes..."
            sleep 300
        fi
    done
}

# Either set dependencies or wait for completion
if command -v sbatch &> /dev/null && [ -n "$(sbatch --help | grep -e "--dependency")" ]; then
    # If sbatch supports dependencies, use them
    echo "Training master job submitted. Submitting evaluation grid job with dependency..."
    EVAL_JOB_ID=$(sbatch --dependency=afterok:${TRAINING_MASTER_ID} submit_evaluation_grid.sh | awk '{print $4}')
    echo "Evaluation grid job submitted with ID ${EVAL_JOB_ID} (will start after training completes)"
    
    # Submit analysis as a dependent job
    echo "Submitting analysis job with dependency on evaluation grid job..."
    ANALYSIS_SCRIPT="${BASE_DIR}/logs/evaluation/run_analysis.sh"
    
    # Create analysis script
    cat > ${ANALYSIS_SCRIPT} << EOF
#!/bin/bash
#SBATCH --job-name=equibot_analysis
#SBATCH --output=${BASE_DIR}/logs/evaluation/analysis_%j.log
#SBATCH --error=${BASE_DIR}/logs/evaluation/analysis_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Set working directory
cd /fs/cml-scratch/amishab/equibot

# Load conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Run analysis
python analysis_plots.py --results_dir ${RESULTS_DIR} --output_dir ${ANALYSIS_DIR} --combined

echo "Analysis complete. Results available in ${ANALYSIS_DIR}"
EOF
    
    chmod +x ${ANALYSIS_SCRIPT}
    ANALYSIS_JOB_ID=$(sbatch --dependency=afterok:${EVAL_JOB_ID} ${ANALYSIS_SCRIPT} | awk '{print $4}')
    echo "Analysis job submitted with ID ${ANALYSIS_JOB_ID} (will start after evaluation completes)"
    
    echo "All jobs submitted. You can check their status with: squeue -u $USER"
    echo "Pipeline will run automatically through dependencies."
else
    # If dependencies are not supported, wait for jobs to complete sequentially
    wait_for_training_completion
    
    # Submit evaluation grid job
    echo "Training completed. Submitting evaluation grid job..."
    EVAL_JOB_ID=$(sbatch submit_evaluation_grid.sh | awk '{print $4}')
    echo "Evaluation grid job submitted with ID ${EVAL_JOB_ID}"
    
    # Wait for evaluation job to complete
    echo "Waiting for evaluation job ${EVAL_JOB_ID} to complete..."
    while squeue -j ${EVAL_JOB_ID} &>/dev/null; do
        echo "Evaluation job ${EVAL_JOB_ID} is still running. Waiting for 5 minutes..."
        sleep 300  # Wait for 5 minutes
    done
    
    # Run analysis directly
    echo "Evaluation completed. Running analysis..."
    python analysis_plots.py --results_dir ${RESULTS_DIR} --output_dir ${ANALYSIS_DIR} --combined
fi

echo "Pipeline setup complete!"
echo "Results will be available in:"
echo "- Checkpoints: ${CHECKPOINTS_DIR}"
echo "- Evaluation results: ${RESULTS_DIR}"
echo "- Analysis and plots: ${ANALYSIS_DIR}" 