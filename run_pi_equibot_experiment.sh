#!/bin/bash

echo "==================================================================="
echo "PI-EquiBot: Physically-Informed EquiBot Experiment"
echo "==================================================================="
echo ""

# Make all scripts executable
chmod +x pi_equibot_generate_data.sh
chmod +x pi_equibot_train.sh
chmod +x pi_equibot_eval.sh

echo "Starting the full experiment pipeline:"
echo "1. Generate data with physics variations"
echo "2. Train both original EquiBot and PI-EquiBot"
echo "3. Evaluate both models on in-distribution and OOD scenarios"
echo ""

# Submit data generation job and wait for it to complete
echo "Submitting data generation job..."
data_job_id=$(sbatch --parsable pi_equibot_generate_data.sh)
echo "Data generation job submitted with ID: $data_job_id"

# Wait for data generation to complete before submitting training job
echo "Waiting for data generation to complete before starting training..."
# Check job status every 60 seconds
while squeue -j $data_job_id -h &> /dev/null; do
    echo "Data generation job $data_job_id is still running... ($(date))"
    sleep 60
done
echo "Data generation job completed!"

# Check if data generation was successful
DATA_DIR="/fs/cml-scratch/amishab/data/fold_physics/pcs"
if [ ! -d "${DATA_DIR}" ] || [ $(find "${DATA_DIR}" -name "*.npz" | wc -l) -eq 0 ]; then
    echo "ERROR: Data generation failed. No data files found in ${DATA_DIR}"
    echo "Check the log file for details."
    exit 1
fi

# Submit training job
echo "Submitting training job..."
train_job_id=$(sbatch --parsable pi_equibot_train.sh)
echo "Training job submitted with ID: $train_job_id"

# Wait for training to complete before submitting evaluation job
echo "Waiting for training to complete before starting evaluation..."
# Check job status every 5 minutes
while squeue -j $train_job_id -h &> /dev/null; do
    echo "Training job $train_job_id is still running... ($(date))"
    sleep 300
done
echo "Training job completed!"

# Submit evaluation job
echo "Submitting evaluation job..."
eval_job_id=$(sbatch --parsable pi_equibot_eval.sh)
echo "Evaluation job submitted with ID: $eval_job_id"

echo ""
echo "Full experiment pipeline is running:"
echo "- Data Generation: Job $data_job_id"
echo "- Training: Job $train_job_id"
echo "- Evaluation: Job $eval_job_id"
echo ""
echo "When all jobs have completed, you can find the results in:"
echo "- /fs/nexus-projects/Sketch_VLM_RL/equibit/checkpoints/ (model checkpoints)"
echo "- /fs/nexus-projects/Sketch_VLM_RL/equibit/eval/videos/ (evaluation videos)"
echo ""
echo "To sync wandb results to cloud (if desired):"
echo "cd /fs/cml-scratch/amishab/equibot && wandb sync --sync-all" 