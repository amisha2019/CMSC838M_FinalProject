#!/bin/bash

# Base output directory
OUTPUT_BASE="/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision"

# Data directories for each task
FOLD_DATA="/fs/nexus-projects/Sketch_REBEL/equibot/data/fold_phy/pcs"
COVER_DATA="/fs/nexus-projects/Sketch_REBEL/equibot/data/cover_phy/pcs"
CLOSE_DATA="/fs/nexus-projects/Sketch_REBEL/equibot/data/close_phy/pcs"

# Seeds to use for training
SEEDS=(42 123 456)

# Number of epochs for training
NUM_EPOCHS=50

# Create logs directory in OUTPUT_BASE
mkdir -p "${OUTPUT_BASE}/logs"
mkdir -p "${OUTPUT_BASE}/configs"

# Function to create and submit job for a task
submit_job() {
    local task=$1
    local data_dir=$2
    local seed=$3
    
    # Create specific output directory for this task+seed
    local output_dir="${OUTPUT_BASE}/${task}_${seed}"
    mkdir -p "$output_dir"
    
    # Create configuration in the OUTPUT_BASE directory
    config_name="pi_phys_${task}_${seed}"
    config_file="${OUTPUT_BASE}/configs/${config_name}.yaml"
    
    # Copy the template config to nexus
    cp equibot/policies/configs/pi_phys.yaml $config_file
    
    # Update the config file for this specific task and seed
    sed -i "s|prefix: physics_supervised|prefix: physics_${task}_${seed}|g" $config_file
    sed -i "s|seed: .*|seed: ${seed}|g" $config_file
    sed -i "s|env_class: .*|env_class: ${task}|g" $config_file
    sed -i "s|path: .*|path: ${data_dir}|g" $config_file
    sed -i "s|num_epochs:.*|num_epochs: ${NUM_EPOCHS}|g" $config_file
    sed -i "s|device: cuda.*|device: cpu  # Using CPU due to PyTorch without CUDA support|g" $config_file
    
    # Create a symlink in the local configs directory pointing to the nexus config
    ln -sf $config_file "equibot/policies/configs/${config_name}.yaml"
    
    # Create a job-specific sbatch file directly in OUTPUT_BASE
    sbatch_file="${OUTPUT_BASE}/train_physics_${task}_${seed}.sbatch"
    cat > $sbatch_file << EOL
#!/bin/bash
#SBATCH --job-name=phy_${task}${seed}
#SBATCH --output=/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision/logs/phy_${task}_${seed}_%j.out
#SBATCH --error=/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision/logs/phy_${task}_${seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --cpus-per-task=4

# Create log directory if it doesn't exist
mkdir -p /fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision/logs

# Activate conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd


# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
# Set wandb to offline mode (can be changed to online if desired)
export WANDB_MODE=offline
# Set PYTHONPATH
export PYTHONPATH=/fs/cml-scratch/amishab/equibot:$PYTHONPATH

# Set output directory
export CUSTOM_OUTPUT_DIR=${output_dir}

# Run the training script with specific output directory
python -m equibot.policies.train --config-name ${config_name}
EOL

    # Make the sbatch file executable
    chmod +x $sbatch_file
    
    # Submit the job
    job_id=$(sbatch $sbatch_file | awk '{print $4}')
    
    echo "Submitted job for task: $task, seed: $seed with job ID: $job_id"
    echo "Output directory: $output_dir"
    echo "Config saved to: $config_file"
    echo "Sbatch file saved to: $sbatch_file"
    
    # Write job information to a file in the output directory
    cat > "${output_dir}/job_info.txt" << EOL
Task: $task
Seed: $seed
Job ID: $job_id
Submission time: $(date)
Sbatch file: $sbatch_file
Config file: $config_file
EOL
}

# Make sure the base output directory exists
echo "Using base output directory: $OUTPUT_BASE"

# Create and submit jobs for each task and seed combination
for seed in "${SEEDS[@]}"; do
    # Fold task
    submit_job "fold" "$FOLD_DATA" "$seed"
    
    # Cover task
    submit_job "cover" "$COVER_DATA" "$seed"
    
    # Close task
    submit_job "close" "$CLOSE_DATA" "$seed"
done

# Also save a copy of this submission script to the output base directory
cp "$0" "${OUTPUT_BASE}/submit_physics_training.sh"

echo "All jobs submitted!"
echo "Job files were saved to ${OUTPUT_BASE}" 