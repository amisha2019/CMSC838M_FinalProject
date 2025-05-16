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

# Function to create and submit job for a task
submit_job() {
    local task=$1
    local data_dir=$2
    local seed=$3
    
    # Create specific output directory for this task+seed
    local output_dir="${OUTPUT_BASE}/${task}_${seed}"
    mkdir -p "$output_dir"
    
    # Create a task-specific config file
    config_file="equibot/policies/configs/pi_phys_${task}_${seed}.yaml"
    cp equibot/policies/configs/pi_phys.yaml $config_file
    
    # Update the config file for this specific task and seed
    sed -i "s|prefix: physics_supervised|prefix: physics_${task}_${seed}|g" $config_file
    sed -i "s|seed: .*|seed: ${seed}|g" $config_file
    sed -i "s|env_class: .*|env_class: ${task}|g" $config_file
    sed -i "s|path: .*|path: ${data_dir}|g" $config_file
    sed -i "s|num_epochs:.*|num_epochs: ${NUM_EPOCHS}|g" $config_file
    
    # Create a job-specific sbatch file
    sbatch_file="train_physics_${task}_${seed}.sbatch"
    cat > $sbatch_file << EOL
#!/bin/bash
#SBATCH --job-name=phy_${task}${seed}
#SBATCH --output=logs/phy_${task}_${seed}_%j.out
#SBATCH --error=logs/phy_${task}_${seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Create log directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set PYTHONPATH
export PYTHONPATH=/fs/cml-scratch/amishab/equibot:$PYTHONPATH

# Set output directory
export CUSTOM_OUTPUT_DIR=${output_dir}

# Run the training script with specific output directory
python -m equibot.policies.train --config-name pi_phys_${task}_${seed}
EOL

    # Make the sbatch file executable
    chmod +x $sbatch_file
    
    # Submit the job
    sbatch $sbatch_file
    
    echo "Submitted job for task: $task, seed: $seed with output directory: $output_dir"
}

# Make sure the log directory exists
mkdir -p logs

# Make sure the base output directory exists
mkdir -p "$OUTPUT_BASE"
echo "Created base output directory: $OUTPUT_BASE"

# Create and submit jobs for each task and seed combination
for seed in "${SEEDS[@]}"; do
    # Fold task
    submit_job "fold" "$FOLD_DATA" "$seed"
    
    # Cover task
    submit_job "cover" "$COVER_DATA" "$seed"
    
    # Close task
    submit_job "close" "$CLOSE_DATA" "$seed"
done

echo "All jobs submitted!" 