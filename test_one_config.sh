#!/bin/bash

# Base output directory
OUTPUT_BASE="/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision"

# Data directories for each task
FOLD_DATA="/fs/nexus-projects/Sketch_REBEL/equibot/data/fold_phy/pcs"
COVER_DATA="/fs/nexus-projects/Sketch_REBEL/equibot/data/cover_phy/pcs"
CLOSE_DATA="/fs/nexus-projects/Sketch_REBEL/equibot/data/close_phy/pcs"

# Number of epochs for training
NUM_EPOCHS=50

# Test function to check one configuration
test_one_config() {
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
    
    echo "Config file created at: $config_file"
    echo "Output directory will be: $output_dir"
    echo "Content of the config file:"
    grep -E 'prefix:|seed:|env_class:|path:|num_epochs:' $config_file
}

# Just test the fold task with seed 42
test_one_config "fold" "$FOLD_DATA" 42 