#!/usr/bin/env python3
import os
import yaml
import shutil
from pathlib import Path

# Source directory with config files
source_dir = "/fs/cml-scratch/amishab/equibot/equibot/policies/configs"

# Target base directory
target_base_dir = "/fs/nexus-projects/Sketch_REBEL/equibot/physics_supervision_new"

# List of tasks
tasks = ["fold", "close", "cover"]

# Ensure target directories exist
for task in tasks:
    for seed in ["42", "123", "456"]:
        task_dir = os.path.join(target_base_dir, f"{task}_{seed}")
        os.makedirs(task_dir, exist_ok=True)
        print(f"Created directory: {task_dir}")

# Find all pi_phys_* config files
config_files = [f for f in os.listdir(source_dir) if f.startswith("pi_phys_") and f != "pi_phys.yaml" and f.endswith(".yaml")]

# Process each config file
for config_file in config_files:
    # Load the config file
    config_path = os.path.join(source_dir, config_file)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the config values
    config['device'] = 'cuda'  # Set device to cuda
    config['training']['num_epochs'] = 2000  # Set num_epochs to 2000
    
    # Add num_demos parameter to limit to 50 demos
    if 'data' in config and 'dataset' in config['data']:
        config['data']['dataset']['num_demos'] = 50
    
    # Determine the task and seed from the filename
    parts = config_file.replace('.yaml', '').split('_')
    task = parts[-2]  # e.g., "fold", "close", "cover"
    seed = parts[-1]  # e.g., "42", "123", "456"
    
    # Create target directory if it doesn't exist
    target_dir = os.path.join(target_base_dir, f"{task}_{seed}")
    
    # Set the target filepath
    target_path = os.path.join(target_dir, config_file)
    
    # Write the updated config to the target location
    with open(target_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated and moved {config_file} to {target_path}")

print("Config files have been updated and moved to their target locations.") 