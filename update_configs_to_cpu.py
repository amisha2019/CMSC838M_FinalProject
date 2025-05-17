#!/usr/bin/env python3
"""
Script to update all pi_phys config files to use CPU instead of CUDA
"""

import os
import glob
import re
import yaml

CONFIG_DIR = "/fs/cml-scratch/amishab/equibot/configs"

def update_config_file(file_path):
    """Update the device setting in a config file to use CPU"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace 'device: cuda' with 'device: cpu'
        updated_content = re.sub(r'device:\s*cuda', 'device: cpu', content)
        
        # Write back the updated content
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated {file_path}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def main():
    # Find all pi_phys config files
    config_files = glob.glob(os.path.join(CONFIG_DIR, "pi_phys*.yaml"))
    
    print(f"Found {len(config_files)} config files to update")
    
    # Update each file
    for config_file in config_files:
        update_config_file(config_file)
    
    print("All config files updated to use CPU.")

if __name__ == "__main__":
    main() 