#!/usr/bin/env python3
import numpy as np
import os

# Create a demo trajectory with at least min_demo_length files
min_demo_length = 16  # This matches the min_demo_length in the config
demo_name = 'demo1'
data_dir = '/fs/nexus-projects/Sketch_REBEL/equibot/data/close_phy/pcs'

for i in range(min_demo_length + 5):  # Create a few extra frames
    filename = f'{demo_name}_t{i:02d}.npz'
    filepath = os.path.join(data_dir, filename)
    
    # Create data with correct shapes
    pc = np.random.randn(1024, 3).astype(np.float32)  # Point cloud
    eef_pos = np.random.randn(1, 4).astype(np.float32)  # End effector position
    action = np.random.randn(4).astype(np.float32)  # Action (matches dof=4)
    physics_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)  # Physics parameters
    
    # Save to npz file
    np.savez(filepath, pc=pc, eef_pos=eef_pos, action=action, physics_vec=physics_vec)
    
    print(f'Created {filename}')

print(f'Created {min_demo_length + 5} demo files') 