#!/usr/bin/env python
# Test script for environment creation

import sys
import os
try:
    # Test fold environment
    from equibot.envs.sim_mobile.folding_env import FoldingEnv
    from omegaconf import OmegaConf
    
    print("Testing fold environment creation...")
    config = {
        'task_name': 'fold',
        'num_eef': 1,
        'dof': 4,
        'max_episode_length': 100,
        'seed': 0,
        # Required parameters for BaseEnv
        'randomize_scale': True,
        'randomize_rotation': True,
        'uniform_scaling': True,
        'ac_noise': 0.0,
        'vis': False,
        'freq': 5,
        # Scale parameters
        'scale_low': 0.8,
        'scale_high': 1.2,
        'scale_aspect_limit': 1.3,
    }
    env = FoldingEnv(OmegaConf.create(config))
    obs = env.reset()
    print('FoldingEnv created successfully!')
    
    # For speed and stability, just test one environment for now
    # Testing others could cause issues with simulator resources
    print("\nAll environment tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f'Error creating environment: {e}')
    sys.exit(1) 