#!/usr/bin/env python3

"""
Test script to verify the training job can run for a few iterations without errors.
This is a simple script that sets up a minimal training environment and runs the main training function.
"""


import os
import sys
import torch
from omegaconf import OmegaConf

def create_test_config():
    """Create a minimal configuration for testing."""
    cfg = OmegaConf.create({
        'mode': 'train',
        'prefix': 'test_training',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_wandb': False,
        'log_dir': './test_logs',  # Add configurable log directory
        'data': {
            'dataset': {
                'path': '/fs/nexus-projects/Sketch_REBEL/equibot/data/close_phy/pcs',
                'dof': 4,
                'num_eef': 1,
                'eef_dim': 3,
                'num_training_steps': 500,
                'num_points': 1024,
                'num_augment': 0,
                'same_aug_per_sample': True,
                'aug_keep_original': True,
                'aug_scale_low': 0.5,
                'aug_scale_high': 1.5,
                'aug_scale_aspect_limit': 1.0,
                'aug_scale_rot': -1,
                'aug_scale_pos': 0.1,
                'aug_zero_z_offset': False,
                'aug_center': [0.0, 0.0, 0.0],
                'shuffle_pc': True,
                'num_workers': 1,  # Reduce workers to avoid warnings
                'reduce_horizon_dim': True,
                'min_demo_length': 16,
                'obs_horizon': 2,
                'pred_horizon': 16,
                'mixup_prob': 0.0  # Add-on #2: Physics-MixUp augmentation (disabled)
            }
        },
        'agent': {
            'agent_name': 'equibot'
        },
        'env': {
            'env_class': 'close',
            'args': {
                'task_name': 'close',
                'max_episode_length': 100,
                'num_eef': 1,
                'dof': 4,
                'seed': 0,
                'vis': False,
                'freq': 5,
                'randomize_scale': True,
                'randomize_rotation': True,
                'uniform_scaling': True,
                'ac_noise': 0.0,
                'scale_low': 0.8,
                'scale_high': 1.2,
                'scale_aspect_limit': 1.3
            },
            'dof': 4,
            'num_eef': 1,
            'eef_dim': 3
        },
        'model': {
            'obs_horizon': 2,
            'ac_horizon': 8,
            'pred_horizon': 16,
            'obs_mode': 'pc',
            'ac_mode': 'diffusion',
            'hidden_dim': 256,
            'encoder': {
                'c_dim': 256,
                'backbone_type': 'vn_pointnet',
                'backbone_args': {
                    'num_layers': 4,
                    'knn': 8
                }
            },
            'use_torch_compile': False,
            'noise_scheduler': {
                '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                'num_train_timesteps': 100,
                'beta_schedule': 'squaredcos_cap_v2',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            },
            'use_physics_embed': True,
            'physics_embed_dim': 4
        },
        'seed': 0,
        'training': {
            'num_epochs': 1,  # Just run 1 epoch for testing
            'batch_size': 4,   # Small batch size for testing
            'lr': 3e-5,
            'weight_decay': 1e-6,
            'save_interval': 50,
            'vis_interval': 100,
            'eval_interval': 50,
            'num_eval_episodes': 1,
            'use_curriculum': False  # Add curriculum parameter
        },
        'wandb': {
            'project': 'equibot_test',
            'entity': 'amishab'
        }
    })
    return cfg

def test_training():
    """Test if the training code can run without errors."""
    # Ensure the current directory is in the Python path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Set environment variable for library compatibility
    os.environ['LD_PRELOAD'] = os.environ.get('CONDA_PREFIX', '') + '/lib/libstdc++.so.6'
    
    try:
        # Import main function from train.py
        from equibot.policies.train import main
        
        # Create test config
        cfg = create_test_config()
        
        # Run a single epoch of training
        print("Running training with test configuration...")
        main(cfg)
        
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("\nTest passed! The training job is working correctly.")
        exit(0)
    else:
        print("\nTest failed! There are still issues with the training job.")
        exit(1) 