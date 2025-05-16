#!/usr/bin/env python3

"""
Test script to diagnose tensor shape issues in the EquiBotPolicy.
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.abspath('.'))

# Set environment variable for library compatibility
os.environ['LD_PRELOAD'] = os.environ.get('CONDA_PREFIX', '') + '/lib/libstdc++.so.6'

def test_tensor_shapes():
    """Test the tensor shapes in the EquiBotPolicy."""
    try:
        from equibot.policies.agents.equibot_policy import EquiBotPolicy
        
        # Create minimal config
        cfg = OmegaConf.create({
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
                    'num_train_timesteps': 100, 
                    'beta_schedule': 'squaredcos_cap_v2', 
                    'clip_sample': True, 
                    'prediction_type': 'epsilon'
                }, 
                'use_physics_embed': True, 
                'physics_embed_dim': 4
            },
            'env': {
                'num_eef': 1,
                'dof': 4
            }
        })
        
        print('Initializing EquiBotPolicy...')
        policy = EquiBotPolicy(cfg, device='cpu')
        print('Successfully initialized EquiBotPolicy!')
        
        # Create a dummy batch with the suspected problematic shape
        # Test the _convert_action_to_vec method
        B = 4  # Batch size
        T = 16  # Sequence length
        num_eef = 1  # Number of end effectors
        dof = 4  # Degrees of freedom (3 for position, 1 for gripper)
        
        # Create action tensor with shape [B, T, num_eef*dof]
        action = torch.randn(B, T, num_eef * dof)
        print(f"Action shape: {action.shape}, total elements: {action.numel()}")
        
        # Test the _convert_action_to_vec method
        try:
            vec_eef_action, vec_gripper_action = policy._convert_action_to_vec(action)
            print(f"vec_eef_action shape: {vec_eef_action.shape}")
            if vec_gripper_action is not None:
                print(f"vec_gripper_action shape: {vec_gripper_action.shape}")
            
            # Test converting back
            reconstructed_action = policy._convert_action_to_scalar(vec_eef_action, vec_gripper_action)
            print(f"Reconstructed action shape: {reconstructed_action.shape}")
            
            # Check if shapes match
            if reconstructed_action.shape == action.shape:
                print("✅ Tensor conversion methods are working correctly!")
            else:
                print(f"❌ Shape mismatch: {reconstructed_action.shape} vs {action.shape}")
                
            # Additional test with problematic shape [4, 16, 1, 4]
            problematic_action = torch.randn(4, 16, 1, 4)
            print(f"Problematic action shape: {problematic_action.shape}, total elements: {problematic_action.numel()}")
            
            # Make the size 896 to match the error
            weird_action = torch.randn(896)
            print(f"Weird action shape: {weird_action.shape}, total elements: {weird_action.numel()}")
            
            # Try reshaping the weird action to the problematic shape
            try:
                reshaped_weird = weird_action.reshape(4, 16, 1, 4)
                print("Weird reshaping worked!")
            except RuntimeError as e:
                print(f"Expected reshape error: {e}")
            
            return True
        except Exception as e:
            print(f"Error in conversion methods: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f'Error: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tensor_shapes() 