#!/usr/bin/env python3
"""
Test script to verify the encoder_handle fix for EquiBotPolicy.
This script creates a simple EquiBotPolicy instance and tests if it has the encoder_handle attribute.
"""

import torch
import hydra
import os
from omegaconf import OmegaConf
from equibot.policies.agents.equibot_policy import EquiBotPolicy

# Enable colored output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def main():
    print("Testing EquiBotPolicy encoder_handle fix...")
    
    # Create a basic configuration
    cfg = OmegaConf.create({
        "model": {
            "obs_mode": "pc",
            "ac_mode": "diffusion",
            "hidden_dim": 32,
            "use_torch_compile": True,
            "noise_scheduler": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "prediction_type": "epsilon"
            },
            "encoder": {
                "c_dim": 32,
                "backbone_type": "vn_pointnet",
                "backbone_args": {
                    "num_layers": 2,
                    "knn": 4
                }
            },
            "use_physics_embed": False,
            "physics_embed_dim": 4
        },
        "env": {
            "num_eef": 1,
            "eef_dim": 3,
            "dof": 4
        }
    })
    
    # Initialize the policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        policy = EquiBotPolicy(cfg, device=device)
        
        # Test if encoder_handle exists
        if hasattr(policy, 'encoder_handle'):
            print(f"{GREEN}✓ Success: encoder_handle attribute exists{RESET}")
        else:
            print(f"{RED}✗ Error: encoder_handle attribute doesn't exist{RESET}")
            return False
            
        # Test if noise_pred_net_handle exists
        if hasattr(policy, 'noise_pred_net_handle'):
            print(f"{GREEN}✓ Success: noise_pred_net_handle attribute exists{RESET}")
        else:
            print(f"{RED}✗ Error: noise_pred_net_handle attribute doesn't exist{RESET}")
            return False
            
        # Test with torch.compile disabled
        cfg.model.use_torch_compile = False
        policy_no_compile = EquiBotPolicy(cfg, device=device)
        
        if hasattr(policy_no_compile, 'encoder_handle'):
            print(f"{GREEN}✓ Success: encoder_handle exists even with use_torch_compile=False{RESET}")
        else:
            print(f"{RED}✗ Error: encoder_handle doesn't exist with use_torch_compile=False{RESET}")
            return False
            
        print(f"{GREEN}All tests passed! The EquiBotPolicy encoder_handle fix works correctly.{RESET}")
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error during testing: {str(e)}{RESET}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 