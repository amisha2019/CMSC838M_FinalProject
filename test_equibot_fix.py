import torch
from omegaconf import OmegaConf

from equibot.policies.agents.equibot_policy import EquiBotPolicy

def test_equibot_policy():
    # Create a simple config for testing
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
        }
    })

    print('Initializing EquiBotPolicy...')
    try:
        policy = EquiBotPolicy(cfg, device='cpu')
        print('Successfully initialized EquiBotPolicy!')
        return True
    except Exception as e:
        print(f'Error: {type(e).__name__}: {e}')
        return False

if __name__ == "__main__":
    test_equibot_policy() 