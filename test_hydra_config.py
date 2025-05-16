#!/usr/bin/env python
# Test script for Hydra configuration

import sys
from typing import Any
from dataclasses import dataclass, field

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    
    @dataclass
    class EnvArgs:
        task_name: str = 'fold'
        num_eef: int = 1
        dof: int = 4
        max_episode_length: int = 100
        seed: int = 0
        randomize_scale: bool = True
        randomize_rotation: bool = True
        uniform_scaling: bool = True
        ac_noise: float = 0.0
        vis: bool = False
        freq: int = 5
        scale_low: float = 0.8
        scale_high: float = 1.2
        scale_aspect_limit: float = 1.3
    
    def default_env_args() -> EnvArgs:
        return EnvArgs()
    
    @dataclass
    class EnvConfig:
        env_class: str = 'fold'
        dof: int = 4
        num_eef: int = 1
        eef_dim: int = 3
        args: EnvArgs = field(default_factory=default_env_args)
    
    @dataclass
    class Config:
        env: EnvConfig = field(default_factory=lambda: EnvConfig())
    
    cs = ConfigStore.instance()
    cs.store(name='config', node=Config)
    
    @hydra.main(config_path=None, config_name='config', version_base=None)
    def test_hydra(cfg):
        print('Hydra config loaded successfully!')
        print(OmegaConf.to_yaml(cfg))
        # Verify we can access env parameters correctly
        assert cfg.env.num_eef == 1, "env.num_eef should be 1"
        assert cfg.env.args.num_eef == 1, "env.args.num_eef should be 1"
        print("All assertions passed!")
        
    # Run the Hydra function
    test_hydra()
    sys.exit(0)
    
except Exception as e:
    print(f'Error with Hydra configuration: {e}')
    sys.exit(1) 