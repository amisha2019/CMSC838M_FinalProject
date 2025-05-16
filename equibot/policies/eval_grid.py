"""
Grid evaluation script for PI-EquiBot variants.

This script evaluates a trained model across a grid of physics parameters
and logs the results to a CSV file.
"""

import os
import sys
import time
import torch
import hydra
import argparse
import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from equibot.policies.utils.misc import get_env_class, get_agent
from equibot.policies.utils.test_time_adaptation import apply_test_time_adaptation


@hydra.main(config_path="configs", config_name="evaluation/variant_E", version_base=None)
def main(cfg):
    """Main entry point for grid evaluation."""
    # Set up environment and agent
    task = cfg.task
    model_path = cfg.model
    variant = cfg.variant
    seed = cfg.seed
    use_adapt = cfg.adapt == 1
    num_episodes = cfg.episodes
    save_path = cfg.save
    verbose = cfg.verbose
    
    # Print configuration
    print(f"Task: {task}")
    print(f"Model: {model_path}")
    print(f"Variant: {variant}")
    print(f"Seed: {seed}")
    print(f"Use adaptation: {use_adapt}")
    print(f"Episodes per config: {num_episodes}")
    print(f"Save path: {save_path}")
    print(f"Verbose: {verbose}")
    
    # Ensure all necessary directories exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Get parameter grid for evaluation
    parameter_grid = get_parameter_grid(task)
    
    # Set up environment and agent
    env, agent, _ = setup_environment(cfg)
    
    # Run grid evaluation
    results = []
    for params in tqdm(parameter_grid, desc="Evaluating parameter configurations"):
        for episode in range(num_episodes):
            result = run_episode(env, agent, params, use_adapt)
            if verbose:
                print(f"Params: {params}, Success: {result['success']}, Time: {result['time']}")
            results.append(result)
    
    # Save results to CSV
    save_results(results, save_path)
    print(f"Results saved to {save_path}")
    
    # Return success rate for easy monitoring
    success_rate = sum(r['success'] for r in results) / len(results)
    print(f"Overall success rate: {success_rate:.2f}")
    return success_rate


def get_parameter_grid(task):
    """Get the parameter grid for a specific task."""
    # Mass and friction apply to all tasks
    mass_values = [0.5, 1.0, 3.0]
    friction_values = [0.2, 0.5, 1.2]
    
    if task in ['fold', 'cover']:
        # For cloth tasks, include stiffness and damping
        stiffness_values = [0.3, 0.6, 1.0]
        damping_values = [0.05, 0.2, 0.5]
        
        # Generate full combinatorial grid
        grid = []
        for mass in mass_values:
            for friction in friction_values:
                for stiffness in stiffness_values:
                    for damping in damping_values:
                        grid.append({
                            'mass': mass,
                            'friction': friction,
                            'stiffness': stiffness,
                            'damping': damping
                        })
    else:
        # For close task, only use mass and friction
        grid = []
        for mass in mass_values:
            for friction in friction_values:
                grid.append({
                    'mass': mass,
                    'friction': friction,
                    'stiffness': 0.6,  # Default values
                    'damping': 0.2     # Default values
                })
    
    return grid


def setup_environment(cfg):
    """Set up the environment and agent for evaluation."""
    # Prepare configuration
    cfg.mode = 'eval'
    
    # Set up test-time adaptation if requested
    if cfg.adapt == 1:
        cfg.eval.use_test_time_adaptation = True
        cfg.eval.tta_num_steps = 1
    
    # Create environment
    env_class = get_env_class(cfg.env.env_class)
    env = env_class(OmegaConf.create(cfg.env.args))
    
    # Create agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    
    # Load checkpoint
    model_path = cfg.model
    if os.path.isdir(model_path):
        # Find best.pth in the directory
        checkpoint_path = os.path.join(model_path, 'best.pth')
        if not os.path.exists(checkpoint_path):
            # Find the latest checkpoint
            checkpoint_files = [f for f in os.listdir(model_path) if f.startswith('ckpt') and f.endswith('.pth')]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {model_path}")
            checkpoint_files.sort()
            checkpoint_path = os.path.join(model_path, checkpoint_files[-1])
    else:
        checkpoint_path = model_path
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent.load_snapshot(checkpoint_path)
    agent.train(False)  # Set to evaluation mode
    
    return env, agent, cfg


def organize_obs(render, state):
    """Organize observations from environment."""
    if isinstance(render, list):
        obs = {
            'pc': [r.get('pc') for r in render],
            'state': np.array(state),
        }
        for k in ['eef_pos', 'eef_rot']:
            if k in render[0]:
                obs[k] = [r.get(k) for r in render]
        return obs
    elif isinstance(render, dict):
        obs = organize_obs([render], [state])
        return {k: v[0] for k, v in obs.items()}


def run_episode(env, agent, params, use_adaptation):
    """Run a single episode with the given parameters and record metrics."""
    # Apply physics parameters to the environment
    env.apply_physics_params(params)
    
    # Reset environment
    state = env.reset()
    render = env.render()
    obs = organize_obs(render, state)
    
    # Apply test-time adaptation if requested
    if use_adaptation and hasattr(agent, 'actor') and hasattr(agent.actor, 'physics_enc'):
        agent.actor = apply_test_time_adaptation(agent.actor, obs, num_steps=1)
    
    # Record metrics
    steps = 0
    done = False
    success = False
    max_steps = 100  # Default maximum steps
    start_time = time.time()
    
    # Get true physics parameters and predicted physics (if available)
    true_physics = params.copy()
    phys_est = None
    phys_est_error = None
    
    if hasattr(agent, 'actor') and hasattr(agent.actor, 'physics_enc'):
        # Extract physics prediction from encoder
        pc = torch.from_numpy(obs['pc']).float().to(agent.device)
        if len(pc.shape) == 2:
            pc = pc.unsqueeze(0)  # Add batch dimension if needed
        
        with torch.no_grad():
            phys_est = agent.actor.physics_enc(pc).cpu().numpy()[0]
            
            # Normalize true physics to the same range as predicted [0,1]
            # These ranges should match those in the dataset
            mass_min, mass_max = 0.3, 2.0
            friction_min, friction_max = 0.1, 3.0
            stiffness_min, stiffness_max = 100.0, 300.0
            damping_min, damping_max = 0.5, 2.0
            
            norm_true = np.array([
                (params['mass'] - mass_min) / (mass_max - mass_min),
                (params['friction'] - friction_min) / (friction_max - friction_min),
                (params['stiffness'] - stiffness_min) / (stiffness_max - stiffness_min),
                (params['damping'] - damping_min) / (damping_max - damping_min)
            ])
            
            # Calculate mean squared error
            phys_est_error = np.mean((norm_true - phys_est) ** 2)
    
    # Run episode
    while not done and steps < max_steps:
        # Get agent action
        action = agent.act(obs)
        
        # Step environment
        state, reward, done, info = env.step(action)
        render = env.render()
        obs = organize_obs(render, state)
        
        # Check success
        if hasattr(env, 'is_success') and env.is_success():
            success = True
            done = True
        
        steps += 1
    
    # Record completion time
    completion_time = time.time() - start_time
    
    # Prepare result
    result = {
        'task': agent.cfg.env.args.task_name,
        'variant': agent.cfg.get('variant', 'unknown'),
        'seed': agent.cfg.seed,
        'mass': params['mass'],
        'friction': params['friction'],
        'stiffness': params['stiffness'],
        'damping': params['damping'],
        'success': int(success),  # Convert boolean to int for CSV
        'completion_time': completion_time,
        'phys_est_error': phys_est_error if phys_est_error is not None else 'N/A'
    }
    
    return result


def save_results(results, save_path):
    """Save evaluation results to a CSV file."""
    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['task', 'variant', 'seed', 'mass', 'friction', 'stiffness', 'damping', 'success', 'completion_time', 'phys_est_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    main() 