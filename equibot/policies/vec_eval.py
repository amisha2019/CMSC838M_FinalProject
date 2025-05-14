import os
import sys
import torch
import hydra
import omegaconf
import wandb
import numpy as np
import getpass as gt
from tqdm import tqdm
import logging

from equibot.policies.eval import organize_obs
from equibot.policies.utils.media import combine_videos, save_video

logger = logging.getLogger(__name__)

def run_eval(
    env,
    agent,
    vis=True,
    num_episodes=10,
    log_dir=None,
    reduce_horizon_dim=False,
    verbose=False,
    use_wandb=True,
    ckpt_name=None,
):
    """Run evaluation episodes."""
    metrics = {}
    all_rews = []
    all_rollouts = []
    
    for ep_ix in tqdm(range(num_episodes), desc="Eval episodes"):
        obs = env.reset()
        done = False
        ep_rew = 0
        rollout = []
        
        while not done:
            with torch.no_grad():
                action = agent.get_action(obs)
            obs, rew, done, info = env.step(action)
            ep_rew += rew
            if vis:
                rollout.append(obs)
        
        all_rews.append(ep_rew)
        if vis:
            all_rollouts.append(np.stack(rollout))
        logger.info(f"Episode {ep_ix} completed with reward {ep_rew}")
    
    metrics["rew_mean"] = np.mean(all_rews)
    metrics["rew_std"] = np.std(all_rews)
    metrics["rew_values"] = all_rews
    
    if vis:
        metrics["vis_rollout"] = all_rollouts
    
    logger.info(f"Evaluation completed. Mean reward: {metrics['rew_mean']:.2f} ± {metrics['rew_std']:.2f}")
    return metrics