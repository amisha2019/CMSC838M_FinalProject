import os
import sys
import copy
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import wandb
import omegaconf
import numpy as np
import getpass as gt
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf
import logging
import shutil

from equibot.policies.utils.media import save_video
from equibot.policies.utils.misc import get_env_class, get_dataset, get_agent
from equibot.policies.vec_eval import run_eval
from equibot.envs.subproc_vec_env import SubprocVecEnv


@hydra.main(config_path="configs", config_name="fold_synthetic", version_base=None)
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)
    
    # Fix Hydra output directory - use configurable path instead of hardcoded one
    if hasattr(cfg, 'log_dir'):
        output_dir = os.path.abspath(cfg.log_dir)
    else:
        output_dir = "/fs/nexus-projects/Sketch_VLM_RL/equibit"  # fallback to original path
    os.makedirs(output_dir, exist_ok=True)
    
    # Get Hydra's current output directory
    hydra_output_dir = os.getcwd()
    logger = None
    
    # Setup logging first, so we can log the directory change
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    logger.info(f"Original Hydra directory: {hydra_output_dir}")
    logger.info(f"Target output directory: {output_dir}")
    
    # Change to our output directory
    os.chdir(output_dir)
    logger.info(f"Changed working directory to: {os.getcwd()}")
    
    # Copy Hydra configs to our output directory if they exist
    hydra_config_dir = os.path.join(hydra_output_dir, ".hydra")
    if os.path.exists(hydra_config_dir):
        os.makedirs(os.path.join(output_dir, ".hydra"), exist_ok=True)
        for hydra_file in os.listdir(hydra_config_dir):
            src = os.path.join(hydra_config_dir, hydra_file)
            dst = os.path.join(output_dir, ".hydra", hydra_file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                logger.info(f"Copied Hydra config: {src} to {dst}")
    else:
        logger.warning(f"Hydra config directory not found at {hydra_config_dir}")
    
    # initialize parameters
    batch_size = cfg.training.batch_size

    # setup wandb
    if cfg.use_wandb:
        try:
            wandb_config = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=False
            )
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                tags=["train"],
                name=cfg.prefix,
                settings=wandb.Settings(code_dir="."),
                config=wandb_config,
                dir=output_dir  # Set wandb directory to our new location
            )
            logger.info("Successfully initialized wandb")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {str(e)}")
            cfg.use_wandb = False

    # init dataloader
    train_dataset = get_dataset(cfg, "train")
    num_workers = cfg.data.dataset.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )
    logger.info(f"Initialized dataloader with {len(train_dataset)} samples")

    # init env
    env_fns = []
    env_class = get_env_class(cfg.env.env_class)
    env_args = dict(OmegaConf.to_container(cfg.env.args, resolve=True))

    def create_env(env_args_dict, i):
        # Make a copy of the environment arguments to avoid modifying the original
        env_args_copy = env_args_dict.copy()
        # Set the seed as an item in the dictionary
        env_args_copy['seed'] = cfg.seed * 100 + i
        # Convert back to OmegaConf object
        return env_class(OmegaConf.create(env_args_copy))

    if cfg.training.eval_interval <= cfg.training.num_epochs:
        try:
            env = SubprocVecEnv(
                [
                    lambda seed=i: create_env(env_args, seed)
                    for i in range(cfg.training.num_eval_episodes)
                ]
            )
            logger.info("Initialized evaluation environment")
        except Exception as e:
            logger.error(f"Failed to initialize evaluation environment: {str(e)}")
            env = None
    else:
        env = None
        logger.info("Skipping evaluation environment initialization")

    # init agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    if hasattr(cfg.training, 'ckpt') and cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
        logger.info(f"Loaded checkpoint from {cfg.training.ckpt}, starting from epoch {start_epoch_ix}")
    else:
        start_epoch_ix = 0
        logger.info("Starting training from scratch")

    # train loop
    global_step = 0
    for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
        batch_ix = 0
        
        # Add-on #4: Extreme-curriculum sampling
        if hasattr(cfg.training, 'use_curriculum') and cfg.training.use_curriculum and hasattr(cfg.training, 'curriculum_T'):
            # Linearly increase the physics parameter range over the first curriculum_T epochs
            curriculum_T = cfg.training.curriculum_T
            curr_scale = min(1.0, epoch_ix / curriculum_T)
            logger.info(f"Epoch {epoch_ix}: Setting curriculum scale to {curr_scale:.3f}")
            try:
                train_dataset.set_phys_range(scale=curr_scale)
            except Exception as e:
                logger.warning(f"Failed to set physics range: {str(e)}")
        
        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            train_metrics = agent.update(
                batch, vis=epoch_ix % cfg.training.vis_interval == 0 and batch_ix == 0
            )
            if cfg.use_wandb:
                try:
                    wandb.log(
                        {"train/" + k: v for k, v in train_metrics.items()},
                        step=global_step,
                    )
                    wandb.log({"epoch": epoch_ix}, step=global_step)
                except Exception as e:
                    logger.error(f"Failed to log to wandb: {str(e)}")
            logger.info(f"Epoch {epoch_ix}, Batch {batch_ix}, Metrics: {train_metrics}")
            del train_metrics
            global_step += 1
            batch_ix += 1
        if (
            (
                epoch_ix % cfg.training.eval_interval == 0
                or epoch_ix == cfg.training.num_epochs - 1
            )
            and epoch_ix > 0
            and env is not None
        ):
            logger.info(f"Running evaluation at epoch {epoch_ix}")
            try:
                eval_metrics = run_eval(
                    env,
                    agent,
                    vis=True,
                    num_episodes=cfg.training.num_eval_episodes,
                    reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
                    use_wandb=cfg.use_wandb,
                )
                
                if cfg.use_wandb:
                    if epoch_ix > cfg.training.eval_interval and "vis_pc" in eval_metrics:
                        # only save one pc per run to save space
                        del eval_metrics["vis_pc"]
                    wandb.log(
                        {
                            "eval/" + k: v
                            for k, v in eval_metrics.items()
                            if not k in ["vis_rollout", "rew_values"]
                        },
                        step=global_step,
                    )
                    
                if "vis_rollout" in eval_metrics:
                    logger.info(f"Saving evaluation videos for epoch {epoch_ix}")
                    for eval_idx, eval_video in enumerate(eval_metrics["vis_rollout"]):
                        video_path = os.path.join(
                            output_dir,
                            f"eval{epoch_ix:05d}_ep{eval_idx}_rew{eval_metrics['rew_values'][eval_idx]}.mp4",
                        )
                        try:
                            save_video(eval_video, video_path)
                            logger.info(f"Saved eval video to {video_path}")
                        except Exception as e:
                            logger.error(f"Failed to save video to {video_path}: {str(e)}")
                del eval_metrics
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                
        if (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(output_dir, f"ckpt{epoch_ix:05d}.pth")
            logger.info(f"Saving checkpoint to {save_path}")
            try:
                num_ckpt_to_keep = 10
                if len(list(glob(os.path.join(output_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
                    # remove old checkpoints
                    for fn in list(sorted(glob(os.path.join(output_dir, "ckpt*.pth"))))[
                        :-num_ckpt_to_keep
                    ]:
                        os.remove(fn)
                        logger.info(f"Removed old checkpoint: {fn}")
                agent.save_snapshot(save_path)
                logger.info(f"Successfully saved checkpoint to {save_path}")
                
                # Also copy to current working directory for Hydra compatibility
                cwd_save_path = os.path.join(os.getcwd(), f"ckpt{epoch_ix:05d}.pth")
                shutil.copy2(save_path, cwd_save_path)
                logger.info(f"Copied checkpoint to current working directory: {cwd_save_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint to {save_path}: {str(e)}")


if __name__ == "__main__":
    main()
