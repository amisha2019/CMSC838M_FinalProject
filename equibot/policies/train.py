import os
import sys
import hydra
import torch
import wandb
import numpy as np
import logging
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from equibot.policies.utils.media import save_video
from equibot.policies.utils.misc import get_env_class, get_dataset, get_agent
from equibot.policies.vec_eval import run_eval
from equibot.envs.subproc_vec_env import SubprocVecEnv

logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="fold_synthetic", version_base=None)
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # where Hydra is writing out to
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.run.dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Outputs will go to: {out_dir}")

    # WandB
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=out_dir
        )

    # DataLoader
    train_ds = get_dataset(cfg, "train")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataset.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_ds) // cfg.training.batch_size
    )
    logger.info(f"Training samples: {len(train_ds)}")

    # Vectorized eval env if desired
    env_class = get_env_class(cfg.env.env_class)
    env_args = dict(OmegaConf.to_container(cfg.env.args, resolve=True))
    def make_env(i):
        args = env_args.copy()
        args["seed"] = cfg.seed * 100 + i
        return env_class(OmegaConf.create(args))

    if cfg.training.eval_interval <= cfg.training.num_epochs:
        env = SubprocVecEnv([lambda i=i: make_env(i) for i in range(cfg.training.num_eval_episodes)])
        logger.info("Eval env initialized")
    else:
        env = None

    # Agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    start_epoch = 0
    if cfg.training.ckpt:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch = int(os.path.basename(cfg.training.ckpt)[4:9])
        logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, cfg.training.num_epochs):
        for batch_ix, batch in enumerate(train_loader):
            metrics = agent.update(batch, vis=(epoch % cfg.training.vis_interval == 0 and batch_ix == 0))
            if cfg.use_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                wandb.log({"epoch": epoch}, step=global_step)
            global_step += 1

        # periodic eval
        if env and epoch > 0 and (epoch % cfg.training.eval_interval == 0 or epoch == cfg.training.num_epochs - 1):
            eval_metrics = run_eval(
                env, agent,
                vis=True,
                num_episodes=cfg.training.num_eval_episodes,
                reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
                use_wandb=cfg.use_wandb,
            )
            # log eval scalars
            if cfg.use_wandb:
                for k, v in eval_metrics.items():
                    if k not in ["vis_rollout", "rew_values", "vis_pc"]:
                        wandb.log({f"eval/{k}": v}, step=global_step)

            # save one video per epoch if requested
            if "vis_rollout" in eval_metrics:
                for i, vid in enumerate(eval_metrics["vis_rollout"]):
                    path = os.path.join(out_dir, f"eval_ep{epoch:04d}_{i}.mp4")
                    save_video(vid, path)
                    logger.info(f"Saved eval video: {path}")

        # periodic checkpoint
        if epoch % cfg.training.save_interval == 0 or epoch == cfg.training.num_epochs - 1:
            ckpt = os.path.join(out_dir, f"ckpt{epoch:05d}.pth")
            agent.save_snapshot(ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")

            # clean up old checkpoints
            all_ckpts = sorted(glob(os.path.join(out_dir, "ckpt*.pth")))
            for old in all_ckpts[:-cfg.training.save_interval]:
                os.remove(old)

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
