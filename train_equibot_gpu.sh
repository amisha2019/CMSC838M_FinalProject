#!/bin/bash
#SBATCH --job-name=equibot_train
#SBATCH --output=/fs/cml-scratch/amishab/equibot/logs/equibot_train_%j.log
#SBATCH --error=/fs/cml-scratch/amishab/equibot/logs/equibot_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Parameters (can be overridden via environment variables)
VARIANT=${VARIANT:-"A"}       # A=vanilla_equibot, B=pi_equibot_base, C=pi_equibot_mixup, D=pi_equibot_mixup_drop, E=pi_equibot_full
TASK=${TASK:-"fold"}          # fold, cover, close
SEED=${SEED:-0}               # 0, 1, 2, etc.
EPOCHS=${EPOCHS:-500}         # Number of training epochs
BATCH_SIZE=${BATCH_SIZE:-32}  # Batch size
USE_WANDB=${USE_WANDB:-true}  # Whether to use W&B logging

# Set working directory
cd /fs/cml-scratch/amishab/equibot

# Load conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Create output directories
mkdir -p logs
CHECKPOINT_DIR="/fs/cml-scratch/amishab/equibot/checkpoints/${TASK}"
mkdir -p ${CHECKPOINT_DIR}

# Set variant name based on letter
if [ "${VARIANT}" == "A" ]; then
    VARIANT_NAME="vanilla_equibot"
    VARIANT_PREFIX="variant_A"
elif [ "${VARIANT}" == "B" ]; then
    VARIANT_NAME="pi_equibot_base"
    VARIANT_PREFIX="variant_B"
elif [ "${VARIANT}" == "C" ]; then
    VARIANT_NAME="pi_equibot_mixup"
    VARIANT_PREFIX="variant_C"
elif [ "${VARIANT}" == "D" ]; then
    VARIANT_NAME="pi_equibot_mixup_drop"
    VARIANT_PREFIX="variant_D"
elif [ "${VARIANT}" == "E" ]; then
    VARIANT_NAME="pi_equibot_full"
    VARIANT_PREFIX="variant_E"
else
    echo "Invalid variant: ${VARIANT}. Using vanilla_equibot."
    VARIANT_NAME="vanilla_equibot"
    VARIANT_PREFIX="variant_A"
fi

# Job info
echo "================== EquiBot Training =================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "CPU cores: $SLURM_CPUS_ON_NODE"
echo "GPU information: $(nvidia-smi -L)"
echo "Model variant: ${VARIANT_NAME} (${VARIANT})"
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "W&B logging: ${USE_WANDB}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "===================================================="

# Set output directory for this specific run
OUTPUT_DIR="${CHECKPOINT_DIR}/${VARIANT_NAME}_seed${SEED}"
mkdir -p ${OUTPUT_DIR}

# Run training with the appropriate config and parameters
# Using all our fixes for the normalizer and encoder_handle
echo "Starting training at $(date)"
python -m equibot.policies.train \
    --config-name evaluation/${VARIANT_PREFIX} \
    +mode=train \
    +prefix=${VARIANT_NAME}_${TASK}_seed${SEED} \
    +data.dataset.path=/fs/nexus-projects/Sketch_REBEL/equibot/data/${TASK}_phy/pcs \
    +data.dataset.dof=4 \
    +data.dataset.num_eef=1 \
    +data.dataset.eef_dim=3 \
    +data.dataset.num_training_steps=500 \
    +data.dataset.num_points=1024 \
    +data.dataset.num_augment=0 \
    +data.dataset.same_aug_per_sample=true \
    +data.dataset.aug_keep_original=true \
    +data.dataset.aug_scale_low=0.5 \
    +data.dataset.aug_scale_high=1.5 \
    +data.dataset.aug_scale_aspect_limit=1.0 \
    +data.dataset.aug_scale_rot=-1 \
    +data.dataset.aug_scale_pos=0.1 \
    +data.dataset.aug_zero_z_offset=false \
    +data.dataset.aug_center=[0.0,0.0,0.0] \
    +data.dataset.shuffle_pc=true \
    +data.dataset.num_workers=8 \
    +data.dataset.reduce_horizon_dim=true \
    +data.dataset.min_demo_length=16 \
    +data.dataset.obs_horizon=2 \
    +data.dataset.pred_horizon=16 \
    +device=cuda \
    +use_wandb=${USE_WANDB} \
    +wandb.project=equibot_training \
    +wandb.entity=amishab \
    +agent.agent_name=equibot \
    +env.env_class=${TASK} \
    +env.args.task_name=${TASK} \
    +env.args.max_episode_length=100 \
    +env.args.num_eef=1 \
    +env.args.dof=4 \
    +env.args.seed=${SEED} \
    +env.args.randomize_scale=true \
    +env.args.randomize_rotation=true \
    +env.args.uniform_scaling=true \
    +env.args.ac_noise=0.0 \
    +env.args.vis=false \
    +env.args.freq=5 \
    +env.args.scale_low=0.8 \
    +env.args.scale_high=1.2 \
    +env.args.scale_aspect_limit=1.3 \
    +env.dof=4 \
    +env.num_eef=1 \
    +env.eef_dim=3 \
    +model.obs_horizon=2 \
    +model.ac_horizon=8 \
    +model.pred_horizon=16 \
    +model.obs_mode=pc \
    +model.ac_mode=diffusion \
    +model.hidden_dim=256 \
    +model.encoder.c_dim=256 \
    +model.encoder.backbone_type=vn_pointnet \
    +model.encoder.backbone_args.num_layers=4 \
    +model.encoder.backbone_args.knn=8 \
    +model.use_torch_compile=true \
    +model.noise_scheduler._target_=diffusers.schedulers.scheduling_ddpm.DDPMScheduler \
    +model.noise_scheduler.num_train_timesteps=100 \
    +model.noise_scheduler.beta_schedule=squaredcos_cap_v2 \
    +model.noise_scheduler.clip_sample=true \
    +model.noise_scheduler.prediction_type=epsilon \
    +model.use_physics_embed=true \
    +model.physics_embed_dim=4 \
    +seed=${SEED} \
    +training.num_epochs=${EPOCHS} \
    +training.batch_size=${BATCH_SIZE} \
    +training.lr=3e-5 \
    +training.weight_decay=1e-6 \
    +training.save_interval=50 \
    +training.vis_interval=100 \
    +training.eval_interval=50 \
    +training.num_eval_episodes=5

# Check training result
TRAIN_RESULT=$?
if [ $TRAIN_RESULT -eq 0 ]; then
    echo "✅ Training completed successfully at $(date)"
    
    # Create a symbolic link to the best checkpoint
    LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/ckpt*.pth 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        ln -sf $LATEST_CKPT ${OUTPUT_DIR}/best.pth
        echo "Created symbolic link to best checkpoint: $LATEST_CKPT -> ${OUTPUT_DIR}/best.pth"
    else
        echo "Warning: No checkpoints found to link as best.pth"
    fi
    
    exit 0
else
    echo "❌ Training failed with error code $TRAIN_RESULT at $(date)"
    exit $TRAIN_RESULT
fi 