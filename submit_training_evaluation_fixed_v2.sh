#!/bin/bash
#SBATCH --job-name=equibot_master
#SBATCH --output=/fs/nexus-projects/Sketch_REBEL/equibot/logs/equibot_master_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_REBEL/equibot/logs/equibot_master_%j.err
#SBATCH --time=1:00:00  # Shorter time for master job
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# This script submits individual training jobs for each variant/task/seed combination

# Configuration
TASKS=("fold" "cover" "close")
VARIANTS=("A" "B" "C" "D" "E")
SEEDS=(0 1 2)
BASE_DIR="/fs/nexus-projects/Sketch_REBEL/equibot"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints/evaluation"
LOG_DIR="${BASE_DIR}/logs/evaluation"

# Create output directories
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/jobs  # Directory for individual job logs

echo "=============================================="
echo "    PI-EquiBot Evaluation Submission Script   "
echo "=============================================="
echo "Tasks: ${TASKS[@]}"
echo "Variants: ${VARIANTS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "=============================================="

# Create job tracking file to monitor progress
JOB_TRACKING_FILE="${LOG_DIR}/submitted_jobs.txt"
echo "JobID,Variant,Task,Seed,Status" > ${JOB_TRACKING_FILE}

# Create a template for the training job script
TRAINING_JOB_TEMPLATE="${LOG_DIR}/job_template.sh"
cat > ${TRAINING_JOB_TEMPLATE} << 'EOF'
#!/bin/bash
#SBATCH --job-name=equibot_train
#SBATCH --output=%LOG_DIR%/jobs/%VARIANT%_%TASK%_seed%SEED%_%j.log
#SBATCH --error=%LOG_DIR%/jobs/%VARIANT%_%TASK%_seed%SEED%_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Set working directory
cd /fs/cml-scratch/amishab/equibot

# Load conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Get variant name
VARIANT="%VARIANT%"
TASK="%TASK%"
SEED="%SEED%"

# Set variant name based on letter
if [ "${VARIANT}" == "A" ]; then
    VARIANT_NAME="vanilla_equibot"
elif [ "${VARIANT}" == "B" ]; then
    VARIANT_NAME="pi_equibot_base"
elif [ "${VARIANT}" == "C" ]; then
    VARIANT_NAME="pi_equibot_mixup"
elif [ "${VARIANT}" == "D" ]; then
    VARIANT_NAME="pi_equibot_mixup_drop"
elif [ "${VARIANT}" == "E" ]; then
    VARIANT_NAME="pi_equibot_full"
fi

# Set output and checkpoint directories
VARIANT_PREFIX="variant_${VARIANT}"
OUTPUT_DIR="%CHECKPOINT_DIR%/${TASK}/${VARIANT_NAME}_seed${SEED}"
mkdir -p ${OUTPUT_DIR}

echo "Starting training for Variant ${VARIANT} (${VARIANT_NAME}) on task ${TASK} with seed ${SEED}"

# Run training with the appropriate config and parameters using Hydra + syntax
# Added fixes for missing interpolation and parameters
python -m equibot.policies.train \
    --config-name evaluation/${VARIANT_PREFIX} \
    +mode=train \
    +prefix=${VARIANT_NAME}_${TASK}_seed${SEED} \
    +data.dataset.path=%BASE_DIR%/data/${TASK}_phy/pcs \
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
    +use_wandb=true \
    +wandb.project=equibot_evaluation \
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
    +model.use_torch_compile=false \
    +model.noise_scheduler._target_=diffusers.schedulers.scheduling_ddpm.DDPMScheduler \
    +model.noise_scheduler.num_train_timesteps=100 \
    +model.noise_scheduler.beta_schedule=squaredcos_cap_v2 \
    +model.noise_scheduler.clip_sample=true \
    +model.noise_scheduler.prediction_type=epsilon \
    +model.use_physics_embed=true \
    +model.physics_embed_dim=4 \
    +seed=${SEED} \
    +training.num_epochs=500 \
    +training.batch_size=32 \
    +training.lr=3e-5 \
    +training.weight_decay=1e-6 \
    +training.save_interval=50 \
    +training.vis_interval=100 \
    +training.eval_interval=50 \
    +training.num_eval_episodes=5

# Create a symbolic link to the best checkpoint
LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/ckpt*.pth 2>/dev/null | head -n 1)
if [ -n "$LATEST_CKPT" ]; then
    ln -sf $LATEST_CKPT ${OUTPUT_DIR}/best.pth
    echo "Created symbolic link to best checkpoint: $LATEST_CKPT -> ${OUTPUT_DIR}/best.pth"
else
    echo "Warning: No checkpoints found to link as best.pth"
fi

echo "Training completed for Variant ${VARIANT} on task ${TASK} with seed ${SEED}"
EOF

# Submit a job for each combination
for variant in "${VARIANTS[@]}"; do
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Submitting job for Variant ${variant}, task ${task}, seed ${seed}"
            
            # Create a specific job script for this combination
            JOB_SCRIPT="${LOG_DIR}/job_${variant}_${task}_seed${seed}.sh"
            cp ${TRAINING_JOB_TEMPLATE} ${JOB_SCRIPT}
            
            # Replace placeholders with actual values
            sed -i "s|%VARIANT%|${variant}|g" ${JOB_SCRIPT}
            sed -i "s|%TASK%|${task}|g" ${JOB_SCRIPT}
            sed -i "s|%SEED%|${seed}|g" ${JOB_SCRIPT}
            sed -i "s|%LOG_DIR%|${LOG_DIR}|g" ${JOB_SCRIPT}
            sed -i "s|%CHECKPOINT_DIR%|${CHECKPOINT_DIR}|g" ${JOB_SCRIPT}
            sed -i "s|%BASE_DIR%|${BASE_DIR}|g" ${JOB_SCRIPT}
            
            # Make the job script executable
            chmod +x ${JOB_SCRIPT}
            
            # Submit the job
            JOB_ID=$(sbatch ${JOB_SCRIPT} | awk '{print $4}')
            echo "${JOB_ID},${variant},${task},${seed},Submitted" >> ${JOB_TRACKING_FILE}
            
            echo "Job submitted with ID: ${JOB_ID}"
            
            # Add a small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

echo "All training jobs have been submitted."
echo "Monitor progress with: cat ${JOB_TRACKING_FILE}"
echo "Or check individual job status with: squeue -u $USER"
echo ""
echo "Note: The original code had an issue with DDPMScheduler's .to(device) method."
echo "This was fixed by removing the .to(device) call in equibot_policy.py."
echo "Additionally, we replaced hydra.utils.instantiate with direct instantiation of DDPMScheduler."
echo "The script now also includes the missing training.weight_decay parameter (default: 1e-6)."
echo "We also fixed train.py to handle the case when training.ckpt parameter is missing."
echo "Another fix in train.py: changed cfg.train to cfg.training for curriculum-related checks."
echo "Fixed shape mismatch in Normalizer class to handle tensors with different dimensions."
echo "If you see errors related to DDPMScheduler or missing parameters, please check if these fixes were applied."
echo "Also ensure LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 is set for all jobs." 