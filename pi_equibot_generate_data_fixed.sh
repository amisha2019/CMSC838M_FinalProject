#!/bin/bash
#SBATCH --job-name=pi_data_gen
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_data_gen_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_data_gen_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

# Load conda
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Set working directory
cd /fs/cml-scratch/amishab/equibot

# Create data output directory structure
DATA_DIR=/fs/cml-scratch/amishab/data/fold_physics
PCS_DIR=$DATA_DIR/pcs
IMAGES_DIR=$DATA_DIR/images
mkdir -p $DATA_DIR
mkdir -p $PCS_DIR
mkdir -p $IMAGES_DIR

# Print debug info
echo "Starting physics-informed data generation at $(date)"
echo "Working directory: $(pwd)"
echo "Output directory: $DATA_DIR"
echo "PCS directory: $PCS_DIR"
echo "Images directory: $IMAGES_DIR"

# Number of episodes to generate in total
TARGET_EPISODES=100

# Count existing episodes
EXISTING_FILES=$(find $PCS_DIR -type f -name "*.npz" | wc -l)
echo "Found $EXISTING_FILES existing data files"

# Number of episodes to generate and maximum attempts
NUM_TO_GENERATE=$((TARGET_EPISODES - EXISTING_FILES))
MAX_ATTEMPTS=300
ATTEMPTS=0
GENERATED=0

if [ $NUM_TO_GENERATE -le 0 ]; then
    echo "Already have enough episodes ($EXISTING_FILES >= $TARGET_EPISODES). Exiting."
    exit 0
fi

echo "Need to generate $NUM_TO_GENERATE more episodes"

# Success counter
SUCCESS_COUNT=0

# Generate data with varying physics parameters - safer ranges
while [ $SUCCESS_COUNT -lt $NUM_TO_GENERATE ] && [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    ATTEMPTS=$((ATTEMPTS + 1))
    
    # Randomize physics parameters with safer ranges
    MASS=$(python -c "import numpy as np; print(np.random.uniform(0.3, 1.5))")
    FRICTION=$(python -c "import numpy as np; print(np.random.uniform(0.8, 3.0))")
    STIFFNESS=$(python -c "import numpy as np; print(np.random.uniform(120.0, 250.0))")
    
    echo "Attempt $ATTEMPTS (Successful: $SUCCESS_COUNT/$NUM_TO_GENERATE): Mass=$MASS, Friction=$FRICTION, Stiffness=$STIFFNESS"
    
    # Get file count before generating
    BEFORE_COUNT=$(find $PCS_DIR -type f -name "*.npz" | wc -l)
    
    # Run data generation with more stable parameters and explicitly set num_demos
    python -m equibot.envs.sim_mobile.generate_demos \
        --task_name fold \
        --cam_num_views 5 \
        --seed $((1000 + $ATTEMPTS)) \
        --data_out_dir $DATA_DIR \
        --speed_multiplier 2.0 \
        --deform_object_mass $MASS \
        --deform_friction_coeff $FRICTION \
        --deform_elastic_stiffness $STIFFNESS \
        --num_demos 1 \
        --data_rew_threshold 0.7 \
        --randomize_rotation \
        --randomize_scale \
        --scale_low 0.8 \
        --scale_high 1.2 \
        --scale_aspect_limit 1.3
    
    # Enhanced debugging
    echo "=== DEBUG INFO AFTER PYTHON EXECUTION ==="
    echo "Direct contents of DATA_DIR:"
    ls -la $DATA_DIR
    echo "Content of PCS_DIR:"
    ls -la $PCS_DIR
    echo "Content of IMAGES_DIR:"
    ls -la $IMAGES_DIR
    
    # Check if files were generated in the right location
    echo "Checking for generated files..."
    find $DATA_DIR -type f -name "*.npz" -not -path "$PCS_DIR/*" | sort
    LATEST_FILES=$(find $DATA_DIR -type f -name "*.npz" -not -path "$PCS_DIR/*" | wc -l)
    echo "Found $LATEST_FILES .npz files at the root level"
    
    if [ $LATEST_FILES -gt 0 ]; then
        echo "Moving $LATEST_FILES files to proper subdirectories..."
        # Move PC files to PCS_DIR
        find $DATA_DIR -maxdepth 1 -type f -name "*.npz" -exec mv {} $PCS_DIR/ \;
        # Move image files to IMAGES_DIR if they exist
        find $DATA_DIR -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) -exec mv {} $IMAGES_DIR/ \;
    fi
    
    # Get file count after generating
    AFTER_COUNT=$(find $PCS_DIR -type f -name "*.npz" | wc -l)
    
    # If new files were added, count as success
    if [ $AFTER_COUNT -gt $BEFORE_COUNT ]; then
        NEW_FILES=$((AFTER_COUNT - BEFORE_COUNT))
        SUCCESS_COUNT=$((SUCCESS_COUNT + NEW_FILES))
        echo "Success! Generated $NEW_FILES new file(s). Total: $SUCCESS_COUNT/$NUM_TO_GENERATE"
    else
        echo "No new files generated in this attempt"
    fi
    
    # Quick status update every 10 attempts
    if [ $((ATTEMPTS % 10)) -eq 0 ]; then
        echo "Status after $ATTEMPTS attempts: $SUCCESS_COUNT/$NUM_TO_GENERATE episodes generated"
    fi
done

echo "Data generation complete at $(date)"
echo "Total attempts: $ATTEMPTS"
echo "Successful generations: $SUCCESS_COUNT"
echo "Total point cloud files: $(find $PCS_DIR -type f -name "*.npz" | wc -l)"
echo "Total image files: $(find $IMAGES_DIR -type f | wc -l)"

# Verify that we generated a sufficient number of episodes
FINAL_COUNT=$(find $PCS_DIR -type f -name "*.npz" | wc -l)
if [ $FINAL_COUNT -lt $((TARGET_EPISODES * 3/4)) ]; then
    echo "WARNING: Only generated $FINAL_COUNT/$TARGET_EPISODES episodes (less than 75% of target)"
    # Don't exit with error to allow training to proceed with available data
fi

echo "SUCCESS: Data generation completed with $FINAL_COUNT episodes." 