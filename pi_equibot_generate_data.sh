#!/bin/bash
#SBATCH --job-name=pi_data_gen
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_data_gen_%j.log
#SBATCH --error=/fs/nexus-projects/Sketch_VLM_RL/equibit/pi_data_gen_%j.err
#SBATCH --time=8:00:00
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

# Generate data with varying physics parameters
echo "Generating physics-informed data..."

# Number of episodes to generate
NUM_EPISODES=100

# Generate data with varying physics parameters
for i in $(seq 1 $NUM_EPISODES); do
    # Randomize physics parameters
    MASS=$(python -c "import numpy as np; print(np.random.uniform(0.1, 2.0))")
    FRICTION=$(python -c "import numpy as np; print(np.random.uniform(0.5, 5.0))")
    STIFFNESS=$(python -c "import numpy as np; print(np.random.uniform(100.0, 300.0))")
    
    echo "Episode $i: Mass=$MASS, Friction=$FRICTION, Stiffness=$STIFFNESS"
    
    # Run data generation
    python -m equibot.envs.sim_mobile.generate_demos \
        --task_name fold \
        --cam_num_views 5 \
        --seed $((1000 + $i)) \
        --data_out_dir $DATA_DIR \
        --speed_multiplier 2.0 \
        --deform_object_mass $MASS \
        --deform_friction_coeff $FRICTION \
        --deform_elastic_stiffness $STIFFNESS
    
    # Check if files were generated in the right location
    LATEST_FILES=$(find $DATA_DIR -type f -name "*.npz" -not -path "$PCS_DIR/*" | wc -l)
    
    if [ $LATEST_FILES -gt 0 ]; then
        echo "Moving $LATEST_FILES files to proper subdirectories..."
        # Move PC files to PCS_DIR
        find $DATA_DIR -maxdepth 1 -type f -name "*.npz" -exec mv {} $PCS_DIR/ \;
        # Move image files to IMAGES_DIR if they exist
        find $DATA_DIR -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) -exec mv {} $IMAGES_DIR/ \;
    fi
done

echo "Data generation complete at $(date)"
echo "Total point cloud files: $(find $PCS_DIR -type f -name "*.npz" | wc -l)"
echo "Total image files: $(find $IMAGES_DIR -type f | wc -l)"

# Verify data was generated successfully
if [ $(find $PCS_DIR -type f -name "*.npz" | wc -l) -eq 0 ]; then
    echo "ERROR: No data files were generated. Check the logs for errors."
    exit 1
else
    echo "SUCCESS: Data generation completed successfully."
fi 