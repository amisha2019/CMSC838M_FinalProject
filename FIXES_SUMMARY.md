# EquiBot Fixes Summary

This document outlines the fixes applied to the EquiBot codebase to address several critical issues that were causing training jobs to fail.

## Issues Fixed

### 1. Empty Dataset Path
**Problem:** The training job expected data files at a specific path that didn't exist, causing an empty dataset error.

**Fix:** 
- Modified the run_training.sh script to accept a configurable data path
- Added checks to verify that the data directory exists before starting training
- Created a proper directory structure for storing data

### 2. Hard-coded Output Directory
**Problem:** All jobs wrote to the same hard-coded output directory, causing overwrites.

**Fix:**
- Modified train.py to use a configurable log directory via the `log_dir` parameter
- Updated all scripts to use unique output directories for each job
- Ensured that job-specific directories are created before training starts

### 3. Missing Hydra Placeholders
**Problem:** The base.yaml config contained placeholders marked with `???` that needed to be overridden.

**Fix:**
- Ensured all required parameters are provided in the training scripts:
  - `env.args.num_points`
  - `env.args.max_episode_length`
  - `data.dataset.num_training_steps`

### 4. Non-unique Log Directories
**Problem:** Multiple training jobs could overwrite each other's logs and checkpoints.

**Fix:**
- Created a directory structure that ensures unique paths for each job: `${LOG_ROOT}/${TASK}/${VARIANT}_seed${SEED}`
- Added the `submit_jobs.sh` script that uses SLURM job arrays to manage multiple training runs

## Running Training Jobs

Two scripts have been created to simplify running training jobs:

1. **run_training.sh** - For running a single training job:
   ```bash
   bash run_training.sh --task close --variant baseline --seed 0 --data_root /path/to/data --log_root /path/to/logs
   ```

2. **submit_jobs.sh** - For submitting multiple jobs as a SLURM array:
   ```bash
   sbatch submit_jobs.sh
   ```
   This will submit 9 jobs with different task/variant combinations.

## Data Requirements

Before running training, make sure your data is organized in the following structure:
```
/path/to/data/
├── close_phy/
│   └── pcs/
│       └── *.npz files
├── fold_phy/
│   └── pcs/
│       └── *.npz files
└── pour_phy/
    └── pcs/
        └── *.npz files
```

## Environment Setup

Make sure to activate the correct environment before running training:
```bash
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

These commands are included in the run_training.sh script. 