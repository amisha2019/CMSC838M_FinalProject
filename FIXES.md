# PI-EquiBot Evaluation Framework Fixes

This document describes the fixes made to the PI-EquiBot evaluation framework scripts to address various configuration and parameter issues.

## Key Issues Fixed

1. **Environment Configuration Parameters**
   - Added missing environment parameters:
     - `num_eef`: Number of end effectors
     - `dof`: Degrees of freedom
     - `randomize_scale`: Whether to randomize scale
     - `randomize_rotation`: Whether to randomize rotation
     - `uniform_scaling`: Whether to use uniform scaling
     - `ac_noise`: Action noise
     - `vis`: Whether to visualize
     - `freq`: Frequency
     - `scale_low`: Minimum scale
     - `scale_high`: Maximum scale
     - `scale_aspect_limit`: Maximum aspect ratio

2. **Hydra Configuration Fixes**
   - Added `version_base=None` to the `@hydra.main` decorator in eval_grid.py
   - Used proper `+parameter=value` syntax for fields not in the base config
   - Fixed path references in variant YAML files from `-base` to `-../base`

3. **Dataset Parameters**
   - Added missing dataset parameters:
     - `data.dataset.dof`
     - `data.dataset.num_eef`
     - `data.dataset.eef_dim`
     - `data.dataset.reduce_horizon_dim`
     - `data.dataset.min_demo_length`

4. **Model Parameters**
   - Added model parameters:
     - `model.hidden_dim`
     - `model.encoder.c_dim`
     - `model.use_torch_compile`
     - `model.use_physics_embed`
     - `model.physics_embed_dim`
   - Added encoder parameters for SIM3Vec4Latent:
     - `model.encoder.backbone_type`
     - `model.encoder.backbone_args.num_layers`
     - `model.encoder.backbone_args.knn`
   - Added noise scheduler parameters for diffusion model:
     - `model.noise_scheduler._target_`
     - `model.noise_scheduler.num_train_timesteps`
     - `model.noise_scheduler.beta_schedule`
     - `model.noise_scheduler.clip_sample`
     - `model.noise_scheduler.prediction_type`
   - Fixed encoder initialization in equibot_policy.py
   - Fixed VecConditionalUnet1D initialization in equibot_policy.py

5. **Environment Class Names**
   - Fixed environment class names to use correct module imports:
     - FoldingEnv in folding_env.py
     - CoveringEnv in covering_env.py
     - ClosingEnv in closing_env.py

## Fixed Scripts

1. `submit_training_evaluation_fixed_v2.sh`
   - Fixed command line arguments to use proper Hydra syntax with + prefix
   - Added all missing environment and model parameters
   - Ensured proper data paths and checkpoint directories

2. `submit_evaluation_grid_fixed_v2.sh`
   - Fixed environment and model parameters
   - Added proper variant F support (with test-time adaptation)
   - Fixed result collection and CSV generation

3. `equibot/policies/eval_grid.py`
   - Added Hydra configuration with version_base=None 
   - Updated parameter handling to work with the command line arguments

4. `equibot/policies/configs/eval_grid_defaults.yaml`
   - Created default configuration file with all required parameters

## Test Scripts

The following test scripts were created to verify the fixes:

1. `test_env_creation.py` - Tests the environment creation with correct parameters
2. `test_hydra_config.py` - Tests the Hydra configuration with version_base=None
3. `test_equibot_fix.sh` - Master script to run all tests
4. `check_submission.py` - Validates that the submission scripts contain all required parameters 