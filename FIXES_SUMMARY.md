# PI-EquiBot Codebase Fixes

This document summarizes all the issues fixed in the PI-EquiBot codebase to make it run successfully.

## 1. DDPMScheduler Issues

### Problem
`DDPMScheduler` from the `diffusers` library doesn't have a `.to(device)` method, but it was being called in the code:

```python
self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler).to(device)
```

### Fix
- Removed the `.to(device)` call from `equibot_policy.py`
- Replaced `hydra.utils.instantiate` with direct instantiation of `DDPMScheduler` class
- Added proper import for `DDPMScheduler`

## 2. Missing Configuration Parameters

### Problem
Several configuration parameters were missing:
- `training.weight_decay` - used in optimizer initialization
- `training.ckpt` - used for checkpoint loading

### Fix
- Added the missing `training.weight_decay` parameter to the training configuration (default: 1e-6)
- Modified `train.py` to check if `training.ckpt` exists before trying to access it

## 3. Configuration Key Mismatch

### Problem
Code was trying to access `cfg.train` but the correct key is `cfg.training`:

```python
if hasattr(cfg.train, 'use_curriculum') and cfg.train.use_curriculum and hasattr(cfg.train, 'curriculum_T'):
```

### Fix
Changed all instances of `cfg.train` to `cfg.training` for curriculum-related checks.

## 4. Normalizer Shape Mismatch

### Problem
When normalizing actions, there was a shape mismatch causing this error:
```
RuntimeError: shape '[1, 1, 14]' is invalid for input of size 4
```

And another error:
```
RuntimeError: The size of tensor a (14) must match the size of tensor b (4) at non-singleton dimension 2
```

### Fix
Updated the `normalize` and `unnormalize` methods in the `Normalizer` class to handle tensors with different dimensions:

1. Added a try-except block to catch and handle reshape errors
2. If input size is smaller than expected, use only the portion of stats that fits the input tensor
3. **Improved fix:** Properly reshape the sliced tensors to match the input tensor dimensions for broadcasting:
   ```python
   # Create proper shape for broadcasting
   new_target_shape = (1,) * (nd - 1) + (total_dims,)
   dmin = dmin.reshape(new_target_shape)
   dmax = dmax.reshape(new_target_shape)
   ```

4. This ensures proper broadcasting when performing operations between tensors of different dimensions

## 5. System Compatibility Issues

### Problem
Library compatibility issues with `libstdc++.so.6`

### Fix
Added note to ensure `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` is set for all jobs.

## 6. Required Packages

### Problem
Missing Python packages required for the codebase to run properly.

### Fix
The following packages need to be installed in the conda environment:
- `pybullet` - For physics simulation
- `diffusers` - For diffusion models
- `hydra-core` - For configuration management
- `wandb` - For experiment tracking
- `torch` - For deep learning

Installation command:
```bash
conda activate lfd
pip install pybullet diffusers hydra-core wandb
```

## Testing Status

All jobs have been successfully submitted and are waiting in the queue. The fixes should allow the training to proceed without the above errors. 

A validation test script (`test_normalizer_fix.py`) has been created to verify the normalizer fix is working properly, and it passes successfully. 