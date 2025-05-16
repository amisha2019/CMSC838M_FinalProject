#!/usr/bin/env python
# Check that the submission scripts contain all required parameters

import sys
import re


def check_submission_scripts():
    scripts = [
        'submit_training_evaluation_fixed_v2.sh',
        'submit_evaluation_grid_fixed_v2.sh'
    ]
    required_params = [
        '+env.args.num_eef=',
        '+env.args.dof=',
        '+env.args.randomize_scale=',
        '+env.args.randomize_rotation=',
        '+env.args.uniform_scaling=',
        '+env.args.ac_noise=',
        '+env.args.vis=',
        '+env.args.freq=',
        '+env.args.scale_low=',
        '+env.args.scale_high=',
        '+env.args.scale_aspect_limit=',
        '+data.dataset.dof=',
        '+data.dataset.num_eef=',
        '+data.dataset.eef_dim=',
        '+data.dataset.reduce_horizon_dim=',
        '+data.dataset.min_demo_length='
    ]
    
    all_checks_passed = True
    
    for script in scripts:
        print(f"Checking {script}...")
        try:
            with open(script, 'r') as f:
                content = f.read()
            
            # Check all required parameters
            missing_params = []
            for param in required_params:
                if param not in content:
                    missing_params.append(param)
            
            if missing_params:
                print(f"❌ Missing parameters in {script}:")
                for param in missing_params:
                    print(f"   - {param}")
                all_checks_passed = False
            else:
                print(f"✅ All required parameters found in {script}")
            
            # Check additional requirements
            if '+model.hidden_dim=' not in content:
                print(f"❌ Missing model.hidden_dim parameter in {script}")
                all_checks_passed = False
            
            if '+model.encoder.c_dim=' not in content:
                print(f"❌ Missing model.encoder.c_dim parameter in {script}")
                all_checks_passed = False
            
            if '+model.encoder.backbone_type=' not in content:
                print(f"❌ Missing model.encoder.backbone_type parameter in {script}")
                all_checks_passed = False
            
            if '+model.encoder.backbone_args.num_layers=' not in content:
                print(f"❌ Missing model.encoder.backbone_args.num_layers parameter in {script}")
                all_checks_passed = False
            
            if '+model.encoder.backbone_args.knn=' not in content:
                print(f"❌ Missing model.encoder.backbone_args.knn parameter in {script}")
                all_checks_passed = False
            
            if '+model.use_torch_compile=' not in content:
                print(f"❌ Missing model.use_torch_compile parameter in {script}")
                all_checks_passed = False
            
            if '+model.noise_scheduler._target_=' not in content:
                print(f"❌ Missing model.noise_scheduler._target_ parameter in {script}")
                all_checks_passed = False
            
            if '+model.noise_scheduler.num_train_timesteps=' not in content:
                print(f"❌ Missing model.noise_scheduler.num_train_timesteps parameter in {script}")
                all_checks_passed = False
            
            if '+model.noise_scheduler.beta_schedule=' not in content:
                print(f"❌ Missing model.noise_scheduler.beta_schedule parameter in {script}")
                all_checks_passed = False
            
            if '+model.noise_scheduler.clip_sample=' not in content:
                print(f"❌ Missing model.noise_scheduler.clip_sample parameter in {script}")
                all_checks_passed = False
            
            if '+model.noise_scheduler.prediction_type=' not in content:
                print(f"❌ Missing model.noise_scheduler.prediction_type parameter in {script}")
                all_checks_passed = False
            
            if '+model.use_physics_embed=' not in content:
                print(f"❌ Missing model.use_physics_embed parameter in {script}")
                all_checks_passed = False
            
            if '+model.physics_embed_dim=' not in content:
                print(f"❌ Missing model.physics_embed_dim parameter in {script}")
                all_checks_passed = False
            
            # Check for version_base=None in eval_grid.py
            if script == 'submit_evaluation_grid_fixed_v2.sh':
                with open('equibot/policies/eval_grid.py', 'r') as f:
                    eval_content = f.read()
                
                if 'version_base=None' not in eval_content:
                    print("❌ Missing version_base=None in eval_grid.py")
                    all_checks_passed = False
                else:
                    print("✅ Found version_base=None in eval_grid.py")
            
        except FileNotFoundError:
            print(f"❌ Script {script} not found!")
            all_checks_passed = False
    
    return all_checks_passed


if __name__ == "__main__":
    success = check_submission_scripts()
    if success:
        print("\n✅ All submission scripts are correctly configured!")
        sys.exit(0)
    else:
        print("\n❌ Some issues found in submission scripts.")
        sys.exit(1) 