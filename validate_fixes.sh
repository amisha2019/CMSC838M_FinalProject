#!/bin/bash
# Run all validation tests for the PI-EquiBot fixes

# Load conda environment
source /fs/cml-scratch/amishab/miniconda3/etc/profile.d/conda.sh
conda activate lfd

# Set library path
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

echo "===========================================" 
echo "   PI-EquiBot Fix Validation Script       "
echo "===========================================" 

echo -e "\n1. Running environment creation test..."
./test_env_creation.py
ENV_RESULT=$?

echo -e "\n2. Running Hydra configuration test..."
./test_hydra_config.py
HYDRA_RESULT=$?

echo -e "\n3. Checking submission scripts..."
./check_submission.py
SCRIPT_RESULT=$?

echo -e "\n===========================================" 
echo "           Validation Results               "
echo "===========================================" 

if [ $ENV_RESULT -eq 0 ]; then
    echo "✅ Environment Test: PASSED"
else
    echo "❌ Environment Test: FAILED"
fi

if [ $HYDRA_RESULT -eq 0 ]; then
    echo "✅ Hydra Configuration Test: PASSED"
else
    echo "❌ Hydra Configuration Test: FAILED"
fi

if [ $SCRIPT_RESULT -eq 0 ]; then
    echo "✅ Submission Scripts Check: PASSED"
else
    echo "❌ Submission Scripts Check: FAILED"
fi

# Check overall results
if [ $ENV_RESULT -eq 0 ] && [ $HYDRA_RESULT -eq 0 ] && [ $SCRIPT_RESULT -eq 0 ]; then
    echo -e "\n✅ All validation tests PASSED! The PI-EquiBot fixes are working properly."
    echo -e "   You can now use submit_training_evaluation_fixed_v2.sh and submit_evaluation_grid_fixed_v2.sh"
    exit 0
else
    echo -e "\n❌ Some validation tests FAILED! The PI-EquiBot fixes need further adjustment."
    exit 1
fi 