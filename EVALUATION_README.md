# PI-EquiBot Evaluation Framework

This repository contains a comprehensive evaluation framework for testing and analyzing different variants of PI-EquiBot across a range of physical parameters. The framework includes scripts for training, evaluation, and analysis.

## Experimental Matrix

| ID    | Variant                            | Physics Loss | MixUp | Latent-Drop | Curriculum | Test-time Adapt | Purpose                   |
| ----- | ---------------------------------- | ------------ | ----- | ----------- | ---------- | --------------- | ------------------------- |
| **A** | Vanilla EquiBot                    | –            | –     | –           | –          | –               | Baseline                  |
| **B** | PI-EquiBot                         | ✓ (λ=0.1)    | –     | –           | –          | –               | Physics-aware only        |
| **C** | PI-EquiBot + MixUp                 | ✓            | ✓     | –           | –          | –               | Interpolation benefit     |
| **D** | PI-EquiBot + Drop                  | ✓            | ✓     | ✓(p=0.2)    | –          | –               | Robustness to noisy embed |
| **E** | PI-EquiBot + Curriculum            | ✓            | ✓     | ✓           | ✓          | –               | Wider dynamics coverage   |
| **F** | PI-EquiBot + 1-step Adapt (online) | ✓            | ✓     | ✓           | ✓          | ✓               | Best-effort final model   |

## Evaluation Grid

| Dimension                  | Values                              |
| -------------------------- | ----------------------------------- |
| **Mass**                   | 0.5 kg, 1 kg (train mean), **3 kg** |
| **Friction**               | 0.2, 0.5 (train mean), **1.2**      |
| **Stiffness** (cloth only) | 0.3, 0.6 (train mean), **1.0**      |
| **Damping** (cloth only)   | 0.05, 0.2 (train mean), **0.5**     |

*Bold = clear extrapolation*

For **Fold** & **Cover** tasks, we evaluate all 3×3×3×3 = 81 parameter combinations. For the **Close** task, we only vary mass and friction, resulting in 3×3 = 9 combinations.

## Directory Structure

```
├── equibot/
│   ├── policies/
│   │   ├── configs/
│   │   │   ├── evaluation/           # Configuration files for each variant
│   │   │   │   ├── variant_A.yaml
│   │   │   │   ├── variant_B.yaml
│   │   │   │   ├── variant_C.yaml
│   │   │   │   ├── variant_D.yaml
│   │   │   │   ├── variant_E.yaml
│   │   │   │   └── variant_F.yaml
│   │   ├── eval_grid.py              # Script for grid evaluation
│   │   └── ...
│
├── run_training_evaluation.sh        # Script to train all variants
├── run_evaluation_grid.sh            # Script to run evaluation for all variants
└── analysis_plots.py                 # Script to generate analysis plots
```

## Usage

### Step 1: Train all variants

```bash
./run_training_evaluation.sh
```

This script will:
- Train all variants (A-E) for all tasks (fold, cover, close)
- Use 3 random seeds per variant
- Save checkpoints to the specified output directory

### Step 2: Run evaluation

```bash
./run_evaluation_grid.sh
```

This script will:
- Evaluate all variants (A-F) for all tasks, including Test-time Adaptation for variant F
- Run 10 episodes per parameter configuration
- Save results to CSV files

### Step 3: Generate analysis plots

```bash
python analysis_plots.py --results_dir /path/to/results --output_dir /path/to/output --combined
```

This script will generate:
1. Success-rate heatmaps for each task and variant
2. Generalization curves showing success vs out-of-distribution distance
3. Physics-estimation scatter plots
4. Ablation summary tables

## Metrics

The evaluation records the following metrics:

| Metric                                  | Purpose                                     |
| --------------------------------------- | ------------------------------------------- |
| **Success Rate** (0/1)                  | Primary task outcome                        |
| **Completion Time** (steps to terminal) | Shows if heavier objects slow the policy    |
| **Physics-Est Error** = ‖ŷ–y‖₂          | Measures physics encoder quality            |

## Analysis

The framework generates the following analysis artifacts:

### 1. Success-rate Heatmaps

Visualize success rates across different mass and friction values for each variant and task.

### 2. Generalization Curves

Plot success rate vs. out-of-distribution distance to show how well each variant handles novel physical parameters.

### 3. Physics-Estimation Plots

Show the quality of physics parameter estimation across variants.

### 4. Ablation Summary Table

Summarize the performance of each variant across tasks to demonstrate the contribution of each enhancement.

## Expected Results

When running the full evaluation suite, you should observe a monotonic improvement across variants, with the best performance obtained by Variant F (PI-EquiBot + all enhancements + Test-time Adaptation). The results should demonstrate that:

1. Physics supervision improves OOD generalization
2. MixUp provides better interpolation between training examples
3. Latent-dropout enhances robustness to noisy physics estimates
4. Curriculum learning enables coverage of more extreme physical parameters
5. Test-time adaptation provides an additional boost without requiring retraining 