#!/usr/bin/env python
"""
Analysis and visualization script for PI-EquiBot evaluation results.

This script reads the CSV result files and generates:
1. Success-rate heatmaps
2. Generalization curves
3. Physics-estimation scatter plots
4. Ablation summary tables
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14
})


def parse_args():
    parser = argparse.ArgumentParser(description='Analysis of PI-EquiBot evaluation results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing evaluation result CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated plots')
    parser.add_argument('--combined', action='store_true',
                       help='Use combined CSV files (task_combined.csv) for analysis')
    return parser.parse_args()


def load_results(results_dir, combined=False):
    """Load evaluation results from CSV files."""
    tasks = ['fold', 'cover', 'close']
    
    if combined:
        # Load combined files
        dfs = {}
        for task in tasks:
            csv_file = os.path.join(results_dir, f"{task}_combined.csv")
            if os.path.exists(csv_file):
                dfs[task] = pd.read_csv(csv_file)
                # Convert variant names
                variant_mapping = {
                    'A': 'Vanilla EquiBot',
                    'B': 'PI-EquiBot',
                    'C': 'PI-EquiBot + MixUp',
                    'D': 'PI-EquiBot + Drop',
                    'E': 'PI-EquiBot + Curriculum',
                    'F': 'PI-EquiBot + 1-step Adapt'
                }
                dfs[task]['variant_name'] = dfs[task]['variant'].map(variant_mapping)
        return dfs
    
    # Load individual files and combine them
    dfs = {}
    for task in tasks:
        df_list = []
        
        for variant in ['A', 'B', 'C', 'D', 'E', 'F']:
            for seed in [0, 1, 2]:
                # Determine variant name
                if variant == 'A':
                    variant_name = 'vanilla_equibot'
                elif variant == 'B':
                    variant_name = 'pi_equibot_base'
                elif variant == 'C':
                    variant_name = 'pi_equibot_mixup'
                elif variant == 'D':
                    variant_name = 'pi_equibot_mixup_drop'
                elif variant == 'E':
                    variant_name = 'pi_equibot_full'
                elif variant == 'F':
                    variant_name = 'pi_equibot_full_adapt'
                
                # Load CSV file
                csv_file = os.path.join(results_dir, f"{task}_{variant_name}_seed{seed}.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df_list.append(df)
        
        if df_list:
            dfs[task] = pd.concat(df_list, ignore_index=True)
            # Add human-readable variant names
            variant_mapping = {
                'A': 'Vanilla EquiBot',
                'B': 'PI-EquiBot',
                'C': 'PI-EquiBot + MixUp',
                'D': 'PI-EquiBot + Drop',
                'E': 'PI-EquiBot + Curriculum',
                'F': 'PI-EquiBot + 1-step Adapt'
            }
            dfs[task]['variant_name'] = dfs[task]['variant'].map(variant_mapping)
            
    return dfs


def generate_success_heatmaps(dfs, output_dir):
    """Generate success-rate heatmaps for each task and variant."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(dfs.keys())
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'Vanilla EquiBot',
        'B': 'PI-EquiBot',
        'C': 'PI-EquiBot + MixUp',
        'D': 'PI-EquiBot + Drop',
        'E': 'PI-EquiBot + Curriculum',
        'F': 'PI-EquiBot + 1-step Adapt'
    }
    
    for task in tasks:
        df = dfs[task]
        
        # For cloth tasks (fold, cover), create separate heatmaps for different stiffness/damping settings
        if task in ['fold', 'cover']:
            for stiffness in df['stiffness'].unique():
                for damping in df['damping'].unique():
                    # Create a 2x3 grid of heatmaps (one for each variant)
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    axes = axes.flatten()
                    
                    for i, variant in enumerate(variants):
                        # Filter data for this variant, stiffness, and damping
                        variant_df = df[(df['variant'] == variant) & 
                                        (df['stiffness'] == stiffness) & 
                                        (df['damping'] == damping)]
                        
                        # Create success rate heatmap
                        heatmap_data = variant_df.groupby(['mass', 'friction'])['success'].mean().reset_index()
                        heatmap_data = heatmap_data.pivot(index='mass', columns='friction', values='success')
                        
                        # Plot heatmap
                        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', 
                                   vmin=0, vmax=1, ax=axes[i], cbar=True if i % 3 == 2 else False)
                        axes[i].set_title(f"{variant_names[variant]}")
                        axes[i].set_xlabel('Friction')
                        axes[i].set_ylabel('Mass (kg)')
                    
                    plt.tight_layout()
                    plt.suptitle(f"{task.capitalize()} - Success Rate - Stiffness={stiffness}, Damping={damping}", 
                                y=1.02, fontsize=18)
                    
                    # Save figure
                    filename = f"{task}_heatmap_stiffness{stiffness}_damping{damping}.png"
                    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                    plt.close()
        
        else:  # For close task
            # Create a 2x3 grid of heatmaps (one for each variant)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, variant in enumerate(variants):
                # Filter data for this variant
                variant_df = df[df['variant'] == variant]
                
                # Create success rate heatmap
                heatmap_data = variant_df.groupby(['mass', 'friction'])['success'].mean().reset_index()
                heatmap_data = heatmap_data.pivot(index='mass', columns='friction', values='success')
                
                # Plot heatmap
                sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', 
                           vmin=0, vmax=1, ax=axes[i], cbar=True if i % 3 == 2 else False)
                axes[i].set_title(f"{variant_names[variant]}")
                axes[i].set_xlabel('Friction')
                axes[i].set_ylabel('Mass (kg)')
            
            plt.tight_layout()
            plt.suptitle(f"{task.capitalize()} - Success Rate", y=1.02, fontsize=18)
            
            # Save figure
            filename = f"{task}_heatmap.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()


def generate_generalization_curves(dfs, output_dir):
    """Generate generalization curves showing success vs OOD distance."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(dfs.keys())
    
    for task in tasks:
        df = dfs[task]
        
        # Calculate OOD distances
        df['ood_mass'] = abs(df['mass'] - 1.0)  # Distance from training mean (1.0)
        df['ood_friction'] = abs(df['friction'] - 0.5)  # Distance from training mean (0.5)
        
        if task in ['fold', 'cover']:
            df['ood_stiffness'] = abs(df['stiffness'] - 0.6)  # Distance from training mean (0.6)
            df['ood_damping'] = abs(df['damping'] - 0.2)  # Distance from training mean (0.2)
            
            # Compute overall OOD distance (normalized sum)
            df['ood_distance'] = (df['ood_mass']/2.0 + df['ood_friction']/0.7 + 
                                 df['ood_stiffness']/0.4 + df['ood_damping']/0.3) / 4
        else:
            # For close task, only use mass and friction
            df['ood_distance'] = (df['ood_mass']/2.0 + df['ood_friction']/0.7) / 2
        
        # Create binned OOD distance for smoother plotting
        df['ood_distance_bin'] = pd.cut(df['ood_distance'], bins=10)
        binned_data = df.groupby(['variant_name', 'ood_distance_bin'])['success'].mean().reset_index()
        binned_data['ood_bin_mid'] = binned_data['ood_distance_bin'].apply(lambda x: x.mid)
        
        # Plot success vs OOD distance
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=binned_data, x='ood_bin_mid', y='success', hue='variant_name', 
                    marker='o', linewidth=2.5, markersize=8)
        
        plt.title(f"{task.capitalize()} - Success vs Out-of-Distribution Distance", fontsize=18)
        plt.xlabel('Out-of-Distribution Distance', fontsize=14)
        plt.ylabel('Success Rate', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Variant", fontsize=12)
        
        # Add vertical line at 0 (in-distribution)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        plt.text(0.01, 0.02, "In-Distribution", rotation=90, 
                verticalalignment='bottom', horizontalalignment='left')
        
        # Save figure
        filename = f"{task}_generalization_curve.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual parameter curves
        for param in ['mass', 'friction']:
            plt.figure(figsize=(12, 8))
            ood_col = f'ood_{param}'
            param_data = df.groupby(['variant_name', param])['success'].mean().reset_index()
            
            sns.lineplot(data=param_data, x=param, y='success', hue='variant_name', 
                        marker='o', linewidth=2.5, markersize=8)
            
            plt.title(f"{task.capitalize()} - Success vs {param.capitalize()}", fontsize=18)
            plt.xlabel(f"{param.capitalize()}" + (" (kg)" if param == 'mass' else ""), fontsize=14)
            plt.ylabel('Success Rate', fontsize=14)
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend(title="Variant", fontsize=12)
            
            # Add vertical line at training mean
            mean_val = 1.0 if param == 'mass' else 0.5
            plt.axvline(x=mean_val, color='gray', linestyle='--', alpha=0.7)
            plt.text(mean_val+0.01, 0.02, "Training Mean", rotation=90, 
                    verticalalignment='bottom', horizontalalignment='left')
            
            # Save figure
            filename = f"{task}_success_vs_{param}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()


def generate_physics_estimation_plots(dfs, output_dir):
    """Generate scatter plots showing physics estimation quality."""
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(dfs.keys())
    
    for task in tasks:
        df = dfs[task]
        
        # Filter out variants without physics encoder
        df_phys = df[df['variant'] != 'A'].copy()  # Skip vanilla EquiBot
        
        # Remove N/A values from phys_est_error
        df_phys = df_phys[df_phys['phys_est_error'] != 'N/A'].copy()
        df_phys['phys_est_error'] = df_phys['phys_est_error'].astype(float)
        
        # Create plot of physics estimation error across variants
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='variant_name', y='phys_est_error', data=df_phys, palette='viridis')
        
        plt.title(f"{task.capitalize()} - Physics Estimation Error by Variant", fontsize=18)
        plt.xlabel('Variant', fontsize=14)
        plt.ylabel('Physics Estimation Error (L2)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Save figure
        filename = f"{task}_physics_error.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot physics error vs. success
        plt.figure(figsize=(12, 8))
        for variant in df_phys['variant'].unique():
            variant_df = df_phys[df_phys['variant'] == variant]
            variant_name = variant_df['variant_name'].iloc[0]
            
            # Create error bins and calculate success rate for each bin
            variant_df['error_bin'] = pd.cut(variant_df['phys_est_error'], bins=10)
            binned_data = variant_df.groupby('error_bin')['success'].mean().reset_index()
            binned_data['error_bin_mid'] = binned_data['error_bin'].apply(lambda x: x.mid if hasattr(x, 'mid') else None)
            binned_data = binned_data.dropna(subset=['error_bin_mid'])
            
            plt.plot(binned_data['error_bin_mid'], binned_data['success'], marker='o', 
                    linewidth=2.5, markersize=8, label=variant_name)
        
        plt.title(f"{task.capitalize()} - Success Rate vs Physics Error", fontsize=18)
        plt.xlabel('Physics Estimation Error (L2)', fontsize=14)
        plt.ylabel('Success Rate', fontsize=14)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Variant", fontsize=12)
        
        # Save figure
        filename = f"{task}_success_vs_physics_error.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def generate_ablation_table(dfs, output_dir):
    """Generate ablation summary table."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dictionary to store results
    table_data = {
        'Variant': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Variant_Name': ['Vanilla EquiBot', 'PI-EquiBot', 'PI-EquiBot + MixUp',
                         'PI-EquiBot + Drop', 'PI-EquiBot + Curriculum', 'PI-EquiBot + 1-step Adapt']
    }
    
    # Calculate success rates and physics errors for each task
    for task in dfs.keys():
        df = dfs[task]
        
        # Success rate for each variant
        sr_data = []
        for variant in table_data['Variant']:
            variant_df = df[df['variant'] == variant]
            sr = variant_df['success'].mean()
            sr_data.append(round(sr, 2))
        
        table_data[f'{task.capitalize()} SR ↑'] = sr_data
        
        # Physics error for each variant (except A, which has no physics encoder)
        pe_data = []
        for variant in table_data['Variant']:
            if variant == 'A':
                pe_data.append('—')  # No physics encoder
            else:
                variant_df = df[df['variant'] == variant]
                variant_df = variant_df[variant_df['phys_est_error'] != 'N/A'].copy()
                variant_df['phys_est_error'] = variant_df['phys_est_error'].astype(float)
                pe = variant_df['phys_est_error'].mean()
                pe_data.append(round(pe, 2) if not np.isnan(pe) else '—')
        
        table_data[f'{task.capitalize()} Phys-Err ↓'] = pe_data
    
    # Create DataFrame for the table
    table_df = pd.DataFrame(table_data)
    
    # Calculate mean values across tasks
    table_df['Mean SR ↑'] = table_df[[col for col in table_df.columns if 'SR ↑' in col]].mean(axis=1).round(2)
    
    # For phys err, need to handle '—' values
    phys_err_cols = [col for col in table_df.columns if 'Phys-Err ↓' in col]
    
    # Convert '—' to NaN for calculation
    for col in phys_err_cols:
        table_df[col] = pd.to_numeric(table_df[col], errors='coerce')
    
    table_df['Mean Phys-Err ↓'] = table_df[phys_err_cols].mean(axis=1).round(2)
    
    # Convert back to string format for display
    for col in phys_err_cols + ['Mean Phys-Err ↓']:
        table_df[col] = table_df[col].apply(lambda x: '—' if pd.isna(x) else round(x, 2))
    
    # Save as CSV
    table_df.to_csv(os.path.join(output_dir, 'ablation_summary.csv'), index=False)
    
    # Also create a nicely formatted Markdown table
    md_columns = ['Variant', 'Fold SR ↑', 'Cover SR ↑', 'Close SR ↑', 'Mean Phys-Err ↓']
    md_table = table_df[md_columns].copy()
    
    with open(os.path.join(output_dir, 'ablation_summary.md'), 'w') as f:
        # Write table header
        f.write('| Variant | Fold SR ↑ | Cover SR ↑ | Close SR ↑ | Mean Phys-Err ↓ |\n')
        f.write('| ------- | --------- | ---------- | ---------- | --------------- |\n')
        
        # Write table rows
        for _, row in md_table.iterrows():
            variant_idx = table_data['Variant'].index(row['Variant'])
            variant_name = table_data['Variant_Name'][variant_idx]
            
            # Bold for best values
            if row['Variant'] in ['E', 'F']:
                f.write(f"| **{variant_name}** | **{row['Fold SR ↑']}** | **{row['Cover SR ↑']}** | " +
                        f"**{row['Close SR ↑']}** | **{row['Mean Phys-Err ↓']}** |\n")
            else:
                f.write(f"| {variant_name} | {row['Fold SR ↑']} | {row['Cover SR ↑']} | " +
                        f"{row['Close SR ↑']} | {row['Mean Phys-Err ↓']} |\n")
    
    print(f"Ablation summary table saved to {output_dir}/ablation_summary.csv and .md")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    dfs = load_results(args.results_dir, args.combined)
    
    if not dfs:
        print("No results found. Please check the results directory.")
        return
    
    print(f"Loaded results for tasks: {list(dfs.keys())}")
    
    # Generate plots
    print("Generating success-rate heatmaps...")
    generate_success_heatmaps(dfs, os.path.join(args.output_dir, 'heatmaps'))
    
    print("Generating generalization curves...")
    generate_generalization_curves(dfs, os.path.join(args.output_dir, 'curves'))
    
    print("Generating physics estimation plots...")
    generate_physics_estimation_plots(dfs, os.path.join(args.output_dir, 'physics'))
    
    print("Generating ablation summary table...")
    generate_ablation_table(dfs, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 