import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import random
from tqdm import tqdm
import os
import pickle
from collections import defaultdict
import re
from itertools import cycle

# Your existing imports (assuming src.alt exists)
# from src.alt import *
# reset_seeds()

def gauss_entropy(cov, dimension):
    """Compute Gaussian entropy given covariance matrix and dimension"""
    entropy = dimension * 0.5 * np.log(2*np.pi*np.e) + 0.5 * np.log(np.linalg.det(cov))
    return entropy

def create_diagonal_cov_matrix(variance_value, dimension):
    """Create diagonal covariance matrix from variance value"""
    return np.diag([variance_value] * dimension)

def extract_variance_from_path(variance_dir_name):
    """Extract variance value from directory name (assumes format like 'variance_0.1' or similar)"""
    # This assumes the variance is in the directory name
    # You may need to adjust this based on your actual directory naming
    match = re.search(r'[\d.]+', variance_dir_name)
    if match:
        return float(match.group())
    else:
        # If no number found, try to convert the whole string
        try:
            return float(variance_dir_name)
        except ValueError:
            raise ValueError(f"Could not extract variance from directory name: {variance_dir_name}")

def process_single_csv(csv_path, variance_value, dimension):
    """Process a single CSV file and use the actual predicted entropy values and upper bounds"""
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Store entropy values and upper bounds for each model
        entropy_values = {}
        upper_bound_values = {}
        
        # Process each model column (skip Dataset_Size column)
        for column in df.columns:
            if column != 'Dataset_Size':
                if column.endswith('_upper'):
                    # This is an upper bound column
                    model_name = column.replace('_upper', '')
                    upper_bound_values[model_name] = df[column].tolist()
                else:
                    # This is an entropy column
                    model_name = column
                    entropy_values[model_name] = df[column].tolist()
        
        return entropy_values, upper_bound_values, df['Dataset_Size'].tolist()
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None, None

def aggregate_results(data_dir, dimension, context_size):
    """Main function to aggregate entropy results and upper bounds across all trials and variances"""
    
    # Get all variance directories
    variances = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Data structure to store results: {variance: {model: {dataset_size: [entropy_values]}}}
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_upper_bounds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Process each variance directory
    for variance_dir in tqdm(variances, desc="Processing variance directories"):
        variance_path = os.path.join(data_dir, variance_dir)
        
        # Extract variance value from directory name
        try:
            variance_value = extract_variance_from_path(variance_dir)
        except ValueError as e:
            print(f"Skipping directory {variance_dir}: {e}")
            continue
        
        # Get all CSV files in this variance directory
        trial_files = [f for f in os.listdir(variance_path) if f.endswith('.csv')]
        
        print(f"Processing variance {variance_value} with {len(trial_files)} trial files")
        
        # Process each trial file
        for trial_file in tqdm(trial_files, desc=f"Processing trials for variance {variance_value}", leave=False):
            csv_path = os.path.join(variance_path, trial_file)
            
            entropy_values, upper_bound_values, dataset_sizes = process_single_csv(csv_path, variance_value, dimension)
            
            if entropy_values is not None:
                # Store entropy results organized by dataset size
                for i, dataset_size in enumerate(dataset_sizes):
                    for model_name, entropies in entropy_values.items():
                        all_results[variance_value][model_name][dataset_size].append(entropies[i])
                
                # Store upper bound results organized by dataset size
                for i, dataset_size in enumerate(dataset_sizes):
                    for model_name, upper_bounds in upper_bound_values.items():
                        all_upper_bounds[variance_value][model_name][dataset_size].append(upper_bounds[i])
    
    return all_results, all_upper_bounds

def compute_theoretical_bounds(variances, dimension):
    """Compute theoretical bounds (Gaussian entropy) for each variance"""
    bounds = {}
    for variance in variances:
        cov_matrix = create_diagonal_cov_matrix(variance, dimension)
        theoretical_entropy = gauss_entropy(cov_matrix, dimension)
        bounds[variance] = theoretical_entropy
    return bounds

def create_diagonal_cov_matrix(variance_value, dimension):
    """Create diagonal covariance matrix from variance value"""
    return np.diag([variance_value] * dimension)

def compute_statistics(all_results, all_upper_bounds):
    """Compute mean and standard deviation for each variance/model/dataset_size combination"""
    stats_results = {}
    upper_bound_stats = {}
    
    # Process entropy results
    for variance_value in all_results:
        stats_results[variance_value] = {}
        
        for model_name in all_results[variance_value]:
            stats_results[variance_value][model_name] = {}
            
            for dataset_size in all_results[variance_value][model_name]:
                entropy_values = all_results[variance_value][model_name][dataset_size]
                
                if len(entropy_values) > 0:
                    mean_entropy = np.mean(entropy_values)
                    std_entropy = np.std(entropy_values, ddof=1) if len(entropy_values) > 1 else 0.0
                    
                    stats_results[variance_value][model_name][dataset_size] = {
                        'mean': mean_entropy,
                        'std': std_entropy,
                        'n_trials': len(entropy_values)
                    }
    
    # Process upper bound results
    for variance_value in all_upper_bounds:
        upper_bound_stats[variance_value] = {}
        
        for model_name in all_upper_bounds[variance_value]:
            upper_bound_stats[variance_value][model_name] = {}
            
            for dataset_size in all_upper_bounds[variance_value][model_name]:
                upper_bound_values = all_upper_bounds[variance_value][model_name][dataset_size]
                
                if len(upper_bound_values) > 0:
                    mean_upper_bound = np.mean(upper_bound_values)
                    std_upper_bound = np.std(upper_bound_values, ddof=1) if len(upper_bound_values) > 1 else 0.0
                    
                    upper_bound_stats[variance_value][model_name][dataset_size] = {
                        'mean': mean_upper_bound,
                        'std': std_upper_bound,
                        'n_trials': len(upper_bound_values)
                    }
    
    return stats_results, upper_bound_stats

def create_entropy_visualization(stats_results, upper_bound_stats, dimension, save_path=None):
    """Create visualization showing means with theoretical bounds and upper bounds"""
    
    # Define colors for each solver
    solver_colors = {
        'least_squares': '#e74c3c',  # Red
        'ridge': '#3498db',          # Blue
        'lasso': '#2ecc71',          # Green
        'oracle': '#9b59b6'          # Purple
    }
    
    # Define shapes (markers) for each variance - smaller sizes
    variance_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    
    # Get all unique variances and solvers
    all_variances = sorted(stats_results.keys())
    all_solvers = set()
    for variance_results in stats_results.values():
        all_solvers.update(variance_results.keys())
    all_solvers = sorted(all_solvers)
    
    # Create shape mapping for variances
    variance_shape_map = {var: variance_shapes[i % len(variance_shapes)] 
                         for i, var in enumerate(all_variances)}
    
    # Compute theoretical bounds
    theoretical_bounds = compute_theoretical_bounds(all_variances, dimension)
    
    # Set up the plot with extra width for legends - increased figure width
    fig, ax = plt.subplots(figsize=(24, 12), constrained_layout=True)
    
    # Get all dataset sizes for consistent x-axis
    all_dataset_sizes = set()
    for variance_results in stats_results.values():
        for solver_results in variance_results.values():
            all_dataset_sizes.update(solver_results.keys())
    all_dataset_sizes = sorted(all_dataset_sizes)
    
    # Transform dataset sizes to test sizes (floor(0.2 * dataset_size))
    all_test_sizes = [int(np.floor(0.2 * ds)) for ds in all_dataset_sizes]
    
    # Plot theoretical bounds as horizontal lines
    x_range = [min(all_test_sizes), max(all_test_sizes)]
    for variance in all_variances:
        theoretical_entropy = theoretical_bounds[variance]
        ax.plot(x_range, [theoretical_entropy, theoretical_entropy], 
                color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add markers at regular intervals to show the variance shape
        x_positions = np.logspace(np.log10(min(all_test_sizes)), 
                                 np.log10(max(all_test_sizes)), 8)
        ax.scatter(x_positions, [theoretical_entropy] * len(x_positions),
                   marker=variance_shape_map[variance], color='black', 
                   s=30, alpha=0.7, zorder=5)
    
    # Plot for each solver and variance combination
    for solver in all_solvers:
        for variance in all_variances:
            if solver in stats_results[variance]:
                # Collect data points for entropy (main lines)
                plot_data = []
                
                for ds in sorted(all_dataset_sizes):
                    if ds in stats_results[variance][solver]:
                        test_size = int(np.floor(0.2 * ds))
                        mean_val = stats_results[variance][solver][ds]['mean']
                        std_val = stats_results[variance][solver][ds]['std']
                        plot_data.append((test_size, mean_val, std_val))
                
                if plot_data:  # Only plot if we have data
                    # Sort by test size to ensure proper line plotting
                    plot_data.sort(key=lambda x: x[0])
                    
                    # Separate the data
                    test_sizes = [x[0] for x in plot_data]
                    means = [x[1] for x in plot_data]
                    stds = [x[2] for x in plot_data]
                    
                    # Convert to numpy arrays for easier manipulation
                    test_sizes_np = np.array(test_sizes)
                    means_np = np.array(means)
                    stds_np = np.array(stds)
                    
                    # Plot confidence band (mean ± 1 std)
                    ax.fill_between(
                        test_sizes_np,
                        means_np - stds_np,
                        means_np + stds_np,
                        color=solver_colors.get(solver, '#000000'),
                        alpha=0.2,
                        zorder=1
                    )
                    
                    # Plot means line with markers
                    ax.plot(
                        test_sizes,
                        means, 
                        marker=variance_shape_map[variance],
                        color=solver_colors.get(solver, '#000000'),
                        linestyle='-',
                        linewidth=2,
                        markersize=5,
                        label=f'{solver} (σ²={variance})',
                        alpha=0.8,
                        zorder=3
                    )
            
            # Plot upper bounds (dashed lines) - SKIP FOR ORACLE
            if solver != 'oracle' and solver in upper_bound_stats.get(variance, {}):
                # Collect data points for upper bounds
                upper_plot_data = []
                
                for ds in sorted(all_dataset_sizes):
                    if ds in upper_bound_stats[variance][solver]:
                        test_size = int(np.floor(0.2 * ds))
                        upper_mean = upper_bound_stats[variance][solver][ds]['mean']
                        upper_plot_data.append((test_size, upper_mean))
                
                if upper_plot_data:  # Only plot if we have data
                    # Sort by test size to ensure proper line plotting
                    upper_plot_data.sort(key=lambda x: x[0])
                    
                    # Separate the data
                    upper_test_sizes = [x[0] for x in upper_plot_data]
                    upper_means = [x[1] for x in upper_plot_data]
                    
                    # Plot upper bound line with same color but dashed
                    ax.plot(
                        upper_test_sizes,
                        upper_means, 
                        marker=variance_shape_map[variance],
                        color=solver_colors.get(solver, '#000000'),
                        linestyle='--',  # Dashed line for upper bounds
                        linewidth=2,
                        markersize=5,
                        label=f'{solver} upper (σ²={variance})',
                        alpha=0.8,
                        zorder=3
                    )
    
    # Customize the plot
    ax.set_xscale('log')
    ax.set_xlabel('Test Dataset Size', fontsize=24)  # Label still accurate
    ax.set_ylabel('nats/symbol', fontsize=24)
    ax.set_title('Model PECEP vs Theoretical Additive Gaussian Noise and Hadamard Bounds Across Dataset Sizes', 
                 fontsize=24)
    ax.grid(True, alpha=0.3)
    
    # Set custom x-axis ticks to show all test dataset sizes
    unique_test_sizes = sorted(set(all_test_sizes))
    ax.set_xticks(unique_test_sizes)
    ax.set_xticklabels([f'{size:.0e}' for size in unique_test_sizes])
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    # Create custom legend
    # Create legend for solvers (colors) - include both solid and dashed lines
    solver_legend_elements = []
    for solver in all_solvers:
        if any(solver in stats_results[var] for var in all_variances):
            # Add solid line for main entropy
            solver_legend_elements.append(
                plt.Line2D([0], [0], color=solver_colors.get(solver, '#000000'), 
                          linestyle='-', linewidth=2, label=f'{solver}')
            )
            # Add dashed line for upper bound (skip oracle)
            if solver != 'oracle':
                solver_legend_elements.append(
                    plt.Line2D([0], [0], color=solver_colors.get(solver, '#000000'), 
                              linestyle='--', linewidth=2, label=f'{solver}\nhadamard bound')
                )
    
    # Add theoretical bounds to legend
    solver_legend_elements.append(
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                   label='Theoretical Bounds')
    )
    
    # Create legend for variances (shapes)
    variance_legend_elements = [
        plt.Line2D([0], [0], color='black', marker=variance_shape_map[var], 
                   linestyle='', markersize=5, label=f'σ²={var}')
        for var in all_variances
    ]
    
    # Add legends outside the plot area with better positioning
    # Add legends outside the plot area with better positioning
    legend1 = fig.legend(
    handles=solver_legend_elements, title='Curves',
    loc='center left', bbox_to_anchor=(1.001, 0.70),  # figure coords
    fontsize=18, title_fontsize=20, frameon=False  # Increased from 10 and 11
    )
    legend2 = fig.legend(
        handles=variance_legend_elements, title='Noise Levels',
        loc='center left', bbox_to_anchor=(1.001, 0.32),  # figure coords
        fontsize=18, title_fontsize=20, frameon=False  # Increased from 10 and 11
    )
    
    # Add the first legend back
    ax.add_artist(legend1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# Updated main execution
if __name__ == "__main__":
    # Configuration
    data_dir = "results/synthetic_data_exp1_izs/"
    dimension = 32
    context_size = 8
    
    print(f"Starting entropy analysis...")
    print(f"Data directory: {data_dir}")
    print(f"Dimension: {dimension}")
    print(f"Context size: {context_size}")
    
    # Process all data (now returns both entropy and upper bounds)
    all_results, all_upper_bounds = aggregate_results(data_dir, dimension, context_size)
    
    # Compute statistics (now for both entropy and upper bounds)
    stats_results, upper_bound_stats = compute_statistics(all_results, all_upper_bounds)
    
    # Print summary for debugging
    #print_summary(stats_results)
    
    # Create visualization (now includes upper bounds)
    create_entropy_visualization(stats_results, upper_bound_stats, dimension, save_path="sanity_check/exp1_vis_2.png")
    
    print(f"\nAnalysis complete!")

def print_summary(stats_results):
    """Print a summary of the results"""
    print("\n=== RESULTS SUMMARY ===")
    
    for variance_value in sorted(stats_results.keys()):
        print(f"\nVariance: {variance_value}")
        
        for model_name in sorted(stats_results[variance_value].keys()):
            print(f"  Model: {model_name}")
            
            dataset_sizes = sorted(stats_results[variance_value][model_name].keys())
            for dataset_size in dataset_sizes:
                stats = stats_results[variance_value][model_name][dataset_size]
                print(f"    Dataset Size {dataset_size}: "
                      f"Mean = {stats['mean']:.4f}, "
                      f"Std = {stats['std']:.4f}, "
                      f"N = {stats['n_trials']}")

def extract_variance_from_path(variance_dir_name):
    """Extract variance value from directory name (assumes format like 'variance_0.1' or similar)"""
    # This assumes the variance is in the directory name
    # You may need to adjust this based on your actual directory naming
    import re
    match = re.search(r'[\d.]+', variance_dir_name)
    if match:
        return float(match.group())
    else:
        # If no number found, try to convert the whole string
        try:
            return float(variance_dir_name)
        except ValueError:
            raise ValueError(f"Could not extract variance from directory name: {variance_dir_name}")