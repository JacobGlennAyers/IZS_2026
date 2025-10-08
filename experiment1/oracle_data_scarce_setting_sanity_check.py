import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import pickle
from joblib import Parallel, delayed
from utils import *  # Your existing functions

def reset_seeds(val=42):
    """Reset random seeds for reproducibility"""
    np.random.seed(val)
    random.seed(val)

def calculate_true_gaussian_entropy(dimension, variance):
    """Calculate true differential entropy of multivariate Gaussian noise"""
    # For multivariate Gaussian with covariance matrix σ²I:
    # h(X) = (d/2) * log(2πe) + (1/2) * log(det(σ²I))
    # h(X) = (d/2) * log(2πe) + (1/2) * log((σ²)^d)
    # h(X) = (d/2) * log(2πe) + (d/2) * log(σ²)
    return (dimension / 2) * np.log(2 * np.pi * np.e) + (dimension / 2) * np.log(variance)

def process_single_dimension(dimension, fixed_params):
    """Process a single dimension experiment"""
    # Extract fixed parameters
    context_size = fixed_params['context_size']
    dataset_size = fixed_params['dataset_size']
    variance = fixed_params['variance']
    matrix_normalization_method = fixed_params['matrix_normalization_method']
    decay = fixed_params['decay']
    rate = fixed_params['rate']
    num_trials = fixed_params['num_trials']
    
    print(f"Processing dimension: {dimension}")
    
    # Store results for this dimension
    oracle_entropies = []
    ols_entropies = []
    
    for trial in range(num_trials):
        try:
            # Set seed for reproducibility
            reset_seeds(trial)
            
            # Generate synthetic data
            full_data, coefficients, noise_cov_mat = generate_random_data(
                dimension=dimension,
                context_size=context_size,
                dataset_size=dataset_size,
                stationary_variance=variance,
                matrix_normalization_method=matrix_normalization_method,
                decay=decay,
                rate=rate
            )
            
            # Use only the first part of the data to ensure we get exactly 200 test points
            # We need context_size + train_size + 200 total points
            train_size = int(dataset_size * 0.8)
            test_size = 200
            required_size = context_size + train_size + test_size
            
            if len(full_data) < required_size:
                print(f"Warning: Generated data too small for dimension {dimension}, trial {trial}")
                continue
                
            data = full_data[:required_size]
            
            # Oracle prediction (using ground truth coefficients)
            try:
                estimated_cov_oracle, _ = estimate_noise_and_data(
                    data, context_size, train_percent=0.8, 
                    solve_method='oracle', gt_coefficients=coefficients
                )
                entropy_oracle = gauss_entropy(estimated_cov_oracle, dimension)
                oracle_entropies.append(entropy_oracle)
            except Exception as e:
                print(f"Oracle failed for dimension {dimension}, trial {trial}: {e}")
                continue
            
            # OLS prediction
            try:
                estimated_cov_ols, _ = estimate_noise_and_data(
                    data, context_size, train_percent=0.8, 
                    solve_method='least_squares'
                )
                entropy_ols = gauss_entropy(estimated_cov_ols, dimension)
                ols_entropies.append(entropy_ols)
            except Exception as e:
                print(f"OLS failed for dimension {dimension}, trial {trial}: {e}")
                continue
                
        except Exception as e:
            print(f"Trial {trial} failed for dimension {dimension}: {e}")
            continue
    
    # Calculate statistics
    oracle_mean = np.mean(oracle_entropies) if oracle_entropies else np.nan
    oracle_std = np.std(oracle_entropies) if oracle_entropies else np.nan
    ols_mean = np.mean(ols_entropies) if ols_entropies else np.nan
    ols_std = np.std(ols_entropies) if ols_entropies else np.nan
    
    # Calculate true entropy
    true_entropy = calculate_true_gaussian_entropy(dimension, variance)
    
    return {
        'dimension': dimension,
        'oracle_mean': oracle_mean,
        'oracle_std': oracle_std,
        'ols_mean': ols_mean,
        'ols_std': ols_std,
        'true_entropy': true_entropy,
        'oracle_trials': len(oracle_entropies),
        'ols_trials': len(ols_entropies)
    }

def main():
    """Main execution function"""
    
    # Fixed parameters (based on your original experiment)
    fixed_params = {
        'context_size': 8,
        'dataset_size': int(5e3),  # Smaller dataset since we only need 200 test points
        'variance': 1e-2,  # Using one of your middle variance values
        'matrix_normalization_method': 'fro',
        'decay': True,
        'rate': 0.85,
        'num_trials': 10  # Reduced for faster execution
    }
    
    # Dimension sweep
    dimensions = list(range(8, 128))  # 2 to 48 inclusive
    
    print(f"Running dimension sweep from {min(dimensions)} to {max(dimensions)}")
    print(f"Fixed parameters: {fixed_params}")
    
    # Process all dimensions in parallel
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_single_dimension)(dim, fixed_params) 
        for dim in tqdm(dimensions, desc="Processing dimensions")
    )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "dimension_sweep_results.csv"), index=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot true entropy
    ax.plot(results_df['dimension'], results_df['true_entropy'], 
            'k-', linewidth=3, label='True Gaussian Entropy', zorder=3)
    
    # Plot Oracle results with error bars
    valid_oracle = ~np.isnan(results_df['oracle_mean'])
    if valid_oracle.any():
        ax.errorbar(results_df.loc[valid_oracle, 'dimension'], 
                   results_df.loc[valid_oracle, 'oracle_mean'],
                   yerr=results_df.loc[valid_oracle, 'oracle_std'],
                   fmt='o-', color='red', alpha=0.7, capsize=3,
                   label='Oracle PECEP', zorder=2)
    
    # Plot OLS results with error bars
    valid_ols = ~np.isnan(results_df['ols_mean'])
    if valid_ols.any():
        ax.errorbar(results_df.loc[valid_ols, 'dimension'], 
                   results_df.loc[valid_ols, 'ols_mean'],
                   yerr=results_df.loc[valid_ols, 'ols_std'],
                   fmt='s-', color='blue', alpha=0.7, capsize=3,
                   label='OLS PECEP', zorder=1)
    
    # Formatting
    ax.set_xlabel('Dimension', fontsize=14)
    ax.set_ylabel('Entropy (nats)', fontsize=14)
    ax.set_title('PECEP vs True Gaussian Entropy Across Dimensions\n'
                f'(Variance={fixed_params["variance"]}, 200 test points, '
                f'{fixed_params["num_trials"]} trials)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    if not results_df['oracle_mean'].isna().all():
        y_min = min(results_df['true_entropy'].min(), 
                   results_df['oracle_mean'].min() - results_df['oracle_std'].max())
        y_max = max(results_df['true_entropy'].max(), 
                   results_df['oracle_mean'].max() + results_df['oracle_std'].max())
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join("figures", "dimension_sweep_plot.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Dimensions tested: {min(dimensions)} to {max(dimensions)}")
    print(f"Successful Oracle trials: {results_df['oracle_trials'].sum()}")
    print(f"Successful OLS trials: {results_df['ols_trials'].sum()}")
    
    # Check for oracle below bound
    oracle_below_bound = results_df['oracle_mean'] < results_df['true_entropy']
    if oracle_below_bound.any():
        print(f"\nOracle below theoretical bound in {oracle_below_bound.sum()} dimensions:")
        below_dims = results_df.loc[oracle_below_bound, 'dimension'].tolist()
        print(f"Dimensions: {below_dims}")
        
        # Calculate average deficit
        deficits = results_df.loc[oracle_below_bound, 'true_entropy'] - results_df.loc[oracle_below_bound, 'oracle_mean']
        print(f"Average deficit: {deficits.mean():.4f} nats")
        print(f"Max deficit: {deficits.max():.4f} nats")
    
    return results_df

if __name__ == "__main__":
    results = main()
    print("Dimension sweep experiment completed!")