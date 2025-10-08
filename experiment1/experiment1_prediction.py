import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import random
from tqdm import tqdm
import os
import pickle
from joblib import Parallel, delayed

from src.alt import *



reset_seeds()

data_dir = "results/synthetic_data_exp1_izs"
dimension = 32
context_size = 8
variances = os.listdir(data_dir)

solvers = ["oracle"]#,"least_squares"]#, "ridge", "lasso"]

# Collecting the paths to all of the datasets
dataset_paths = []
for variance in variances:
    variance_dir = os.path.join(data_dir, variance)
    trial_files = os.listdir(variance_dir)
    for trial_file in trial_files:
        if trial_file.endswith(".pkl"):
            dataset_path = os.path.join(variance_dir, trial_file)
            dataset_paths.append(dataset_path)

print(f"Total datasets collected: {len(dataset_paths)}")

# I know that I generate 5 million sample datapoints
dataset_sizes = [int(1e3)]#, int(5e3), int(1e4)]#, int(5e4), int(1e5), int(5e5), int(1e6)]#, int(2.5e6), int(5e6)]
#dataset_sizes = list(range(int(1e3), int(1e6)+1, int(1e3)))

def process_solver_combination(dataset_path, cur_size, solver, context_size, dimension):
    """Process a single (dataset, size, solver) combination"""
    #print("Data Path: ",dataset_path)
    full_data, coefficients, noise_cov_mat = load_pickled_data(dataset_path)
    data = full_data[:cur_size]
    
    estimated_cov, test_predictions = estimate_noise_and_data(data, context_size, solve_method=solver, gt_coefficients=coefficients)
    entropy_value = gauss_entropy(estimated_cov, dimension)
    hadamard_upper = hadamard_upper_bound(estimated_cov, dimension)
    
    return dataset_path, cur_size, solver, entropy_value, hadamard_upper

# Generate all combinations
combinations = [
    (dataset_path, cur_size, solver)
    for dataset_path in dataset_paths
    for cur_size in dataset_sizes
    for solver in solvers
]

# Process ALL combinations in parallel
results = Parallel(n_jobs=40, verbose=2, backend='multiprocessing')(
    delayed(process_solver_combination)(dataset_path, cur_size, solver, context_size, dimension)
    for dataset_path, cur_size, solver in combinations
)

# Reorganize results by dataset - MODIFIED to include hadamard_upper
dataset_results = {}
dataset_upper_bounds = {}  # New: separate storage for upper bounds

for dataset_path, cur_size, solver, entropy_value, hadamard_upper in results:
    if dataset_path not in dataset_results:
        dataset_results[dataset_path] = {}
        dataset_upper_bounds[dataset_path] = {}  # New
    
    if cur_size not in dataset_results[dataset_path]:
        dataset_results[dataset_path][cur_size] = {}
        dataset_upper_bounds[dataset_path][cur_size] = {}  # New
    
    # Store both entropy and upper bound
    dataset_results[dataset_path][cur_size][solver] = entropy_value
    dataset_upper_bounds[dataset_path][cur_size][f"{solver}_upper"] = hadamard_upper  # New

# Save results for each dataset - MODIFIED
for dataset_path, results_dict in dataset_results.items():
    csv_output_path = dataset_path.replace('.pkl', '.csv')
    
    # Combine entropy results and upper bounds
    combined_results = {}
    for cur_size in results_dict:
        combined_results[cur_size] = {}
        # Add entropy values
        combined_results[cur_size].update(results_dict[cur_size])
        # Add upper bound values
        combined_results[cur_size].update(dataset_upper_bounds[dataset_path][cur_size])
    
    results_df = pd.DataFrame.from_dict(combined_results, orient='index')
    results_df.index.name = 'Dataset_Size'
    results_df.to_csv(csv_output_path)
    print(f"✓ Completed: {csv_output_path}")