import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

def reset_seeds(val=42):
    # Set the seed for numpy
    np.random.seed(val)
    # Set the seed for Python's random library
    random.seed(val)

def MSE( gt, pred, axis=None):
    MSE = np.mean(np.square(np.subtract( gt, pred)), axis=axis)
    return MSE
def error_cov_matrix_and_det(predicted, actual):
    difference_matrix = actual - predicted
    diff_cov = np.cov(np.transpose(difference_matrix))
    #print(diff_cov.shape)
    return diff_cov, np.linalg.det(diff_cov)
def visualize_spectrogram(spec):
    plt.figure(figsize=(12, 3))
    plt.imshow(spec.T, aspect='auto', origin='lower', cmap='grey')
    #plt.colorbar(label='Intensity')
    plt.title('Synthetic Spectrogram')
    plt.xlabel('"Time"')
    plt.ylabel('"Frequencies"')
    plt.tight_layout()
    plt.show()
    plt.close()




def visualize_MSE(gt_spec, predicted_spec, title):
    gt_spec = gt_spec.T
    predicted_spec = predicted_spec.T
    MSE_plot = np.mean((gt_spec - predicted_spec) ** 2, axis=0)

    # Determine common vmin and vmax for consistent color scale
    vmin = min(gt_spec.min(), predicted_spec.min(), (gt_spec - predicted_spec).min())
    vmax = max(gt_spec.max(), predicted_spec.max(), (gt_spec - predicted_spec).max())

    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.4})

    # Add color bar space
    cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.02])

    # Plot ground truth spectrogram
    img1 = axs[0].imshow(gt_spec, aspect='auto', origin='lower', cmap='grey', vmin=vmin, vmax=vmax)
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Ground Truth Spectrogram')

    # Plot predicted spectrogram
    axs[1].imshow(predicted_spec, aspect='auto', origin='lower', cmap='grey', vmin=vmin, vmax=vmax)
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Predicted Spectrogram')

    # Plot the Difference
    axs[2].imshow(gt_spec - predicted_spec, aspect='auto', origin='lower', cmap='grey', vmin=vmin, vmax=vmax)
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Difference (Ground Truth - Predicted)')

    # Add color bar to align with the first three plots
    fig.colorbar(img1, cax=cbar_ax, orientation='horizontal')

    # Plot MSE
    axs[3].plot(np.arange(MSE_plot.shape[0]), MSE_plot, color='red', label='MSE')
    axs[3].set_xlabel('Time Frames')
    axs[3].set_ylabel('MSE')
    axs[3].tick_params(axis='y')
    axs[3].legend(loc='upper right')

    # Add title and save the figure
    fig.suptitle(title)
    output_path = os.path.join(f"{title.replace(' ', '_')}_plot.png")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
    plt.close()


def normalize_spectral_norm(matrix):
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(matrix)
    
    # Normalize the singular values into a histogram
    S_normalized = S / np.sum(S)#S[0]
    
    # Reconstruct the normalized matrix
    normalized_matrix = U @ np.diag(S_normalized) @ Vt
    
    return normalized_matrix


def generate_center_band_matrix(n, bandwidth, lower=-1.0, upper=1.0):
    """
    Generate an n x n matrix with random values along the main diagonal
    and within a specified bandwidth around it.

    Parameters:
    - n (int): Size of the matrix (n x n).
    - bandwidth (int): Number of off-diagonal bands to populate.
    - lower (float): Lower bound for random values.
    - upper (float): Upper bound for random values.

    Returns:
    - np.ndarray: The generated matrix.
    """
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((n, n))
    
    # Fill the diagonal and bands within the specified bandwidth with random values
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            matrix[i, j] = np.random.uniform(lower, upper)
    
    return matrix

def generate_gaussian_noise_matrix(n, variance=1.0):
    """
    Generate an n x n matrix with random values distributed around the main diagonal,
    defined by a Gaussian distribution with the specified variance.

    Parameters:
    - n (int): Size of the matrix (n x n).
    - variance (float): Variance along the diagonal for the data being generated.

    Returns:
    - np.ndarray: The generated matrix.
    """
    # Create the covariance matrix for the noise distribution
    diagonal = variance * np.ones(n)
    noise_covariance = np.diag(diagonal)
    
    # Generate the random matrix using the multivariate normal distribution
    matrix = np.random.multivariate_normal(mean=np.zeros(n), cov=noise_covariance, size=n)
    
    return matrix

def prepare_data(data, p):
    N, d = data.shape  # N: number of time points, d: dimensionality of each vector

    # Create the design matrix Y
    Y = np.zeros((N - p, p * d))  # (N - p) x (p * d)
    X = data[p:]  # Target values, excluding the first p time steps
    
    # Populate the design matrix Y
    for t in range(p, N):
        # Stack the past p time steps as one row of the design matrix Y
        Y[t - p] = np.hstack([data[t - i - 1] for i in range(p)])
    return X, Y

def estimate_coefficients(X, Y, p):
    """
    Estimate the coefficient matrices A1, A2, ..., Ap for the autoregressive model using least squares.

    Parameters:
    - data: The generated data (N x d), where N is the number of time steps and d is the dimension of each vector.
    - p: The order of the autoregressive model (number of past steps to consider).
    
    Returns:
    - A_matrices: A list of coefficient matrices A1, A2, ..., Ap (each d x d).
    """
    N, d = X.shape  # N: number of time points, d: dimensionality of each vector
    
    # Solve for the coefficient matrices using least squares
    # We are solving for A1, A2, ..., Ap in the equation Y @ A = X
    A_flat = np.linalg.lstsq(Y, X, rcond=None)[0]  # (p * d) x d matrix
    
    # Reshape the result back into p coefficient matrices, each of size (d, d)
    A_matrices = [A_flat[i * d: (i + 1) * d, :] for i in range(p)]
    
    return A_matrices


def estimate_noise_covariance_and_data( data, A_estimates, p):
    """
    Estimate the stationary noise covariance matrix Sigma_epsilon for the autoregressive model.

    Parameters:
    - data: The generated data (N x d), where N is the number of time steps and d is the dimension of each vector.
    - A_matrices: List of coefficient matrices [A1, A2, ..., Ap] obtained from autoregressive model estimation.
    - p: The order of the autoregressive model (number of past steps to consider).
    
    Returns:
    - noise_covariance: The estimated stationary noise covariance matrix (d x d).
    """
    N, d = data.shape  # N: number of time points, d: dimensionality of each vector
    data_estimate = np.zeros((N,d))
    residuals = []  # To store the residuals
    
    # Loop through the data points and compute the residuals
    for t in range(p, N):
        # Predict x_t using the autoregressive model
        x_t_pred = np.zeros(d)
        for i in range(p):
            prior_ndx = t - (i+1)
            cur_weight = np.zeros(data[0].shape)
            if prior_ndx >= 0:
                cur_weight += data[prior_ndx]
            x_t_pred += A_estimates[i] @ cur_weight
        data_estimate[t] = x_t_pred
        
        # Compute the residual (error) at time t
        epsilon_t = data[t] - x_t_pred
        residuals.append(epsilon_t)
    
    # Estimate the noise covariance matrix as the sample covariance of residuals
    residuals = np.array(residuals)
    noise_covariance = np.cov(residuals.T, bias=False)  # Compute sample covariance
    
    return noise_covariance, data_estimate

def generate_random_data( dimension, context_size, dataset_size, stationary_variance=1):
    assert isinstance(dimension, int)
    assert isinstance(context_size, int)
    assert dimension > 0
    assert context_size > 0

    # Defining stationary noise epsilon_t
    stationary_noise_mean = np.zeros((dimension,))
    stationary_noise_covariance_matrix = np.diag( np.zeros( (dimension,) ) + stationary_variance )

    coefficient_matrices = []

    for ndx in range(context_size):
        #cur_coefficient_matrix = normalize_spectral_norm(generate_gaussian_noise_matrix(dimension, 1e-2)) * (1/(ndx+1))
        cur_coefficient_matrix = normalize_spectral_norm(generate_center_band_matrix(dimension, context_size)) * (1/(ndx+1))
        coefficient_matrices.append(cur_coefficient_matrix)



    data = np.zeros(( dataset_size, dimension))
    #print(data[0].shape)
    #print(data.shape)
    for datapoint in range(dataset_size):
        for ndx in range(context_size):
            data[datapoint] += coefficient_matrices[ndx] @ data[datapoint-(ndx+1)]
        data[datapoint] += np.random.multivariate_normal(mean=stationary_noise_mean, cov=stationary_noise_covariance_matrix, size=1).squeeze()
        #print(data[datapoint])
    return data, coefficient_matrices, stationary_noise_covariance_matrix

def generate_deterministic_data( dimension, context_size, dataset_size):
    assert isinstance(dimension, int)
    assert isinstance(context_size, int)
    assert dimension > 0
    assert context_size > 0

    
    coefficient_matrices = []

    for ndx in range(context_size):
        #cur_coefficient_matrix = normalize_spectral_norm(generate_gaussian_noise_matrix(dimension, 1e-2)) * (1/(ndx+1))
        cur_coefficient_matrix = normalize_spectral_norm(generate_center_band_matrix(dimension, context_size)) * (1/(ndx+1))
        coefficient_matrices.append(cur_coefficient_matrix)



    data = np.zeros(( dataset_size, dimension))
    #print(data[0].shape)
    #print(data.shape)
    first_datapoint = True
    for datapoint in range(dataset_size):
        if first_datapoint:
            data[datapoint] += 0.1
            first_datapoint = False
            continue
        for ndx in range(context_size):
            data[datapoint] += coefficient_matrices[ndx] @ data[datapoint-(ndx+1)]
        data[datapoint] += 0.1
        #print(data[datapoint])
    return data, coefficient_matrices

def gauss_exp_entropy(cov, dimension):
    entropy = 0.5*dimension*(np.log((2*np.pi*np.e)) + np.log(np.linalg.det(cov)))
    return np.e**(2*entropy)

def gauss_entropy(cov, dimension):
    entropy = dimension * 0.5 * np.log(2*np.pi*np.e) + 0.5 * np.log(np.linalg.det(cov))
    #0.5*dimension*np.log((2*np.pi*np.e)) + 0.5*np.log(np.linalg.det(cov))
    return entropy
def list_to_multidimensional_matrix(A_matrices):
    """
    Convert a list of numpy arrays into a multidimensional matrix.

    Parameters:
    - A_matrices: A list of numpy arrays (each d x d).

    Returns:
    - A_multidimensional_matrix: A numpy array of shape (p, d, d), where p is the number of arrays,
      and d is the dimension of each array.
    """
    # Stack the list of arrays into a single 3D numpy array
    A_multidimensional_matrix = np.stack(A_matrices, axis=0)
    
    return A_multidimensional_matrix


# Plot Training and Testing MSE
def plot_MSE(dataset_sizes, train_MSE, test_MSE):
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, train_MSE, label='Training MSE', color='blue')
    plt.plot(dataset_sizes, test_MSE, label='Testing MSE', color='red')
    plt.xlabel('Dataset Size')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training vs Testing MSE over Dataset Sizes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Training and Testing Exponential Entropy
def plot_entropy(dataset_sizes, train_exp_entropy, test_exp_entropy, gt_entropy):
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, train_exp_entropy, label='Training Entropy Estimate', color='green')
    plt.plot(dataset_sizes, test_exp_entropy, label='Testing Entropy Estimate', color='orange')
    plt.axhline(y=gt_entropy, color='red', linestyle='--', label='Ground Truth Entropy')
    plt.xlabel('Dataset Size')
    plt.ylabel('Entropy')
    plt.title('Training and Testing Entropy Estimate over Dataset Sizes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Coefficient Matrices MSE for Training and Testing
def plot_coefficient_matrices_MSE(dataset_sizes, coefficient_matrices_MSE_estimates):#, coefficient_matrices_MSE_test):
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, coefficient_matrices_MSE_estimates, label='Coefficient Matrice Estimates MSE', color='purple')
    plt.xlabel('Dataset Size')
    plt.ylabel('Coefficient Matrices MSE')
    plt.title('Coefficient Matrices MSE over Dataset Sizes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot MSE vs. Exponential Entropy for Training and Testing
def plot_MSE_vs_entropy(train_MSE, train_exp_entropy, test_MSE, test_exp_entropy):
    plt.figure(figsize=(10, 6))
    
    # Plot for Training Data
    plt.scatter(np.log(train_MSE), train_exp_entropy, label='Training', color='blue', alpha=0.7)
    
    # Plot for Testing Data
    plt.scatter(test_MSE, test_exp_entropy, label='Testing', color='red', alpha=0.7)
    
    plt.xlabel('Natural Log Mean Squared Error (MSE)')
    plt.ylabel('Entropy Estimate')
    plt.title('Log MSE vs. Entropy Estimate for Training and Testing Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def linear_fit(x, y):
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the coefficients for a linear fit
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return m, b


def plot_slope_intercepts( gt_slope_intercepts, slope_intercept_estimates):
    """
    Plots the ground truth and estimated slope-intercept parameters.
    
    Parameters:
    - gt_slope_intercepts: Ground truth slope-intercept parameters (shape: [num_dims, 2])
    - slope_intercept_estimates: Estimated slope-intercept parameters (shape: [num_dims, 2])
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot for ground truth
    ax.scatter(gt_slope_intercepts[:, 1], gt_slope_intercepts[:, 0], color='blue', label='GT', marker='o')
    
    # Scatter plot for estimated values
    ax.scatter(slope_intercept_estimates[:, 1], slope_intercept_estimates[:, 0], color='red', label='Estimated', marker='x')
    
    ax.set_xlabel("Intercept")
    ax.set_ylabel("Slope")
    ax.set_title("Ground Truth vs Estimated Slope-Intercept Parameters")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()


def compare_entropy_and_MSE(noise_vs_MSE, noise_vs_entropy):
    plt.figure(figsize=(8, 6))
    
    # Plot MSE vs noise
    plt.plot(noise_vs_MSE[:, 0], noise_vs_MSE[:, 1], label='MSE', marker='o', linestyle='-')
    
    # Plot entropy vs noise
    plt.plot(noise_vs_entropy[:, 0], noise_vs_entropy[:, 1], label='Entropy', marker='s', linestyle='--')
    
    plt.xlabel("Noise Variance")
    plt.ylabel("Metric Value")
    plt.title("Noise vs MSE and Entropy")
    plt.legend()
    plt.grid(True)
    
    plt.show()


def error_bounds(noise_vs_PECEP, noise_vs_entropy=None, noise_vs_upper_bound=None):
    plt.figure(figsize=(8, 6))

    # Plot DEC vs noise
    plt.plot(noise_vs_PECEP[:, 0], noise_vs_PECEP[:, 1], label='PECEP', marker='o', linestyle='-')

    # Plot entropy vs noise if provided
    if noise_vs_entropy is not None:
        plt.plot(noise_vs_entropy[:, 0], noise_vs_entropy[:, 1], label='Entropy', marker='s', linestyle='--')

    # Plot upper bound vs noise if provided
    if noise_vs_upper_bound is not None:
        plt.plot(noise_vs_upper_bound[:, 0], noise_vs_upper_bound[:, 1], label='Upper Bound', marker='^', linestyle='-.')

    plt.xlabel("Noise Variance")
    plt.ylabel("Metric Value (nats/sec)")
    plt.title("Noise vs Error Bounds")
    plt.legend()
    plt.grid(True)
    
    plt.show()

def error_differences(noise_vs_PECEP, noise_vs_entropy=None, noise_vs_upper_bound=None):
    plt.figure(figsize=(8, 6))
    
    # Compute and plot entropy - DEC if entropy is provided
    if noise_vs_entropy is not None:
        entropy_diff = noise_vs_entropy[:, 1] - noise_vs_PECEP[:, 1]
        plt.plot(noise_vs_entropy[:, 0], entropy_diff, label='Entropy - PECEP', marker='s', linestyle='--', color='orange')
    
    # Compute and plot upper bound - DEC if upper bound is provided
    if noise_vs_upper_bound is not None:
        upper_bound_diff = noise_vs_upper_bound[:, 1] - noise_vs_PECEP[:, 1]
        plt.plot(noise_vs_upper_bound[:, 0], upper_bound_diff, label='Upper Bound - PECEP', marker='^', linestyle='-.', color='green')
    
    plt.xlabel("Noise Variance")
    plt.ylabel("Difference from PECEP")
    plt.ylim(-0.75,0.5)
    plt.title("Noise vs Difference from PECEP")
    plt.legend()
    plt.grid(True)
    
    plt.show()


'''
def analyze_noise_entropy(
    stationary_noise_variances: range=(1,101),
    dimension: int = 2,
    dataset_size: int = 10000,
    context_size: int = 8,
    noise_step_size: float = 1e-2,
    fix_random_seed = True
):
    """
    Computes entropy estimates for a range of noise variances.
    
    Parameters:
    - stationary_noise_variances (range): Range of noise variances to evaluate.
    - dimension (int): Dimensionality of the data.
    - dataset_size (int): Number of data points to generate.
    - context_size (int): Context size used for training and testing.
    
    Returns:
    - tuple of np.ndarray: Arrays containing noise vs entropy, Determinant of Error Covariance Matrix (PECEP), and upper bound values.
    """
    num_variances = len(list(stationary_noise_variances))
    noise_vs_entropy = np.zeros((num_variances, 2))
    noise_vs_PECEP = np.zeros((num_variances, 2))
    noise_vs_upper_bound = np.zeros((num_variances, 2))
    
    for ndx, noise in enumerate(tqdm(stationary_noise_variances)):
        noise = noise * noise_step_size
        noise_vs_entropy[ndx, 0] = noise
        noise_vs_PECEP[ndx, 0] = noise
        noise_vs_upper_bound[ndx, 0] = noise

        if fix_random_seed:
            reset_seeds()  # Ensure reproducibility

        success = False
        while not success:
            try:
                # Generate synthetic data
                cur_synthetic_data, _, gt_noise = generate_random_data(dimension, context_size, dataset_size, noise)
                train_size = int(dataset_size * 0.8)
                cur_train_data = cur_synthetic_data[:train_size]
                cur_test_data = cur_synthetic_data[train_size:]
                X_train, Y_train = prepare_data(cur_train_data, context_size)

                # Estimate coefficients and noise covariance
                coefficient_matrices_estimates = estimate_coefficients(X_train, Y_train, context_size)
                noise_covariance_estimate_test, test_data_estimate = estimate_noise_covariance_and_data(
                    cur_test_data, coefficient_matrices_estimates, context_size
                )

                # Compute entropy metrics
                constant = dimension * 0.5 * np.log(2*np.pi*np.e)
                cur_entropy = gauss_entropy(gt_noise, dimension)
                mat, det = error_cov_matrix_and_det(cur_test_data, test_data_estimate)
                cur_DEC = constant + 0.5 * np.log(det)
                #0.5 * np.log(det * (2 * np.pi * np.e) ** dimension)
                cur_upper = constant + 0.5 * np.trace(np.log(mat))
                #0.5 * np.trace(np.log(2 * np.pi * np.e * mat))

                # Store results
                noise_vs_entropy[ndx, 1] = cur_entropy
                noise_vs_PECEP[ndx, 1] = cur_DEC    
                noise_vs_upper_bound[ndx, 1] = cur_upper

                success = True  # Exit loop if everything succeeds

            except np.linalg.LinAlgError:
                print(f"Least-squares failure at noise level {noise}. Retrying...")
                continue
    
    return noise_vs_entropy, noise_vs_PECEP, noise_vs_upper_bound
'''

def process_single_noise_variance(noise_variance, noise_step_size, dimension, context_size, 
                                 dataset_size, fix_random_seed, test_metrics=True):
    """
    Process a single noise variance value.
    
    Parameters:
    - noise_variance: The noise variance to process
    - Other parameters: Same as main function
    
    Returns:
    - tuple: (noise_value, entropy, PECEP, upper_bound)
    """
    noise = noise_variance * noise_step_size
    
    if fix_random_seed:
        reset_seeds()  # Ensure reproducibility
    
    success = False
    while not success:
        try:
            # Generate synthetic data
            cur_synthetic_data, _, gt_noise = generate_random_data(
                dimension, context_size, dataset_size, noise
            )
            train_size = int(dataset_size * 0.8)
            cur_train_data = cur_synthetic_data[:train_size]
            cur_test_data = cur_synthetic_data[train_size:]
            X_train, Y_train = prepare_data(cur_train_data, context_size)
            
            # Estimate coefficients and noise covariance
            coefficient_matrices_estimates = estimate_coefficients(
                X_train, Y_train, context_size
            )

            noise_covariance_estimate = None
            data_estimate = None
            if test_metrics:
                # Estimate noise covariance and data estimate for testing
                noise_covariance_estimate, data_estimate = estimate_noise_covariance_and_data(
                    cur_test_data, coefficient_matrices_estimates, context_size
                )
            else:
                # For training metrics, we can use the same data
                noise_covariance_estimate, data_estimate = estimate_noise_covariance_and_data(
                    cur_train_data, coefficient_matrices_estimates, context_size
                )
            
            # Compute entropy metrics
            constant = dimension * 0.5 * np.log(2 * np.pi * np.e)
            cur_entropy = gauss_entropy(gt_noise, dimension)
            mat, det = error_cov_matrix_and_det(cur_test_data, data_estimate)
            cur_DEC = constant + 0.5 * np.log(det)
            # estimate upper bound
            #cur_upper = constant + 0.5 * np.sum(np.log(np.diag(mat)))
            cur_upper = constant + 0.5 * np.sum(np.log(np.diag(gt_noise)))
            success = True
            return noise, cur_entropy, cur_DEC, cur_upper
            
        except np.linalg.LinAlgError:
            #print(f"Least-squares failure at noise level {noise}. Retrying...")
            continue

def analyze_noise_entropy(
    stationary_noise_variances: range = range(1, 101),
    dimension: int = 2,
    dataset_size: int = 10000,
    context_size: int = 8,
    noise_step_size: float = 1e-2,
    fix_random_seed: bool = True,
    n_jobs: int = -4,
    verbose: int = 0
):
    """
    Computes entropy estimates for a range of noise variances using parallel processing.
    
    Parameters:
    - stationary_noise_variances (range): Range of noise variances to evaluate.
    - dimension (int): Dimensionality of the data.
    - dataset_size (int): Number of data points to generate.
    - context_size (int): Context size used for training and testing.
    - noise_step_size (float): Step size for noise scaling.
    - fix_random_seed (bool): Whether to fix random seed for reproducibility.
    - n_jobs (int): Number of parallel jobs. -1 uses all available cores.
    - verbose (int): Verbosity level for joblib progress.
    
    Returns:
    - tuple of np.ndarray: Arrays containing noise vs entropy, PECEP, and upper bound values.
    """
    # Convert range to list for processing
    noise_variances = list(stationary_noise_variances)
    num_variances = len(noise_variances)
    
    # Process all noise variances in parallel
    print(f"Processing {num_variances} noise variances with {n_jobs} jobs...")
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_single_noise_variance)(
            noise_var, noise_step_size, dimension, context_size, 
            dataset_size, fix_random_seed
        ) for noise_var in noise_variances
    )
    
    # Initialize result arrays
    noise_vs_entropy = np.zeros((num_variances, 2))
    noise_vs_PECEP = np.zeros((num_variances, 2))
    noise_vs_upper_bound = np.zeros((num_variances, 2))
    
    # Fill result arrays
    for i, (noise, entropy, PECEP, upper_bound) in enumerate(results):
        noise_vs_entropy[i] = [noise, entropy]
        noise_vs_PECEP[i] = [noise, PECEP]
        noise_vs_upper_bound[i] = [noise, upper_bound]
    
    return noise_vs_entropy, noise_vs_PECEP, noise_vs_upper_bound