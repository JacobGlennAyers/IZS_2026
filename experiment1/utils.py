import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
from collections import defaultdict
from itertools import product
import pickle

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from threadpoolctl import threadpool_limits

...

def estimate_coefficients_regularized(X, Y, p, method='ridge', alpha=0.1):
    """
    Regularized coefficient estimation with automatic hyperparameter selection
    """
    N, d = X.shape
    
    if method == 'ridge':
        reg = Ridge(alpha=alpha)
    elif method == 'lasso':
        reg = Lasso(alpha=alpha, max_iter=300)
    elif method == 'elastic_net':
        reg = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=300)

    # 💡 Limit BLAS threads only during model fitting
    with threadpool_limits(limits=1):
        reg.fit(Y, X)

    A_flat = reg.coef_.T
    A_matrices = [A_flat[i * d: (i + 1) * d, :] for i in range(p)]
    actual_alpha = getattr(reg, 'alpha_', reg.alpha)
    return A_matrices, actual_alpha


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
    diff_cov = np.cov(np.transpose(difference_matrix), bias=False)
    #print(diff_cov.shape)
    return diff_cov, np.linalg.det(diff_cov)
def visualize_data(data):
    plt.figure(figsize=(12, 3))
    plt.imshow(data.T, aspect='auto', origin='lower', cmap='grey')
    #plt.colorbar(label='Intensity')
    plt.title('Synthetic Spectrogram')
    plt.xlabel('"Time Step"')
    plt.ylabel('"Feature Dimension"')
    plt.tight_layout()
    plt.show()
    plt.close()




def visualize_MSE(gt_data, predicted_data, title):
    gt_data = gt_data.T
    predicted_data = predicted_data.T
    MSE_plot = np.mean((gt_data - predicted_data) ** 2, axis=0)

    # Determine common vmin and vmax for consistent color scale
    vmin = min(gt_data.min(), predicted_data.min(), (gt_data - predicted_data).min())
    vmax = max(gt_data.max(), predicted_data.max(), (gt_data - predicted_data).max())

    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.4})

    # Add color bar space
    cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.02])

    # Plot ground truth spectrogram
    img1 = axs[0].imshow(gt_data, aspect='auto', origin='lower', cmap='grey', vmin=vmin, vmax=vmax)
    axs[0].set_ylabel('Feature Dimension')
    axs[0].set_title('Ground Truth Time-Series')

    # Plot predicted spectrogram
    axs[1].imshow(predicted_data, aspect='auto', origin='lower', cmap='grey', vmin=vmin, vmax=vmax)
    axs[1].set_ylabel('Feature Dimension')
    axs[1].set_title('Predicted Time-Series')

    # Plot the Difference
    axs[2].imshow(gt_data - predicted_data, aspect='auto', origin='lower', cmap='grey', vmin=vmin, vmax=vmax)
    axs[2].set_ylabel('Feature Dimension')
    axs[2].set_title('Difference (Ground Truth - Predicted)')

    # Add color bar to align with the first three plots
    fig.colorbar(img1, cax=cbar_ax, orientation='horizontal')

    # Plot MSE
    axs[3].plot(np.arange(MSE_plot.shape[0]), MSE_plot, color='red', label='MSE')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('MSE')
    axs[3].tick_params(axis='y')
    axs[3].legend(loc='upper right')

    # Add title and save the figure
    fig.suptitle(title)
    output_path = os.path.join(f"{title.replace(' ', '_')}_plot.png")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
    plt.close()


def histogram_norm(matrix):
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(matrix)
    
    # Normalize the singular values into a histogram
    S_normalized = S / np.sum(S)#S[0]
    
    # Reconstruct the normalized matrix
    normalized_matrix = U @ np.diag(S_normalized) @ Vt
    
    return normalized_matrix


def softmax_norm(matrix, temperature=1.0):
    """
    Normalize a matrix using softmax applied to its singular values.
    
    Parameters:
    - matrix: Input matrix to normalize
    - temperature: Temperature parameter for softmax (lower = more peaked distribution)
    
    Returns:
    - normalized_matrix: Matrix reconstructed with softmax-normalized singular values
    """
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Apply softmax to singular values to create a probability distribution
    # Softmax formula: exp(x_i / T) / sum(exp(x_j / T))
    S_softmax = np.exp(S / temperature) / np.sum(np.exp(S / temperature))
    
    # Reconstruct the normalized matrix
    normalized_matrix = U @ np.diag(S_softmax) @ Vt
    
    return normalized_matrix


def spectral_norm(matrix, target_radius=0.9):
    """
    Normalize so the spectral radius (largest absolute eigenvalue) equals target.
    Critical for autoregressive stability.
    
    Parameters:
    - matrix: Input matrix
    - target_radius: Desired spectral radius (< 1 for stability)
    
    Returns:
    - Matrix scaled to have specified spectral radius
    """
    eigenvals = np.linalg.eigvals(matrix)
    current_radius = np.max(np.abs(eigenvals))
    
    if current_radius > 0:
        scale_factor = target_radius / current_radius
        return matrix * scale_factor
    return matrix
    

def fro_norm(matrix, target_norm=1.0):
    """
    Normalize by Frobenius norm - preserves all structure, just scales magnitude.
    
    Parameters:
    - matrix: Input matrix
    - target_norm: Desired Frobenius norm
    
    Returns:
    - Normalized matrix with specified Frobenius norm
    """
    current_norm = np.linalg.norm(matrix, 'fro')
    if current_norm > 0:
        return matrix * (target_norm / current_norm)
    return matrix

def normalize_matrix(matrix, method='fro'):
    """
    Normalize a matrix using the specified method.
    
    Parameters:
    - matrix: Input matrix to normalize
    - method: Normalization method ('fro' for Frobenius norm, 'spectral' for spectral norm)
    
    Returns:
    - normalized_matrix: Normalized matrix
    """
    if method == 'fro':
        return fro_norm(matrix)
    elif method == 'spectral':
        return spectral_norm(matrix)
    elif method == 'softmax':
        return softmax_norm(matrix)
    elif method == 'histogram':
        return histogram_norm(matrix)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

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
    N, d = data.shape  # N: number of time points, d: dimensionality of each vectora

    # Create the design matrix Y
    Y = np.zeros((N - p, p * d))  # (N - p) x (p * d)
    X = data[p:]  # Target values, excluding the first p time steps
    
    # Populate the design matrix Y
    for t in range(p, N):
        # Stack the past p time steps as one row of the design matrix Y
        Y[t - p] = np.hstack([data[t - i - 1] for i in range(p)])
    return X, Y

def estimate_coefficients(X, Y, p):
    N, d = X.shape
    
    # Solve for coefficients - this gives us A^T
    A_flat_transposed = np.linalg.lstsq(Y, X, rcond=None)[0]  # (p * d) x d matrix
    
    # Transpose to get the correct orientation
    A_flat = A_flat_transposed.T  # Now d x (p * d)
    
    # Reshape into coefficient matrices
    A_matrices = [A_flat[:, i * d: (i + 1) * d] for i in range(p)]
    
    return A_matrices


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

def estimate_coefficients_regularized(X, Y, p, method='ridge', alpha=0.1):
    """
    Regularized coefficient estimation with automatic hyperparameter selection
    """
    N, d = X.shape
    
    if method == 'ridge':
        # Ridge regression - handles multicollinearity well
        reg = Ridge(alpha=alpha)
    elif method == 'lasso':
        # LASSO - automatic lag selection
        reg = Lasso(alpha=alpha, max_iter=300)
    elif method == 'elastic_net':
        # Best of both worlds
        reg = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=300)
    
    # Fit the model
    with threadpool_limits(limits=1):
        reg.fit(Y, X)
    A_flat = reg.coef_.T
    
    # Reshape back to coefficient matrices
    A_matrices = [A_flat[i * d: (i + 1) * d, :] for i in range(p)]
    # Return the regularization parameter used
    # For Ridge and Lasso, use the alpha parameter that was set
    actual_alpha = getattr(reg, 'alpha_', reg.alpha)
    return A_matrices, actual_alpha

# Cross-validation for hyperparameter selection
def select_optimal_alpha(X, Y, p, method='ridge', alphas=np.logspace(-4, 2, 50)):
    """
    Use cross-validation to select optimal regularization parameter
    """
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alphas:
        if method == 'ridge':
            reg = Ridge(alpha=alpha)
        elif method == 'lasso':
            reg = Lasso(alpha=alpha, max_iter=300)
        elif method == 'elastic_net':
            reg = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=300)
        
        # Cross-validation score
        with threadpool_limits(limits=1):
            scores = cross_val_score(reg, Y, X, cv=5, scoring='r2', n_jobs=1)
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    return best_alpha

from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def estimate_coefficients_mle(X, Y, p):
    """
    Maximum likelihood estimation for VAR parameters
    """
    N, d = X.shape
    
    def negative_log_likelihood(params):
        # Unpack parameters
        A_flat = params[:p*d*d].reshape(p, d, d)
        sigma_flat = params[p*d*d:]
        
        # Ensure positive definite covariance
        L = np.zeros((d, d))
        idx = 0
        for i in range(d):
            for j in range(i+1):
                L[i, j] = sigma_flat[idx]
                idx += 1
        Sigma = L @ L.T + 1e-6 * np.eye(d)  # Add small regularization
        
        # Compute log-likelihood
        ll = 0
        for t in range(N):
            y_t = Y[t]  # Lagged values
            x_t = X[t]  # Current observation
            
            # Predicted value
            x_pred = sum(A_flat[i] @ y_t[i*d:(i+1)*d] for i in range(p))
            residual = x_t - x_pred
            
            # Add to log-likelihood
            ll += multivariate_normal.logpdf(residual, mean=np.zeros(d), cov=Sigma)
        
        return -ll
    
    # Initial guess (use OLS estimates)
    A_init = np.linalg.lstsq(Y, X, rcond=None)[0]
    residuals = X - Y @ A_init
    Sigma_init = np.cov(residuals.T, bias=False)
    
    # Flatten parameters
    L_init = np.linalg.cholesky(Sigma_init)
    sigma_init = []
    for i in range(d):
        for j in range(i+1):
            sigma_init.append(L_init[i, j])
    
    params_init = np.concatenate([A_init.flatten(), sigma_init])
    
    # Optimize
    result = minimize(negative_log_likelihood, params_init, method='L-BFGS-B')
    
    # Extract results
    A_flat = result.x[:p*d*d].reshape(p, d, d)
    A_matrices = [A_flat[i] for i in range(p)]
    
    return A_matrices

def estimate_coefficients_bootstrap(X, Y, p, n_bootstrap=500):
    """
    Bootstrap bias correction for AR coefficient estimation
    """
    N, d = X.shape
    
    # Original estimates
    A_original = np.linalg.lstsq(Y, X, rcond=None)[0]
    A_matrices_original = [A_original[i * d: (i + 1) * d, :] for i in range(p)]
    
    # Bootstrap samples
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(N, size=N, replace=True)
        X_boot = X[indices]
        Y_boot = Y[indices]
        
        # Estimate coefficients
        A_boot = np.linalg.lstsq(Y_boot, X_boot, rcond=None)[0]
        A_matrices_boot = [A_boot[i * d: (i + 1) * d, :] for i in range(p)]
        bootstrap_estimates.append(A_matrices_boot)
    
    # Bias correction
    A_matrices_corrected = []
    for i in range(p):
        # Compute bias
        bootstrap_mean = np.mean([est[i] for est in bootstrap_estimates], axis=0)
        bias = bootstrap_mean - A_matrices_original[i]
        
        # Bias-corrected estimate
        A_corrected = A_matrices_original[i] - bias
        A_matrices_corrected.append(A_corrected)
    
    return A_matrices_corrected

def make_stationary_projection(A_matrices):
    """
    Project coefficient matrices onto stationary region
    """
    p = len(A_matrices)
    d = A_matrices[0].shape[0]
    
    # Construct companion matrix
    companion = np.zeros((p*d, p*d))
    for i in range(p):
        companion[:d, i*d:(i+1)*d] = A_matrices[i]
    
    # Add identity blocks for lagged terms
    for i in range(1, p):
        companion[i*d:(i+1)*d, (i-1)*d:i*d] = np.eye(d)
    
    # Check eigenvalues
    eigenvals = np.linalg.eigvals(companion)
    max_eigenval = np.max(np.abs(eigenvals))
    
    if max_eigenval >= 1.0:
        # Scale to make stationary
        scale_factor = 0.95 / max_eigenval
        A_matrices = [A * scale_factor for A in A_matrices]
    
    return A_matrices

def estimate_coefficients_stationary(X, Y, p, method='ridge', alpha=0.1):
    """
    Estimate coefficients with stationarity constraint
    """
    # Get initial estimates
    A_matrices, _ = estimate_coefficients_regularized(X, Y, p, method, alpha)
    
    # Project onto stationary region
    A_matrices = make_stationary_projection(A_matrices)
    
    return A_matrices



# ============================================================================
# NON-PARAMETRIC PREDICTION METHODS
# ============================================================================

def fit_kernel_ridge(X, Y):
    """Fit Kernel Ridge Regression model"""
    kr = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.1)
    kr.fit(Y, X)
    return kr

def fit_gaussian_process(X, Y):
    """Fit Gaussian Process models (one per output dimension)"""
    d = X.shape[1]
    gps = []
    for i in range(d):
        kernel = RBF(length_scale=1.0) * Matern(length_scale=1.0, nu=1.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
        gp.fit(Y, X[:, i])
        gps.append(gp)
    return gps

class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def fit_neural_network(X, Y, epochs=500):
    """Fit Neural Network model"""
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    
    model = TimeSeriesPredictor(Y.shape[1], X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(Y_tensor)
        loss = criterion(predictions, X_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"NN Epoch {epoch}, Loss: {loss.item():.6f}")
    
    model.eval()
    return model

def fit_random_forest(X, Y):
    """Fit Random Forest model"""
    rf = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    ))
    rf.fit(Y, X)
    return rf

def fit_gradient_boosting(X, Y):
    """Fit Gradient Boosting model"""
    gb = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ))
    gb.fit(Y, X)
    return gb

def fit_svr(X, Y):
    """Fit Support Vector Regression model"""
    svr = MultiOutputRegressor(SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        epsilon=0.01
    ))
    svr.fit(Y, X)
    return svr

# ============================================================================
# PREDICTION FUNCTIONS FOR NON-PARAMETRIC METHODS
# ============================================================================

def predict_kernel_ridge(model, context):
    """Predict using Kernel Ridge model"""
    return model.predict(context.reshape(1, -1))[0]

def predict_gaussian_process(models, context):
    """Predict using Gaussian Process models"""
    d = len(models)
    x_t_pred = np.zeros(d)
    pred_var = np.zeros(d)
    
    context_reshaped = context.reshape(1, -1)
    for i, gp in enumerate(models):
        pred_mean, pred_std = gp.predict(context_reshaped, return_std=True)
        x_t_pred[i] = pred_mean[0]
        pred_var[i] = pred_std[0]**2
    
    return x_t_pred, pred_var

def predict_neural_network(model, context):
    """Predict using Neural Network model"""
    with torch.no_grad():
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        prediction = model(context_tensor).numpy()[0]
    return prediction

def predict_sklearn_model(model, context):
    """Predict using any sklearn model (RF, GB, SVR)"""
    return model.predict(context.reshape(1, -1))[0]


def estimate_noise_and_data(data, p, train_percent=0.8, solve_method='least_squares', gt_coefficients=None):
    """
    Optimized version that only runs inference on test data to save computation.
    """
    # Define method categories
    parametric_solvers = {'oracle','least_squares', 'ridge', 'lasso', 'mle', 'bootstrap', 'stationary_ridge'}
    nonparametric_solvers = {'kernel_ridge', 'gaussian_process', 'neural_network', 
                           'random_forest', 'gradient_boosting', 'svr'}
    
    # Split data into training and test sets
    train_size = int(len(data) * train_percent)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    N, d = data.shape
    
    # ========================================================================
    # PARAMETRIC METHODS
    # ========================================================================
    if solve_method in parametric_solvers:
        # Prepare training data
        X_train, Y_train = prepare_data(train_data, p)
        
        # Choose estimation method
        if solve_method == 'least_squares':
            A_estimates = estimate_coefficients(X_train, Y_train, p)
        elif solve_method == 'ridge':
            optimal_alpha = select_optimal_alpha(X_train, Y_train, p, 'ridge')
            A_estimates, _ = estimate_coefficients_regularized(X_train, Y_train, p, 'ridge', optimal_alpha)
        elif solve_method == 'lasso':
            optimal_alpha = select_optimal_alpha(X_train, Y_train, p, 'lasso')
            A_estimates, _ = estimate_coefficients_regularized(X_train, Y_train, p, 'lasso', optimal_alpha)
        elif solve_method == 'mle':
            A_estimates = estimate_coefficients_mle(X_train, Y_train, p)
        elif solve_method == 'bootstrap':
            A_estimates = estimate_coefficients_bootstrap(X_train, Y_train, p)
        elif solve_method == 'stationary_ridge':
            A_estimates = estimate_coefficients_stationary(X_train, Y_train, p, 'ridge')
        elif solve_method == 'oracle':
            if gt_coefficients is not None:
                A_estimates = gt_coefficients
        
        # Generate predictions ONLY on TEST data
        test_predictions = []
        test_residuals = []
        
        # Only run inference on test portion
        for t in range(train_size, N):
            x_t_pred = np.zeros(d)
            for i in range(p):
                prior_ndx = t - (i+1)
                if prior_ndx >= 0:
                    x_t_pred += A_estimates[i] @ data[prior_ndx]
            
            test_predictions.append(x_t_pred)
            epsilon_t = data[t] - x_t_pred
            test_residuals.append(epsilon_t)
        
        # Compute covariance from test residuals
        if test_residuals:
            print("Computing covariance from test residuals...")
            test_residuals = np.array(test_residuals)
            noise_covariance = np.cov(test_residuals.T, bias=False)
            flattened_residuals = test_residuals.flatten()
            variance_value = np.var(flattened_residuals)
            #print("variance of flattened residuals:", np.var(flattened_residuals))
            #print(noise_covariance)
            # Append to CSV
            csv_filename = 'variance_results.csv'  # Change to your desired filename

            # Write header if file doesn't exist, then append variance
            try:
                # Try to read existing file to check if it exists
                with open(csv_filename, 'r') as f:
                    pass
                # File exists, just append
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([variance_value])
                    
            except FileNotFoundError:
                # File doesn't exist, create with header
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['variance'])  # Header
                    writer.writerow([variance_value])  # First data row
            
            # Ensure positive definite
            if np.linalg.det(noise_covariance) <= 0:
                print("Warning: Non-positive definite covariance matrix estimated. Adding regularization.")
                noise_covariance = np.eye(d) * 1e-6
        else:
            noise_covariance = np.eye(d) * 1e-6
        
        # Convert test predictions to array for consistency
        test_predictions = np.array(test_predictions) if test_predictions else np.empty((0, d))
        
        return noise_covariance, test_predictions#, A_estimates
    
    # ========================================================================
    # NON-PARAMETRIC METHODS
    # ========================================================================
    elif solve_method in nonparametric_solvers:
        # Prepare training data
        X_train, Y_train = prepare_data(train_data, p)
        
        # Fit the non-parametric model on training data
        if solve_method == 'kernel_ridge':
            model = fit_kernel_ridge(X_train, Y_train)
        elif solve_method == 'gaussian_process':
            model = fit_gaussian_process(X_train, Y_train)
        elif solve_method == 'neural_network':
            model = fit_neural_network(X_train, Y_train)
        elif solve_method == 'random_forest':
            model = fit_random_forest(X_train, Y_train)
        elif solve_method == 'gradient_boosting':
            model = fit_gradient_boosting(X_train, Y_train)
        elif solve_method == 'svr':
            model = fit_svr(X_train, Y_train)
        
        # Generate predictions ONLY on TEST data
        test_predictions = []
        test_residuals = []
        prediction_variances = []  # For GP uncertainty
        
        # Only run inference on test portion
        for t in range(train_size, N):
            # Construct context vector
            context = np.hstack([data[t-i-1] for i in range(p)])
            
            # Make prediction based on method
            if solve_method == 'kernel_ridge':
                x_t_pred = predict_kernel_ridge(model, context)
            elif solve_method == 'gaussian_process':
                x_t_pred, pred_var = predict_gaussian_process(model, context)
                prediction_variances.append(pred_var)
            elif solve_method == 'neural_network':
                x_t_pred = predict_neural_network(model, context)
            elif solve_method in ['random_forest', 'gradient_boosting', 'svr']:
                x_t_pred = predict_sklearn_model(model, context)
            
            test_predictions.append(x_t_pred)
            epsilon_t = data[t] - x_t_pred
            test_residuals.append(epsilon_t)
        
        # Compute covariance from test residuals
        if test_residuals:
            test_residuals = np.array(test_residuals)
            noise_covariance = np.cov(test_residuals.T, bias=True)
            
            # Ensure positive definite
            if np.linalg.det(noise_covariance) <= 0:
                noise_covariance = np.eye(d) * 1e-6
        else:
            noise_covariance = np.eye(d) * 1e-6
        
        # Convert test predictions to array for consistency
        test_predictions = np.array(test_predictions) if test_predictions else np.empty((0, d))
        
        # For Gaussian Process, also return prediction uncertainty
        if solve_method == 'gaussian_process' and len(prediction_variances) > 0:
            prediction_variances = np.array(prediction_variances)
            uncertainty_cov = np.diag(np.mean(prediction_variances, axis=0))
            return noise_covariance, test_predictions#, model, uncertainty_cov
        else:
            return noise_covariance, test_predictions#, model
    
    else:
        raise ValueError(f"Unknown solve_method: {solve_method}. "
                        f"Available methods: {parametric_solvers | nonparametric_solvers}")

def process_dataset_size_multi_solver(dataset_size, dimension, context_length, stationary_noise_variance,
                                     gt_coefficient_matrices, gt_covariance_matrix,
                                     matrix_normalization_method='fro', decay=True, rate=0.85,
                                     solvers=['least_squares', 'ridge', 'lasso', 'mle', 'bootstrap', 'stationary_ridge']):
    """
    Process a single dataset size with multiple solvers
    """
    reset_seeds()
    
    # Data generation and preparation
    cur_synthetic_data, _, _ = generate_random_data(dimension, context_length, dataset_size, stationary_noise_variance,
                                                   matrix_normalization_method=matrix_normalization_method, 
                                                   decay=decay, rate=rate)
    
    # Split into train/test
    train_size = int(dataset_size * 0.8)
    cur_test_data = cur_synthetic_data[train_size:]
    
    results = {
        'dataset_size': dataset_size,
        'solvers': {}
    }
    
    # Test each solver
    for solver in solvers:
        try:
            print("Running: ", solver, "for dataset size:", dataset_size)
            # Estimate using current solver
            noise_covariance_estimate_test, test_data_estimate, A_estimates = estimate_noise_and_data(
                cur_synthetic_data, context_length, train_percent=0.8, solve_method=solver)
            
            # Calculate metrics
            test_mse = MSE(cur_test_data, test_data_estimate[train_size:])
            test_entropy_val = gauss_entropy(noise_covariance_estimate_test, dimension)
            
            # Coefficient MSE
            coefficient_matrix = list_to_multidimensional_matrix(gt_coefficient_matrices)
            coefficient_matrices_mse_test = MSE(coefficient_matrix, A_estimates)
            
            results['solvers'][solver] = {
                'test_MSE': test_mse,
                'test_entropy': test_entropy_val,
                'coefficient_matrices_MSE_test': coefficient_matrices_mse_test
            }
            
        except Exception as e:
            print(f"Solver {solver} failed for dataset size {dataset_size}: {e}")
            results['solvers'][solver] = {
                'test_MSE': np.inf,
                'test_entropy': np.inf,
                'coefficient_matrices_MSE_test': np.inf
            }
    
    return results

# Main execution function
def run_multi_solver_comparison(dimension, context_length, stationary_noise_variance):
    """
    Run comparison of multiple solvers
    """
    
    # Generate ground truth
    reset_seeds()
    _, gt_coefficient_matrices, gt_covariance_matrix = generate_random_data(
        dimension, context_length, 100, stationary_noise_variance)
    gt_entropy = gauss_entropy(gt_covariance_matrix, dimension)
    
    # Dataset sizes
    dataset_sizes = list(range(int(2e3), int(2e4)+1, int(2e3))) 
    
    # Solvers to test
    solvers = ['least_squares', 'ridge', 'lasso', 'mle', 'bootstrap', 'stationary_ridge']
    #solvers = ['least_squares', 'ridge', 'lasso', 'stationary_ridge']#, 'mle']
    print(f"Processing {len(dataset_sizes)} dataset sizes with {len(solvers)} solvers in parallel...")
    
    # Parallel processing
    results = Parallel(n_jobs=15, verbose=1)(
        delayed(process_dataset_size_multi_solver)(
            dataset_size, dimension, context_length, stationary_noise_variance,
            gt_coefficient_matrices, gt_covariance_matrix,
            solvers=solvers
        ) for dataset_size in tqdm(dataset_sizes)
    )
    print("All dataset sizes processed.")
    # Organize results
    solver_results = {solver: {
        'test_MSE': [],
        'test_entropy': [],
        'coefficient_matrices_MSE_test': []
    } for solver in solvers}
    
    for result in results:
        for solver in solvers:
            if solver in result['solvers']:
                solver_results[solver]['test_MSE'].append(result['solvers'][solver]['test_MSE'])
                solver_results[solver]['test_entropy'].append(result['solvers'][solver]['test_entropy'])
                solver_results[solver]['coefficient_matrices_MSE_test'].append(result['solvers'][solver]['coefficient_matrices_MSE_test'])
            else:
                solver_results[solver]['test_MSE'].append(np.inf)
                solver_results[solver]['test_entropy'].append(np.inf)
                solver_results[solver]['coefficient_matrices_MSE_test'].append(np.inf)
    
    # Plot results
    plot_multi_solver_comparison(dataset_sizes, solver_results, gt_entropy)
    
    return dataset_sizes, solver_results, gt_entropy

def plot_multi_solver_comparison(dataset_sizes, solver_results, gt_entropy):
    """
    Plot comparison of different solvers
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different solvers
    colors = {
        'least_squares': 'red',
        'ridge': 'blue', 
        'lasso': 'green',
        'mle': 'purple',
        'bootstrap': 'orange',
        'stationary_ridge': 'brown'
    }
    
    # Plot 1: Test MSE
    ax1 = axes[0, 0]
    for solver, color in colors.items():
        if solver in solver_results:
            ax1.plot(dataset_sizes, solver_results[solver]['test_MSE'], 
                    label=solver.replace('_', ' ').title(), color=color, linewidth=2)
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Test MSE vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Test Entropy
    ax2 = axes[0, 1]
    for solver, color in colors.items():
        if solver in solver_results:
            ax2.plot(dataset_sizes, solver_results[solver]['test_entropy'], 
                    label=solver.replace('_', ' ').title(), color=color, linewidth=2)
    ax2.axhline(y=gt_entropy, color='black', linestyle='--', linewidth=2, label='Ground Truth Entropy')
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Test Entropy')
    ax2.set_title('Test Entropy vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient MSE
    ax3 = axes[1, 0]
    for solver, color in colors.items():
        if solver in solver_results:
            ax3.plot(dataset_sizes, solver_results[solver]['coefficient_matrices_MSE_test'], 
                    label=solver.replace('_', ' ').title(), color=color, linewidth=2)
    ax3.set_xlabel('Dataset Size')
    ax3.set_ylabel('Coefficient MSE')
    ax3.set_title('Coefficient Matrix MSE vs Dataset Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Entropy Error (Distance from Ground Truth)
    ax4 = axes[1, 1]
    for solver, color in colors.items():
        if solver in solver_results:
            entropy_error = [abs(ent - gt_entropy) for ent in solver_results[solver]['test_entropy']]
            ax4.plot(dataset_sizes, entropy_error, 
                    label=solver.replace('_', ' ').title(), color=color, linewidth=2)
    ax4.set_xlabel('Dataset Size')
    ax4.set_ylabel('|Test Entropy - Ground Truth|')
    ax4.set_title('Entropy Error vs Dataset Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


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

def generate_random_data( dimension, context_size, dataset_size, stationary_variance=1, matrix_normalization_method='fro', decay=True, rate=0.5):
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
        cur_coefficient_matrix = normalize_matrix(generate_center_band_matrix(dimension, context_size), method=matrix_normalization_method) 
        if decay:
            cur_coefficient_matrix *= (rate ** (ndx+1))
        #* (1/(ndx+1))
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

def gauss_entropy(cov, dimension):
    entropy = dimension * 0.5 * np.log(2*np.pi*np.e) + 0.5 * np.log(np.linalg.det(cov))
    #0.5*dimension*np.log((2*np.pi*np.e)) + 0.5*np.log(np.linalg.det(cov))
    return entropy

def hadamard_upper_bound(cov, dimension):
    constant = dimension * 0.5 * np.log(2 * np.pi * np.e)
    upper = constant + 0.5 * np.sum(np.log(np.diag(cov)))
    return upper


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
    #plt.ylim(-0.75,0.5)
    plt.title("Noise vs Difference from PECEP")
    plt.legend()
    plt.grid(True)
    
    plt.show()



    noise = noise_variance * noise_step_size
    
    if fix_random_seed:
        reset_seeds()  # Ensure reproducibility
    
    success = False
    while not success:
        try:
            # Generate synthetic data
            cur_synthetic_data, _, gt_noise = generate_random_data(
                dimension, context_size, dataset_size, noise, matrix_normalization_method=matrix_normalization_method,
                    decay=decay, rate=rate
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
            mat = None
            det = None
            if test_metrics:
                mat, det = error_cov_matrix_and_det(cur_test_data, data_estimate)
            else:
                mat, det = error_cov_matrix_and_det(cur_train_data, data_estimate)
            cur_PECEP = constant + 0.5 * np.log(det)
            # estimate upper bound
            cur_upper = constant + 0.5 * np.sum(np.log(np.diag(mat)))
            #cur_upper = constant + 0.5 * np.sum(np.log(np.diag(gt_noise)))
            success = True
            return noise, cur_entropy, cur_PECEP, cur_upper
            
        except np.linalg.LinAlgError:
            #print(f"Least-squares failure at noise level {noise}. Retrying...")
            continue

def analyze_noise_entropy(
    stationary_noise_variances: range = range(1, 101),
    dimension: int = 2,
    dataset_size: int = 10000,
    context_size: int = 8,
    noise_step_size: float = 1e-2,
    matrix_normalization_method: str = 'fro',
    decay: bool  = True,
    rate: float = 0.85,
    fix_random_seed: bool = True,
    test_metrics: bool = True,
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
            dataset_size, fix_random_seed,
            matrix_normalization_method=matrix_normalization_method,
            decay=decay, rate=rate, test_metrics=test_metrics
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


def gauss_entropy(covariance_matrix, dimension):
    """
    Calculate the differential entropy of a multivariate Gaussian distribution.
    """
    try:
        det_cov = np.linalg.det(covariance_matrix)
        if det_cov <= 0:
            return np.inf
        return 0.5 * dimension * np.log(2 * np.pi * np.e) + 0.5 * np.log(det_cov)
    except:
        return np.inf

def run_comprehensive_experiment(experiment_config):
    """
    Run a comprehensive experiment across multiple parameters and generate visualization.
    
    Parameters:
    experiment_config (dict): Dictionary containing experiment parameters
        - dataset_sizes (list): List of dataset sizes to test
        - noise_variances (list): List of additive gaussian noise variances
        - dimension (int): Dimensionality of the data
        - context_length (int): Context length for autoregressive model
        - matrix_normalization_methods (list): List of normalization methods
        - decay_rates (list): List of decay rates
        - solvers (list): List of solvers to compare
        - use_decay (bool): Whether to use decay in data generation
        - num_trials (int): Number of trials to average over (default: 1)
        - plot_config (dict): Optional plot configuration
    
    Returns:
    dict: Results dictionary with all experimental data
    """
    
    # Extract parameters
    dataset_sizes = experiment_config['dataset_sizes']
    noise_variances = experiment_config['noise_variances']
    dimension = experiment_config['dimension']
    context_length = experiment_config['context_length']
    matrix_normalization_methods = experiment_config.get('matrix_normalization_methods', ['fro'])
    decay_rates = experiment_config.get('decay_rates', [0.85])
    solvers = experiment_config['solvers']
    use_decay = experiment_config.get('use_decay', True)
    num_trials = experiment_config.get('num_trials', 1)
    plot_config = experiment_config.get('plot_config', {})
    
    # Initialize results storage - restructured to properly handle all combinations
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # Calculate theoretical bounds (ground truth noise differential entropies)
    theoretical_bounds = {}
    for noise_var in noise_variances:
        gt_covariance = np.diag(np.ones(dimension) * noise_var)
        theoretical_bounds[noise_var] = gauss_entropy(gt_covariance, dimension)
    
    # Generate all parameter combinations
    param_combinations = list(product(
        dataset_sizes, noise_variances, matrix_normalization_methods, decay_rates
    ))
    
    print(f"Running {len(param_combinations)} parameter combinations with {num_trials} trials each")
    print(f"Total experiments: {len(param_combinations) * num_trials * len(solvers)}")
    
    # Run experiments
    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        
        
        for dataset_size, noise_var, norm_method, decay_rate in tqdm(param_combinations, 
                                                                    desc=f"Trial {trial + 1}"):
            
            # Generate ground truth data for this configuration
            try:
                # Note: You'll need to implement reset_seeds() and generate_random_data()
                # For now, I'll assume they exist
                  # Ensure reproducibility
                #reset_seeds(trial)
                # Generate synthetic data ONCE per configuration
                synthetic_data, gt_coefficient_matrices, gt_covariance_matrix = generate_random_data(
                    dimension=dimension,
                    context_size=context_length,
                    dataset_size=dataset_size,
                    stationary_variance=noise_var,
                    matrix_normalization_method=norm_method,
                    decay=use_decay,
                    rate=decay_rate
                )
                
                # Test each solver on the SAME data
                solver_results = process_dataset_size_multi_solver_shared_data(
                    synthetic_data=synthetic_data,
                    dataset_size=dataset_size,
                    dimension=dimension,
                    context_length=context_length,
                    stationary_noise_variance=noise_var,
                    gt_coefficient_matrices=gt_coefficient_matrices,
                    gt_covariance_matrix=gt_covariance_matrix,
                    solvers=solvers
                )
                
                # Store results with proper indexing
                for solver in solvers:
                    if solver in solver_results['solvers']:
                        results[noise_var][norm_method][decay_rate][solver].append({
                            'dataset_size': dataset_size,
                            'test_entropy': solver_results['solvers'][solver]['test_entropy'],
                            'test_MSE': solver_results['solvers'][solver]['test_MSE'],
                            'coefficient_matrices_MSE_test': solver_results['solvers'][solver]['coefficient_matrices_MSE_test']
                        })
                    else:
                        results[noise_var][norm_method][decay_rate][solver].append({
                            'dataset_size': dataset_size,
                            'test_entropy': np.inf,
                            'test_MSE': np.inf,
                            'coefficient_matrices_MSE_test': np.inf
                        })
                        
            except Exception as e:
                print(f"Error in experiment: {e}")
                # Store failed results
                for solver in solvers:
                    results[noise_var][norm_method][decay_rate][solver].append({
                        'dataset_size': dataset_size,
                        'test_entropy': np.inf,
                        'test_MSE': np.inf,
                        'coefficient_matrices_MSE_test': np.inf
                    })
    
    # Process results (average over trials and organize by dataset size)
    processed_results = {}
    
    for noise_var in results:
        for norm_method in results[noise_var]:
            for decay_rate in results[noise_var][norm_method]:
                config_key = f"noise_{noise_var}_norm_{norm_method}_decay_{decay_rate}"
                processed_results[config_key] = {
                    'noise_var': noise_var,
                    'norm_method': norm_method,
                    'decay_rate': decay_rate,
                    'solvers': {}
                }
                
                for solver in results[noise_var][norm_method][decay_rate]:
                    solver_data = results[noise_var][norm_method][decay_rate][solver]
                    
                    # Group by dataset size
                    size_grouped = defaultdict(list)
                    for result in solver_data:
                        size_grouped[result['dataset_size']].append(result)
                    
                    # Average over trials for each dataset size
                    processed_results[config_key]['solvers'][solver] = {}
                    for dataset_size in size_grouped:
                        values = [r['test_entropy'] for r in size_grouped[dataset_size]]
                        mse_values = [r['test_MSE'] for r in size_grouped[dataset_size]]
                        coeff_mse_values = [r['coefficient_matrices_MSE_test'] for r in size_grouped[dataset_size]]
                        
                        # Remove infinite values for averaging
                        finite_values = [v for v in values if np.isfinite(v)]
                        finite_mse_values = [v for v in mse_values if np.isfinite(v)]
                        finite_coeff_mse_values = [v for v in coeff_mse_values if np.isfinite(v)]
                        
                        if finite_values:
                            processed_results[config_key]['solvers'][solver][dataset_size] = {
                                'entropy_mean': np.mean(finite_values),
                                'entropy_std': np.std(finite_values) if len(finite_values) > 1 else 0,
                                'mse_mean': np.mean(finite_mse_values) if finite_mse_values else np.inf,
                                'mse_std': np.std(finite_mse_values) if len(finite_mse_values) > 1 else 0,
                                'coeff_mse_mean': np.mean(finite_coeff_mse_values) if finite_coeff_mse_values else np.inf,
                                'coeff_mse_std': np.std(finite_coeff_mse_values) if len(finite_coeff_mse_values) > 1 else 0,
                                'success_rate': len(finite_values) / len(values)
                            }
                        else:
                            processed_results[config_key]['solvers'][solver][dataset_size] = {
                                'entropy_mean': np.inf,
                                'entropy_std': 0,
                                'mse_mean': np.inf,
                                'mse_std': 0,
                                'coeff_mse_mean': np.inf,
                                'coeff_mse_std': 0,
                                'success_rate': 0
                            }
    
    # Generate visualization
    fig = create_comprehensive_plot(
        processed_results, 
        theoretical_bounds, 
        experiment_config,
        plot_config
    )
    
    return {
        'results': processed_results,
        'theoretical_bounds': theoretical_bounds,
        'experiment_config': experiment_config,
        'figure': fig
    }



def load_pickled_data(filepath):
    """Load pickled data from file"""
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def analyze_oracle_residuals(data_dir, dataset_size=1000, context_size=8, dimension=32):
    """
    Analyze residual errors for oracle experiment at a specific dataset size.
    Returns what percentage of residuals fall within true standard deviation at each variance level.
    
    Parameters:
    - data_dir: Directory containing the variance subdirectories
    - dataset_size: Size of dataset to analyze (default: 1000)
    - context_size: Context size for AR model
    - dimension: Dimension of the data
    
    Returns:
    - results_dict: Dictionary with variance levels as keys and percentage within true std as values
    """
    variances = os.listdir(data_dir)
    results_dict = {}
    
    print(f"Analyzing oracle residuals for dataset size {dataset_size}")
    
    for variance_dir in tqdm(variances, desc="Processing variances"):
        variance_path = os.path.join(data_dir, variance_dir)
        if not os.path.isdir(variance_path):
            continue
            
        # Extract variance value from directory name
        try:
            variance_value = float(variance_dir.replace('variance_', ''))
        except:
            print(f"Could not parse variance from {variance_dir}")
            continue
            
        trial_files = [f for f in os.listdir(variance_path) if f.endswith('.pkl')]
        
        all_residuals = []
        
        # Process each trial
        for trial_file in trial_files:
            dataset_path = os.path.join(variance_path, trial_file)
            
            try:
                # Load the data
                full_data, coefficients, noise_cov_mat = load_pickled_data(dataset_path)
                data = full_data[:dataset_size]
                
                # Get oracle predictions (using ground truth coefficients)
                estimated_cov, test_predictions, residuals = estimate_noise_and_data_with_residuals(
                    data, context_size, solve_method='oracle', gt_coefficients=coefficients
                )
                
                # Store residuals from this trial
                all_residuals.extend(residuals)
                
            except Exception as e:
                print(f"Error processing {dataset_path}: {e}")
                continue
        
        if all_residuals:
            # Convert to numpy array
            all_residuals = np.array(all_residuals)
            
            # Calculate true standard deviation (from the known noise variance)
            true_std = np.sqrt(variance_value)
            
            # For each dimension, check what percentage falls within ±1 true std
            percentages_within_std = []
            
            for dim in range(dimension):
                residuals_dim = all_residuals[:, dim]
                within_std = np.abs(residuals_dim) <= true_std
                percentage = np.mean(within_std) * 100
                percentages_within_std.append(percentage)
            
            # Average across dimensions
            avg_percentage = np.mean(percentages_within_std)
            results_dict[variance_value] = {
                'avg_percentage_within_std': avg_percentage,
                'per_dimension_percentages': percentages_within_std,
                'total_residuals': len(all_residuals),
                'true_std': true_std
            }
            
            print(f"Variance {variance_value}: {avg_percentage:.2f}% within true std")
    
    return results_dict

def estimate_noise_and_data_with_residuals(data, p, train_percent=0.8, solve_method='oracle', gt_coefficients=None):
    """
    Modified version of estimate_noise_and_data that also returns residuals.
    """
    # Split data into training and test sets
    train_size = int(len(data) * train_percent)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    N, d = data.shape
    
    # For oracle method, use ground truth coefficients
    if solve_method == 'oracle' and gt_coefficients is not None:
        A_estimates = gt_coefficients
    else:
        # Prepare training data for other methods
        X_train, Y_train = prepare_data(train_data, p)
        A_estimates = estimate_coefficients(X_train, Y_train, p)
    
    # Generate predictions and collect residuals for TEST data only
    test_predictions = []
    test_residuals = []
    
    for t in range(train_size, N):
        x_t_pred = np.zeros(d)
        for i in range(p):
            prior_ndx = t - (i+1)
            if prior_ndx >= 0:
                x_t_pred += A_estimates[i] @ data[prior_ndx]
        
        test_predictions.append(x_t_pred)
        epsilon_t = data[t] - x_t_pred
        test_residuals.append(epsilon_t)
    
    # Compute covariance from test residuals
    if test_residuals:
        test_residuals = np.array(test_residuals)
        noise_covariance = np.cov(test_residuals.T, bias=False)
        
        # Ensure positive definite
        if np.linalg.det(noise_covariance) <= 0:
            noise_covariance = np.eye(d) * 1e-6
    else:
        noise_covariance = np.eye(d) * 1e-6
        test_residuals = np.empty((0, d))
    
    test_predictions = np.array(test_predictions) if test_predictions else np.empty((0, d))
    
    return noise_covariance, test_predictions, test_residuals