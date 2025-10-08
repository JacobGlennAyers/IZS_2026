
import torch
from torch import Tensor
from torch.nn import functional as F

class SpectralConvergenceLoss(torch.nn.Module):
    """
    Spectral Convergence Loss Function
    Implements: SC = ||S_pred - S_true||_F / ||S_true||_F
    
    Commonly used for spectrogram reconstruction tasks as it normalizes
    the Frobenius norm difference by the target magnitude.
    """
    __constants__ = ["reduction", "eps", "norm_dim"]
    
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-7,
        norm_dim: int = -1  # Which dimension(s) to compute norms over
    ) -> None:
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.reduction = reduction
        self.eps = eps
        self.norm_dim = norm_dim
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute spectral convergence loss.
        
        Args:
            input: Predicted spectrogram [batch_size, ..., freq_bins] or any shape
            target: Ground truth spectrogram [batch_size, ..., freq_bins]
            
        Returns:
            Spectral convergence loss based on reduction parameter
        """
        if input.shape != target.shape:
            raise ValueError(f"Input and target shapes must match: {input.shape} vs {target.shape}")
        
        # Compute difference
        diff = input - target
        
        # Compute norms
        if self.norm_dim == -1:
            # Default: norm over last dimension (frequency bins)
            numerator = torch.norm(diff, dim=-1)  # [batch_size, ...]
            denominator = torch.norm(target, dim=-1) + self.eps  # [batch_size, ...]
        else:
            # Custom dimension(s) for norm computation
            numerator = torch.norm(diff, dim=self.norm_dim)
            denominator = torch.norm(target, dim=self.norm_dim) + self.eps
        
        # Compute spectral convergence
        sc_loss = numerator / denominator
        
        # Apply reduction
        if self.reduction == 'none':
            return sc_loss
        elif self.reduction == 'sum':
            return sc_loss.sum()
        else:  # 'mean'
            return sc_loss.mean()

class PECEPLoss(torch.nn.Module):
    """
    Prediction Error Conditional Entropy Proxy (PECEP) Loss Function
    
    Implements equation
    PECEP = (d/2) * ln(2πe) + (1/2) * ln(|Σ̂_ε|)
    
    This can be used either as a loss function or as a complexity measure.
    """
    
    __constants__ = ["reduction", "regularization", "return_components"]
    
    def __init__(
        self, 
        reduction: str = "mean",
        regularization: float = 1e-9,
        return_components: bool = False
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.regularization = regularization
        self.return_components = return_components
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Compute PECEP from prediction residuals.
        
        Args:
            input: Predicted values [batch_size, ..., feature_dim] or [batch_size, feature_dim]
            target: Ground truth values [batch_size, ..., feature_dim] or [batch_size, feature_dim]
            
        Returns:
            PECEP value(s) based on reduction parameter
        """
        # Compute residuals (prediction errors)
        residuals = input - target  # Shape: [batch_size, ..., feature_dim]
        
        # Flatten all dimensions except the last (feature dimension)
        original_shape = residuals.shape
        if len(original_shape) > 2:
            residuals = residuals.view(-1, original_shape[-1])
        
        batch_size, d = residuals.shape
        
        if batch_size < 2:
            # Can't compute covariance with less than 2 samples
            # Return a default high entropy value
            return torch.tensor(1000.0, device=residuals.device, dtype=residuals.dtype)
        
        # Compute sample covariance matrix
        # Σ̂_ε = (1/(N-1)) * Σ(residual_i * residual_i^T)
        residuals_centered = residuals - torch.mean(residuals, dim=0, keepdim=True)
        cov_matrix = torch.mm(residuals_centered.T, residuals_centered) / (batch_size - 1)
        
        # Add regularization for numerical stability
        cov_matrix += self.regularization * torch.eye(d, device=residuals.device, dtype=residuals.dtype)
        
        # Compute log determinant using Cholesky decomposition for stability
        try:
            L = torch.linalg.cholesky(cov_matrix)
            log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
        except RuntimeError:
            # If Cholesky fails, use eigenvalue decomposition as fallback
            eigenvals = torch.linalg.eigvals(cov_matrix).real
            eigenvals = torch.clamp(eigenvals, min=self.regularization)
            log_det = torch.sum(torch.log(eigenvals))
        
        # Compute PECEP according to equation (5)
        entropy_constant = (d / 2.0) * torch.log(torch.tensor(2 * torch.pi * torch.e, device=residuals.device))
        pecep = entropy_constant + 0.5 * log_det
        
        if self.return_components:
            # Also compute Hadamard bound for analysis
            diagonal_elements = torch.diag(cov_matrix)
            hadamard_bound = entropy_constant + 0.5 * torch.sum(torch.log(diagonal_elements))
            gaussianizing_criterion = hadamard_bound - pecep
            return pecep, hadamard_bound, gaussianizing_criterion
        
        return pecep

# Loss function lookup table
CRITERION_LOOKUP = {
    "MSE": torch.nn.MSELoss,
    "CrossEntropy": torch.nn.CrossEntropyLoss,
    "PECEP": PECEPLoss,  # Example custom loss
    "Huber": torch.nn.HuberLoss,
    "SpectralConvergence": SpectralConvergenceLoss
}

def get_loss(experiment_parameters):
    """Retrieves the appropriate loss function based on experiment parameters."""
    loss_function_name = experiment_parameters["criterion"]
    loss_function_params = experiment_parameters.get("criterion_parameters", {})
    
    if loss_function_name not in CRITERION_LOOKUP:
        raise ValueError(f"Loss function {loss_function_name} not recognized. Available options are: {list(CRITERION_LOOKUP.keys())}")

    loss_function_class = CRITERION_LOOKUP[loss_function_name]
    criterion = loss_function_class(**loss_function_params)
    
    return criterion