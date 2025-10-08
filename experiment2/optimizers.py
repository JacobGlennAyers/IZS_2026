import torch.optim as optim

# Optimizer lookup table
OPTIMIZER_LOOKUP = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "AdamW": optim.AdamW,
}

def get_optimizer(model, experiment_parameters):
    """Retrieves the appropriate optimizer based on experiment parameters."""
    optimizer_name = experiment_parameters["optimizer"]
    optimizer_params = experiment_parameters.get("optimizer_parameters", {})
    
    if optimizer_name not in OPTIMIZER_LOOKUP:
        raise ValueError(f"Optimizer {optimizer_name} not recognized. Available options are: {list(OPTIMIZER_LOOKUP.keys())}")
    
    optimizer_class = OPTIMIZER_LOOKUP[optimizer_name]
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    return optimizer
