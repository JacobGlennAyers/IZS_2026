import torch
import torch.optim.lr_scheduler as lr_scheduler

# Scheduler lookup table
SCHEDULER_LOOKUP = {
    "StepLR": lr_scheduler.StepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealing": lr_scheduler.CosineAnnealingLR,
    "MultiStepLR": lr_scheduler.MultiStepLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,
}

def get_scheduler(optimizer, experiment_parameters):
    """
    Retrieves the appropriate scheduler based on experiment parameters.
    
    Args:
        optimizer (Optimizer): The optimizer for which the scheduler will adjust the learning rate.
        experiment_parameters (dict): A dictionary containing the scheduler configuration.
        
    Returns:
        Scheduler: The initialized learning rate scheduler.
    """
    experiment_parameters["scheduler_parameters"]["T_max"] = experiment_parameters.get("epoch_count", 1)
    scheduler_name = experiment_parameters["scheduler"]
    scheduler_params = experiment_parameters.get("scheduler_parameters", {})
    
    if scheduler_name not in SCHEDULER_LOOKUP:
        raise ValueError(f"Scheduler {scheduler_name} not recognized. Available options are: {list(SCHEDULER_LOOKUP.keys())}")
    
    scheduler_class = SCHEDULER_LOOKUP[scheduler_name]
    scheduler = scheduler_class(optimizer, **scheduler_params)
    
    return scheduler
