import torch
import torch.nn as nn
from pydvl.influence.torch import EkfacInfluence
from typing import List, Optional, Tuple, Union
import numpy as np

def compute_influence(
    dataloader: torch.utils.data.DataLoader,
    pruning_ratio: float
) -> torch.utils.data.DataLoader:
    """
    Compute influence scores using EKFAC method from pydvl and return pruned dataset.
    
    Args:
        dataloader: DataLoader containing training data
        pruning_ratio: Ratio of samples to remove based on influence scores
        
    Returns:
        DataLoader containing the pruned dataset
    """
    # Get model from the first batch
    first_batch = next(iter(dataloader))
    if isinstance(first_batch, (tuple, list)):
        inputs = first_batch[0]
    else:
        inputs = first_batch
    
    # Create a simple model for influence computation
    input_size = inputs.view(inputs.size(0), -1).size(1)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 10)  # Assuming 10 classes, adjust if needed
    )
    
    # Initialize EKFAC influence calculator with default parameters
    if_model = EkfacInfluence(
        model,
        update_diagonal=True,  # Use EKFAC
        hessian_regularization=0.0,
        device=inputs.device
    )
    
    # Fit the influence calculator
    if_model.fit(dataloader)
    
    # Use the same dataloader for test data
    test_loader = dataloader
    
    # Compute influence scores
    influence_scores = if_model.compute_influence(
        inputs,
        first_batch[1] if isinstance(first_batch, (tuple, list)) else torch.zeros(inputs.size(0)),
        test_loader
    )
    
    # Convert influence scores to numpy array
    influence_scores = np.array(influence_scores)
    
    # Get indices of samples to keep (remove samples with most negative influence)
    n_samples = len(influence_scores)
    n_remove = int(n_samples * pruning_ratio)
    keep_indices = np.argsort(influence_scores)[n_remove:]
    
    # Create new dataset with kept samples
    dataset = dataloader.dataset
    if hasattr(dataset, 'data'):
        # For datasets like CIFAR, MNIST etc.
        if isinstance(dataset.data, torch.Tensor):
            dataset.data = dataset.data[keep_indices]
        else:
            dataset.data = np.array(dataset.data)[keep_indices]
        if hasattr(dataset, 'targets'):
            if isinstance(dataset.targets, torch.Tensor):
                dataset.targets = dataset.targets[keep_indices]
            else:
                dataset.targets = np.array(dataset.targets)[keep_indices]
    else:
        # For custom datasets
        dataset = torch.utils.data.Subset(dataset, keep_indices)
    
    # Create new dataloader with pruned dataset
    pruned_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        shuffle=dataloader.shuffle,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )
    
    return pruned_loader
