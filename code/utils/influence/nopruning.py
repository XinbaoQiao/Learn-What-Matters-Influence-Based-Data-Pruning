import torch

def nopruning(dataloader, pruning_ratio):
    """
    No pruning function. Returns the original dataloader without any modifications.
    
    Args:
        dataloader: The original dataloader.
        pruning_ratio: The pruning ratio (ignored in this function).
    
    Returns:
        The original dataloader.
    """
    return dataloader 