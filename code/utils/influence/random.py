import torch
import random

def random_pruning(dataloader, pruning_ratio):
    """
    Randomly removes a portion of data from the dataloader based on the pruning ratio.
    
    Args:
        dataloader: The original dataloader.
        pruning_ratio: The ratio of data to be removed (0 to 1).
    
    Returns:
        A new dataloader with randomly pruned data.
    """
    if pruning_ratio <= 0:
        return dataloader
    
    # Convert dataloader to a list of batches
    batches = list(dataloader)
    
    # Calculate the number of batches to remove
    num_batches_to_remove = int(len(batches) * pruning_ratio)
    
    # Randomly select batches to remove
    batches_to_remove = random.sample(batches, num_batches_to_remove)
    
    # Remove the selected batches
    for batch in batches_to_remove:
        batches.remove(batch)
    
    # Create a new dataloader with the remaining batches
    new_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(*[torch.cat([b[i] for b in batches]) for i in range(len(batches[0]))]),
        batch_size=dataloader.batch_size,
        shuffle=dataloader.shuffle
    )
    
    return new_dataloader 