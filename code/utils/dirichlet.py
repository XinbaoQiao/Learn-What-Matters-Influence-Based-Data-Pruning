import numpy as np
from torch.utils.data import DataLoader, Subset, RandomSampler

def create_dataloaders(dataset, n, samples_per_loader, batch_size=32, all_class_weights=None, nb_class=10):
    dataloaders = []
    for i in range(n):
        # Create a unique class distribution for each dataloader
        if all_class_weights is not None:
            class_weights = all_class_weights[i]
        else:
            class_weights = np.random.dirichlet(np.ones(nb_class))

        sampler = RandomSampler(dataset, num_samples=samples_per_loader)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        dataloaders.append(dataloader)

    return dataloaders 