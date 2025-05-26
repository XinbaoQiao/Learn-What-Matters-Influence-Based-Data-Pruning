import numpy as np
from torch.utils.data import DataLoader, Subset

def create_dataloaders(dataset, n, samples_per_loader, batch_size=32, all_class_weights=None, nb_class=10):
    indices = np.arange(len(dataset))
    splits = np.array_split(indices, n)
    dataloaders = []
    for split in splits:
        subset = Subset(dataset, split)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        dataloaders.append(dataloader)
    return dataloaders 