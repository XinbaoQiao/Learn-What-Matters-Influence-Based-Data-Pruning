import torch
import numpy as np
import os
import datetime
import socket

def generate_P(mode, size):
    """
    Generate communication matrix P based on the mode.
    
    Args:
        mode: Communication mode ('dqn_chooseone', 'dsgd', etc.)
        size: Number of workers
        
    Returns:
        P: Communication matrix of shape (size, size)
    """
    P = torch.zeros((size, size))
    if mode == "all":
        P = torch.ones((size, size)) / size
    elif mode == "single":
        for i in range(size):
            P[i][i] = 1
    elif mode == "ring":
        for i in range(size):
            P[i][i] = 1 / 3
            P[i][(i - 1 + size) % size] = 1 / 3
            P[i][(i + 1) % size] = 1 / 3
    elif mode == "right":
        for i in range(size):
            P[i][i] = 1 / 2
            P[i][(i + 1) % size] = 1 / 2
    elif mode == "star":
        for i in range(size):
            P[i][i] = 1 - 1 / size
            P[0][i] = 1 / size
            P[i][0] = 1 / size
    elif mode == "meshgrid":
        assert size > 0
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        # print(shape, flush=True)
        topo = np.zeros((size, size))
        for i in range(size):
            topo[i][i] = 1.0
            if (i + 1) % ncol != 0:
                topo[i][i + 1] = 1.0
                topo[i + 1][i] = 1.0
            if i + ncol < size:
                topo[i][i + ncol] = 1.0
                topo[i + ncol][i] = 1.0
        topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
        for i in range(size):
            for j in topo_neighbor_with_self[i]:
                if i != j:
                    topo[i][j] = 1.0 / max(
                        len(topo_neighbor_with_self[i]), len(topo_neighbor_with_self[j])
                    )
            topo[i][i] = 2.0 - topo[i].sum()
        P = torch.tensor(topo, dtype=torch.float)
    elif mode == "exponential":
        x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
        x /= x.sum()
        topo = np.empty((size, size))
        for i in range(size):
            topo[i] = np.roll(x, i)
        P = torch.tensor(topo, dtype=torch.float)
    # print(P, flush=True)
    elif mode == "random":
        P = None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return P.clone().detach().float().requires_grad_(True)

def update_dsgd(worker_list, P, args, train_loaders):
    """
    Update workers using DSGD algorithm.
    
    Args:
        worker_list: List of workers
        P: Communication matrix
        args: Arguments
        train_loaders: List of train dataloaders
    """
    for worker, train_loader in zip(worker_list, train_loaders):
        worker.step(train_loader)
        worker.update_grad()

def merge_without_update(worker_list, P, args, valid_loader):
    """
    Merge models without updating gradients.
    
    Args:
        worker_list: List of workers
        P: Communication matrix
        args: Arguments
        valid_loader: Validation dataloader
    """
    # Store original parameters
    original_params = []
    for worker in worker_list:
        original_params.append(worker.model.state_dict())
    
    # Merge models
    for i, worker in enumerate(worker_list):
        merged_params = {}
        for param_name in original_params[0].keys():
            # Initialize with zeros of the same type as the original parameter
            merged_params[param_name] = torch.zeros_like(original_params[0][param_name])
            for j in range(len(worker_list)):
                # Ensure P[i,j] is converted to the same type as the parameter
                weight = P[i, j].to(original_params[j][param_name].dtype)
                merged_params[param_name] += weight * original_params[j][param_name]
        worker.model.load_state_dict(merged_params)

def merge_model(worker_list, P):
    """
    Merge model parameters using communication matrix P.
    
    Args:
        worker_list: List of workers
        P: Communication matrix
    """
    # Store original parameters
    original_params = []
    for worker in worker_list:
        original_params.append(worker.model.state_dict())
    
    # Merge models
    for i, worker in enumerate(worker_list):
        merged_params = {}
        for param_name in original_params[0].keys():
            # Initialize with zeros of the same type as the original parameter
            merged_params[param_name] = torch.zeros_like(original_params[0][param_name])
            for j in range(len(worker_list)):
                # Ensure P[i,j] is converted to the same type as the parameter
                weight = P[i, j].to(original_params[j][param_name].dtype)
                merged_params[param_name] += weight * original_params[j][param_name]
        worker.model.load_state_dict(merged_params)
