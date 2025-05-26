import torch
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from networks import load_model
import concurrent.futures
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.func import functional_call, vmap, grad
import copy
import socket
import datetime
from utils.random import seed_torch
from utils.utils import generate_P
from torch.utils.data import DataLoader
from pydvl.influence.torch import EkfacInfluence

# ========== Distributed Environment Management ==========

def setup(rank, world_size):
    """Initialize distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    timeout = datetime.timedelta(seconds=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)

def cleanup():
    """Clean up distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

# ========== Model Loading and Gradient Computation ==========

def load_model_with_state_dict(model_path, args, device, nb_class):
    """Load model and its state dictionary"""
    model = load_model(args.model_A, nb_class, weights=args.pretrained)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def compute_val_grad(model, val_loader, device):
    """Compute gradients on validation set"""
    model.eval()
    data_iter = iter(val_loader)
    try:
        batch = next(data_iter)
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch
    except StopIteration:
        raise RuntimeError("Validation set is empty, cannot compute val_grad")
    
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    if not isinstance(outputs, torch.Tensor):
        outputs = outputs.logits
    loss = torch.nn.CrossEntropyLoss()(outputs, targets)
    val_grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
    return [g.detach().cpu().numpy() for g in val_grad]

def load_all_worker_models(models_dir, num_workers, epoch, model_name, nb_class, device, pretrained):
    """Load all worker models for a specific epoch"""
    worker_models = []
    for i in range(num_workers):
        model = load_model(model_name, nb_class, weights=pretrained)
        model_path = os.path.join(models_dir, f'local_model_{i}_epoch_{epoch}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        worker_models.append(model)
    return worker_models

# ========== Data Management ==========

def get_sample2worker_mapping(train_dataloaders):
    """Create mapping from sample indices to workers"""
    sample2worker = {}
    for worker_id, dataloader in enumerate(train_dataloaders):
        for batch in dataloader:
            indices = batch[2] if len(batch) > 2 else None
            if indices is None:
                raise NotImplementedError("Dataset must return index")
            for idx in indices:
                idx = int(idx)
                if idx in sample2worker:
                    print(f"Warning: Sample {idx} appears in multiple workers. Overwriting previous assignment.")
                sample2worker[idx] = worker_id
    return sample2worker

# ========== Distributed Computation Worker Process ==========

def worker_influence_process(rank, world_size, train_dataset, model, args, val_grad, lr, return_dict, sample2worker, worker_models, batch_size=128, device_id=0):
    """Worker process for computing influence values (each process bound to specific GPU)"""
    seed_torch(args.seed)  # Ensure consistent randomness across worker processes
    try:
        torch.cuda.init()
        torch.cuda.set_device(device_id)
        setup(rank, world_size)
        worker_id = rank % args.num_workers
        worker_indices = [idx for idx, w_id in sample2worker.items() if w_id == worker_id]
        if not worker_indices:
            print(f"Warning: No data for worker {worker_id} on GPU {device_id}")
            return_dict[rank] = ([], [])
            return
        subset = torch.utils.data.Subset(train_dataset, worker_indices)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        worker_model = worker_models[rank].cuda(device_id)
        worker_model.eval()
        val_grad_torch = [torch.from_numpy(g).to(f'cuda:{device_id}') for g in val_grad]
        local_scores = []
        local_indices = []
        pbar = tqdm(
            total=len(loader),
            desc=f"GPU {device_id} (Worker {worker_id})",
            position=rank,
            leave=True,
            ncols=50,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
            mininterval=1.0,
            maxinterval=5.0,
            disable=False
        )
        params = {k: v.detach() for k, v in worker_model.named_parameters()}
        buffers = {k: v.detach() for k, v in worker_model.named_buffers()}
        def compute_loss(params, buffers, sample, target):
            if sample.dim() == 3:
                sample = sample.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            predictions = functional_call(worker_model, (params, buffers), (sample,))
            if not isinstance(predictions, torch.Tensor):
                predictions = predictions.logits
            loss = torch.nn.CrossEntropyLoss()(predictions, target)
            return loss
        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        for batch in loader:
            data, targets, indices = batch
            data = data.cuda(device_id)
            targets = targets.cuda(device_id).long()
            try:
                ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
                val_grad_flat = torch.cat([g.reshape(-1) for g in val_grad_torch])
                sample_grads_flat = torch.cat([
                    g.reshape(g.shape[0], -1) for g in ft_per_sample_grads.values()
                ], dim=1)
                sample_influences = (sample_grads_flat * val_grad_flat).sum(dim=1)
                sample_influences = sample_influences * lr
                local_scores.extend(sample_influences.tolist())
                local_indices.extend(indices.tolist())
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                print(f"Data shape: {data.shape}, Targets shape: {targets.shape}")
                print(f"Data device: {data.device}, Targets device: {targets.device}")
                print(f"Targets dtype: {targets.dtype}")
                print(f"Targets: {targets}")
                print(f"Targets min: {targets.min()}, max: {targets.max()}")
                raise e
            pbar.update(1)
        pbar.close()
        return_dict[rank] = (local_indices, local_scores)
    except Exception as e:
        print(f"Error in GPU {device_id}: {str(e)}")
        return_dict[rank] = ([], [])
    finally:
        cleanup()

def gradient_dot_scores(train_dataset, device, worker_models, sample2worker, args, val_grad, lr, batch_size=128):
    """Compute influence values for each sample: lr * <val_grad, sample_grad>, using cuda:1~N only"""
    all_cuda_devices = list(range(torch.cuda.device_count()))
    if len(all_cuda_devices) > 1:
        worker_cuda_devices = all_cuda_devices[1:]  # Skip cuda:0
    else:
        worker_cuda_devices = all_cuda_devices
    world_size = len(worker_cuda_devices)
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")
    print(f"\nStarting influence computation on {world_size} GPUs (skip cuda:0 if possible)...")
    print(f"Total workers: {args.num_workers}")
    print(f"Batch size: {batch_size}")
    dummy_model = worker_models[0]
    manager = mp.Manager()
    return_dict = manager.dict()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    influence_scores = np.zeros(len(train_dataset))
    for batch_start in range(0, args.num_workers, world_size):
        batch_end = min(batch_start + world_size, args.num_workers)
        current_batch_workers = list(range(batch_start, batch_end))
        print(f"\nProcessing workers {batch_start} to {batch_end-1}...")
        current_batch_models = [worker_models[i] for i in range(batch_start, batch_end)]
        processes = []
        for local_rank, worker_id in enumerate(current_batch_workers):
            device_id = worker_cuda_devices[local_rank]
            p = mp.Process(
                target=worker_influence_process,
                args=(local_rank, len(current_batch_workers), train_dataset, dummy_model, args, val_grad, lr, return_dict, sample2worker, current_batch_models, batch_size, device_id)
            )
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print(f"\nCollecting influence scores for workers {batch_start} to {batch_end-1}...")
        for local_rank, worker_id in enumerate(current_batch_workers):
            if local_rank in return_dict:
                indices, scores = return_dict[local_rank]
                for idx, score in zip(indices, scores):
                    influence_scores[idx] = score
                print(f"GPU {worker_cuda_devices[local_rank]}: processed {len(indices)} samples")
        return_dict.clear()
        del current_batch_models
        torch.cuda.empty_cache()
    print("Influence computation completed!")
    return influence_scores

def compute_if_scores(models_dir, train_dataset, val_loader, device, model_class, batch_size=128, save_path=None, sample2worker=None, worker_models=None, args=None, nb_class=None):
    """
    Compute the influence of each training sample on all validation samples using EkfacInfluence, only for the last fully connected layer.
    Returns a numpy array of shape (num_train,) with the average influence of each training sample on the validation set.
    """
    from torch.utils.data import DataLoader
    # Load last epoch model
    model_state_dicts = sorted([f for f in os.listdir(models_dir) if f.startswith('global_model_epoch_') and f.endswith('.pth')],
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
    model_path = os.path.join(models_dir, model_state_dicts[-1])
    model = load_model_with_state_dict(model_path, args, device, nb_class)
    model = model.to(device)

    # Freeze all layers except the last fully connected layer
    last_linear_found = False
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'head' in name or 'linear' in name:
            param.requires_grad = True
            last_linear_found = True
        else:
            param.requires_grad = False
    if not last_linear_found:
        raise RuntimeError("No fully connected (fc/classifier/head/linear) layer found in model for influence computation.")

    # Prepare data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_x, val_y = [], []
    for batch in val_loader:
        x, y = batch[:2]
        val_x.append(x)
        val_y.append(y)
    val_x = torch.cat(val_x, dim=0).to(device)
    val_y = torch.cat(val_y, dim=0).to(device)

    train_x, train_y = [], []
    for batch in train_loader:
        x, y = batch[:2]
        train_x.append(x)
        train_y.append(y)
    train_x = torch.cat(train_x, dim=0).to(device)
    train_y = torch.cat(train_y, dim=0).to(device)

    # Compute influences: shape (num_val, num_train)
    if_model = EkfacInfluence(model, update_diagonal=True, hessian_regularization=0.0)
    if_model.fit(train_loader)
    influences = if_model.influences(val_x, val_y, train_x, train_y)  # shape: (num_val, num_train)
    influences = influences.cpu().numpy()

    # Average over all validation samples to get a single score per training sample
    avg_influence = influences.mean(axis=0)  # shape: (num_train,)

    if save_path is not None:
        np.save(save_path, avg_influence)
        print(f"IF scores saved to {save_path}")
    return avg_influence

def compute_dice_scores(models_dir, train_dataset, val_loader, device, model_class, batch_size=128, save_path=None, sample2worker=None, worker_models=None, args=None, nb_class=None):
    """Compute DICE scores (first + second term) for each sample, as in the paper's Proposition 1."""
    P = generate_P(args.mode, args.num_workers)
    N = args.num_workers
    epoch_t = args.epochs - 2
    epoch_t1 = args.epochs - 1
    worker_models_t = load_all_worker_models(models_dir, N, epoch_t, args.model_A, nb_class, device, args.pretrained)
    worker_models_t1 = load_all_worker_models(models_dir, N, epoch_t1, args.model_A, nb_class, device, args.pretrained)
    lr_list = np.load(os.path.join(models_dir, 'lr_list.npy'))
    lr = lr_list[epoch_t]

    dice_first_term = np.zeros(len(train_dataset))
    dice_second_term = np.zeros(len(train_dataset))

    # Precompute validation gradients for each node at round t
    val_grads_t = []
    for model in worker_models_t:
        val_grad = compute_val_grad(model, val_loader, device)
        val_grads_t.append([torch.from_numpy(g).to(device) for g in val_grad])

    for j in range(len(train_dataset)):
        node_j = sample2worker[j]
        model_j_t = worker_models_t[node_j]
        model_j_t.eval()
        x_j, y_j = train_dataset[j][0], train_dataset[j][1]
        if hasattr(x_j, 'unsqueeze'):
            x_j = x_j.unsqueeze(0).to(device)
        else:
            x_j = torch.tensor(x_j).unsqueeze(0).to(device)
        y_j = torch.tensor([y_j]).to(device)
        out = model_j_t(x_j)
        if not isinstance(out, torch.Tensor):
            out = out.logits
        loss = torch.nn.CrossEntropyLoss()(out, y_j)
        grad_j = torch.autograd.grad(loss, model_j_t.parameters(), retain_graph=False)
        grad_j = [g.detach() for g in grad_j]

        # first term
        val_grad = val_grads_t[node_j]
        dot1 = sum([(g1 * g2).sum() for g1, g2 in zip(val_grad, grad_j)])
        qj = 1.0 / N
        dice_first_term[j] = -lr * qj * dot1.item()

        # second term
        sum_k = 0.0
        for k in range(N):
            model_k_t1 = worker_models_t1[k]
            model_k_t1.eval()
            grad_k = compute_val_grad(model_k_t1, val_loader, device)
            grad_k = [torch.from_numpy(g).to(device) for g in grad_k]
            dot2 = sum([(gk * gj).sum() for gk, gj in zip(grad_k, grad_j)])
            qk = 1.0 / N
            weight = qk * (P[k, node_j].item() if isinstance(P, torch.Tensor) else P[k, node_j])
            sum_k += weight * dot2.item()
        dice_second_term[j] = -lr * sum_k

    dice_scores = dice_first_term + dice_second_term
    if save_path is not None:
        np.save(save_path, dice_scores)
        print(f"DICE scores saved to {save_path}")
    return dice_scores

def compute_delete_scores(models_dir, train_dataset, val_loader, device, model_class, batch_size=64, save_path=None, sample2worker=None, worker_models_list=None, args=None, nb_class=None):
    """Compute Delete scores (accumulate gradient dot products across all epochs)
    For each epoch:
    1. Compute gradients for all samples in local dataset for each local model
    2. Compute average gradient on validation set for global model
    3. Dot product these gradients and multiply by learning rate
    """
    model_state_dicts = sorted([f for f in os.listdir(models_dir) if f.startswith('global_model_epoch_') and f.endswith('.pth')],
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
    S = len(train_dataset)
    delete_scores = np.zeros(S)
    lr_list = np.load(os.path.join(models_dir, 'lr_list.npy'))
    
    if sample2worker is None:
        from utils.dirichlet import create_dataloaders
        from data.dataset_loader import IndexDataset
        if not isinstance(train_dataset, IndexDataset):
            train_dataset = IndexDataset(train_dataset)
        num_workers = args.num_workers
        node_datasize = len(train_dataset) // num_workers
        batch_size = args.batch_size
        train_dataloaders = create_dataloaders(train_dataset, num_workers, node_datasize, batch_size, None, nb_class)
        sample2worker = get_sample2worker_mapping(train_dataloaders)

    # for t in range(0, len(model_state_dicts), len(model_state_dicts) // 5):
    for t in range(len(model_state_dicts)):
        print(f"\nComputing delete scores for epoch {t}")
        
        model_path = os.path.join(models_dir, model_state_dicts[t])
        global_model = load_model_with_state_dict(model_path, args, device, nb_class)
        val_grad = compute_val_grad(global_model, val_loader, device)
        if val_grad is None:
            raise ValueError(f"Failed to compute validation gradients for epoch {t}")
        
        worker_models = []
        for i in range(args.num_workers):
            worker_model = load_model(args.model_A, nb_class, weights=args.pretrained)
            worker_model_path = os.path.join(models_dir, f'local_model_{i}_epoch_{t}.pth')
            worker_model.load_state_dict(torch.load(worker_model_path, map_location=device))
            worker_model = worker_model.to(device)
            worker_models.append(worker_model)
        
        lr = lr_list[t]
        scores = gradient_dot_scores(train_dataset, device, worker_models, sample2worker, args, val_grad, lr, batch_size)
        delete_scores += scores
        
        del global_model
        for model in worker_models:
            del model
        del worker_models
        del val_grad
        torch.cuda.empty_cache()
    
    if save_path is not None:
        np.save(save_path, delete_scores)
        print(f"Delete scores saved to {save_path}")
    return delete_scores

# ========== Pruning Interface ==========

def get_influence_scores_path(args, stage1_dir):
    """Generate the cache path for influence scores based on key hyperparameters."""
    fname = f"{args.pruning_algorithm}_w{args.num_workers}_seed{args.seed}.npy"
    cache_dir = os.path.join(stage1_dir, 'influence_scores')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, fname)

def compute_and_cache_influence_scores(args, train_set, valid_loader, stage1_dir, cache_path):
    """Compute and cache influence scores if not already cached"""
    if os.path.exists(cache_path):
        print(f"Influence scores already cached at {cache_path}")
        return np.load(cache_path)
    print(f"Computing influence scores and saving to {cache_path}")
    models_dir = os.path.join(stage1_dir, 'models')
    index_dir = os.path.join(stage1_dir, 'index')
    nb_class = len(train_set.dataset.classes) if hasattr(train_set.dataset, 'classes') else 10
    num_workers = args.num_workers
    worker_indices = {}
    for worker_id in range(num_workers):
        indices_path = os.path.join(index_dir, f'worker_{worker_id}_indices.npy')
        if os.path.exists(indices_path):
            worker_indices[worker_id] = np.load(indices_path)
        else:
            raise ValueError(f"Worker {worker_id} indices not found in {index_dir}")
    sample2worker = {}
    for worker_id, indices in worker_indices.items():
        for idx in indices:
            sample2worker[int(idx)] = worker_id
    epoch = args.epochs - 1
    worker_models = load_all_worker_models(models_dir, num_workers, epoch, args.model_A, nb_class, args.device, args.pretrained)
    if args.pruning_algorithm == 'if':
        scores = compute_if_scores(models_dir, train_set, valid_loader, args.device, None, batch_size=128, sample2worker=sample2worker, worker_models=worker_models, args=args, nb_class=nb_class)
    elif args.pruning_algorithm == 'dice':
        scores = compute_dice_scores(models_dir, train_set, valid_loader, args.device, None, batch_size=128, sample2worker=sample2worker, worker_models=worker_models, args=args, nb_class=nb_class)
    elif args.pruning_algorithm == 'delete':
        scores = compute_delete_scores(models_dir, train_set, valid_loader, args.device, None, batch_size=64, sample2worker=sample2worker, args=args, nb_class=nb_class)
    elif args.pruning_algorithm == 'random':
        scores = np.random.rand(len(train_set))
    elif args.pruning_algorithm == 'noprune':
        scores = None
    else:
        raise ValueError(f"Unknown pruning algorithm: {args.pruning_algorithm}")
    if scores is not None:
        np.save(cache_path, scores)
        print(f"Influence scores saved to {cache_path}")
    return scores

def get_pruned_indices_with_scores(train_set, args, stage1_dir, scores):
    """Prune based on scores and ratio, return indices to keep for each worker"""
    index_dir = os.path.join(stage1_dir, 'index')
    num_workers = args.num_workers
    worker_indices = {}
    for worker_id in range(num_workers):
        indices_path = os.path.join(index_dir, f'worker_{worker_id}_indices.npy')
        if os.path.exists(indices_path):
            worker_indices[worker_id] = np.load(indices_path)
        else:
            raise ValueError(f"Worker {worker_id} indices not found in {index_dir}")
    node_keep_indices = {}
    for worker_id in range(num_workers):
        indices = worker_indices[worker_id]
        if len(indices) == 0:
            print(f"Warning: Worker {worker_id} has no samples!")
            node_keep_indices[worker_id] = []
            continue
        worker_scores = scores[indices]
        num_to_keep = int(len(indices) * args.train_ratio)
        sorted_indices = np.argsort(worker_scores)
        keep_set = set(indices[sorted_indices[:num_to_keep]])
        node_keep_indices[worker_id] = np.array([idx for idx in indices if idx in keep_set])
        print(f"Worker {worker_id}: keeping {len(node_keep_indices[worker_id])} out of {len(indices)} samples")
    return node_keep_indices

# You can add other influence-related utility functions here if needed 
