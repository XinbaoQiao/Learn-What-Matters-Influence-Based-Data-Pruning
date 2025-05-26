import os
import torch
import numpy as np
from datetime import datetime
from utils.random import seed_torch
from utils.dirichlet import create_dataloaders
from utils.args import parse_args
from data.dataset_loader import load_dataset
from networks import load_model
from training.trainer import Trainer
from torch.utils.data import Subset
import copy
from utils.influence import get_sample2worker_mapping, get_influence_scores_path, compute_and_cache_influence_scores, get_pruned_indices_with_scores
from utils.utils import generate_P
from visualization.plot_utils import plot_stage_comparison

def check_indices(dataloaders1, dataloaders2, stage_name=""):
    """Check if dataset indices are consistent between two stages"""
    print(f"\nChecking {stage_name} dataset indices:")
    for i, (dl1, dl2) in enumerate(zip(dataloaders1, dataloaders2)):
        indices1 = dl1.dataset.indices
        indices2 = dl2.dataset.indices
        print(f"  Indices match: {np.array_equal(indices1, indices2)}")
        print(f"  Index length: {len(indices1)} vs {len(indices2)}")

def get_result_dirname(args):
    """Generate result directory name based on main hyperparameters"""
    dirname = f"{args.dataset_name}_{args.model}_lr{args.lr}_ep{args.epochs}_w{args.num_workers}_{args.mode}_seed{args.seed}"
    if args.nonIID:
        dirname += f"_nonIID_a{args.alpha}"
    if args.dirichlet:
        dirname += "_dirichlet"
    return dirname

def get_stage2_dirname(args):
    """Generate STAGE2 directory name with pruning ratio"""
    return f"STAGE2_{args.pruning_algorithm}_ratio{args.train_ratio}"

def create_dirs(base_dir, subdirs):
    """Create directory and its subdirectories"""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def main(args):
    # Load dataset
    train_set_original, valid_set, test_set, nb_class = load_dataset(args)

    # Generate communication matrix P, ensure consistency between stages
    P = generate_P(args.mode, args.num_workers)
    
    # Create result directory
    result_dir = os.path.join('results', get_result_dirname(args))
    os.makedirs(result_dir, exist_ok=True)
    
    # Run different stages based on stage parameter
    if args.stage == 1:
        # Create STAGE1 directory structure
        stage1_dir = os.path.join(result_dir, 'STAGE1')
        create_dirs(stage1_dir, ['models', 'metrics', 'index'])
        
        # Prepare dataloaders
        args_for_dataloaders = copy.deepcopy(args)
        args_for_dataloaders.node_datasize = len(train_set_original) // args.num_workers
        train_dataloaders = create_dataloaders(train_set_original, args.num_workers, args_for_dataloaders.node_datasize, 
                                             args.batch_size, None, nb_class)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Save sample indices for each worker
        for i, dataloader in enumerate(train_dataloaders):
            indices = dataloader.dataset.indices
            np.save(os.path.join(stage1_dir, 'index', f'worker_{i}_indices.npy'), indices)
            print(f"Saved indices for worker {i}: {len(indices)} samples")
        
        # Stage 1: Train with full dataset
        print("\nStage 1: Training with full dataset")
        seed_torch(args.seed)
        model = load_model(args.model, nb_class, weights=args.pretrained).to(args.device)
        trainer = Trainer(args, model, train_dataloaders, [valid_loader], [test_loader], stage1_dir)
        trainer.P = P
        trainer.train()
        
    elif args.stage == 2:
        # Create STAGE2 directory structure
        stage2_dir = os.path.join(result_dir, get_stage2_dirname(args))
        create_dirs(stage2_dir, ['models', 'metrics', 'index'])
        
        # Use STAGE1 from the same directory
        stage1_dir = os.path.join(result_dir, 'STAGE1')
        if not os.path.exists(stage1_dir):
            raise ValueError(f"Stage 1 directory not found: {stage1_dir}. Please run stage 1 first with the same hyperparameters.")
        
        # Load STAGE1 indices and create dataloaders
        train_dataloaders = []
        stage1_indices = {}
        for i in range(args.num_workers):
            indices_path = os.path.join(stage1_dir, 'index', f'worker_{i}_indices.npy')
            if os.path.exists(indices_path):
                indices = np.load(indices_path)
                stage1_indices[i] = indices
                train_dataloaders.append(torch.utils.data.DataLoader(
                    Subset(train_set_original, indices),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0
                ))
            else:
                raise ValueError(f"Stage 1 indices not found for worker {i}")
        
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Calculate influence scores and get nodes to keep
        print("\nCalculating influence scores...")
        influence_scores_path = get_influence_scores_path(args, stage1_dir)
        if args.pruning_algorithm == 'noprune':
            # Keep original indices without pruning
            node_keep_indices = {}
            for worker_id in range(args.num_workers):
                node_keep_indices[worker_id] = stage1_indices[worker_id]
        else:
            scores = compute_and_cache_influence_scores(args, train_set_original, valid_loader, stage1_dir, influence_scores_path)
            node_keep_indices = get_pruned_indices_with_scores(train_set_original, args, stage1_dir, scores)
        
        # Create pruned dataset for each node
        pruned_train_dataloaders = []
        for worker_id in range(args.num_workers):
            if worker_id in node_keep_indices:
                worker_indices = node_keep_indices[worker_id]
                if len(worker_indices) > 0:
                    worker_indices = np.sort(worker_indices)  # Sort indices
                    np.save(os.path.join(stage2_dir, 'index', f'worker_{worker_id}_indices.npy'), worker_indices)
                    print(f"Saved pruned indices for worker {worker_id}: {len(worker_indices)} samples (original: {len(stage1_indices[worker_id])})")
                    pruned_train_dataloaders.append(torch.utils.data.DataLoader(
                        Subset(train_set_original, worker_indices),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0
                    ))
        
        # Stage 2: Train with pruned dataset
        print("\nStage 2: Training with pruned dataset")
        seed_torch(args.seed)

        model = load_model(args.model, nb_class, weights=args.pretrained).to(args.device)
        trainer = Trainer(args, model, pruned_train_dataloaders, [valid_loader], [test_loader], stage2_dir)
        trainer.P = P
        trainer.train()

        ######################### 
        # Load metrics from both stages for comparison
        stage1_metrics_dir = os.path.join(stage1_dir, 'metrics')
        stage2_metrics_dir = os.path.join(stage2_dir, 'metrics')
        stage1_train_losses = np.load(os.path.join(stage1_metrics_dir, 'train_losses.npy'))
        stage1_test_accs = np.load(os.path.join(stage1_metrics_dir, 'test_accs.npy'))
        stage2_train_losses = np.load(os.path.join(stage2_metrics_dir, 'train_losses.npy'))
        stage2_test_accs = np.load(os.path.join(stage2_metrics_dir, 'test_accs.npy'))
        
        # Plot comparison
        plot_stage_comparison(stage1_train_losses, stage1_test_accs, 
                            stage2_train_losses, stage2_test_accs, 
                            result_dir)
        ######################### 
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed)
    main(args)
