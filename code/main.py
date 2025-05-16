import os
import torch
import numpy as np
from datetime import datetime
from utils.random import set_seed
from utils.dirichlet import create_dataloaders
from utils.args import parse_args
from data.dataset_loader import load_dataset
from networks import load_model
from training.trainer import Trainer

def create_result_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def main(args):
    # Create result directory
    result_dir = create_result_dir()
    
    # Print CUDA information
    print("\nCUDA Device Information:")
    if torch.cuda.is_available():
        device_idx = 0 if not ':' in args.device else int(args.device.split(':')[1])
        print(f"Using CUDA Device: {torch.cuda.get_device_name(device_idx)}")
        print(f"Device Index: {args.device}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.\n")
    
    set_seed(args.seed)
    
    # Load dataset
    train_set, valid_set, test_set, nb_class = load_dataset(args)
    
    args.node_datasize = len(train_set) // args.size
    # Create dataloaders
    train_dataloaders = create_dataloaders(train_set, args.size, args.node_datasize, 
                                         args.batch_size, None, nb_class)
    valid_dataloaders = create_dataloaders(valid_set, args.size, args.node_datasize, 
                                         args.batch_size, None, nb_class)
    test_dataloaders = create_dataloaders(test_set, args.size, args.node_datasize, 
                                         args.batch_size, None, nb_class)
    
    # Load model
    model = load_model(args.model, nb_class, weights=args.pretrained)
    
    # Initialize trainer and start training
    trainer = Trainer(args, model, train_dataloaders, valid_dataloaders, test_dataloaders, result_dir)
    trainer.train()
    
    # Load saved metrics
    metrics_dir = os.path.join(result_dir, 'metrics')
    history = {
        'train_losses': np.load(os.path.join(metrics_dir, 'train_losses.npy')),
        'train_accs': np.load(os.path.join(metrics_dir, 'train_accs.npy')),
        'valid_losses': np.load(os.path.join(metrics_dir, 'valid_losses.npy')),
        'valid_accs': np.load(os.path.join(metrics_dir, 'valid_accs.npy')),
        'test_losses': np.load(os.path.join(metrics_dir, 'test_losses.npy')),
        'test_accs': np.load(os.path.join(metrics_dir, 'test_accs.npy')),
        'args': vars(args)
    }
    torch.save(history, os.path.join(result_dir, 'training_history.pth'))

if __name__ == "__main__":
    args = parse_args()
    main(args)
