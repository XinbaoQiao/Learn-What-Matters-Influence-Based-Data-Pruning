import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Decentralized Learning Project')
    
    # Dataset Parameters
    parser.add_argument('--dataset_path', type=str, default="datasets", 
                      help="Path to the dataset directory")
    parser.add_argument('--dataset_name', type=str, default="cifar100", 
                      help="Dataset name: svhn, cifar10, cifar100")
    parser.add_argument('--image_size', type=int, default=32, 
                      help="Input image size")
    parser.add_argument('--batch_size', type=int, default=50000, 
                      help="Batch size for training")
    parser.add_argument('--node_datasize', type=int, default=50000, 
                      help="Number of data samples per node")
    
    # Model Parameters
    parser.add_argument('--model', type=str, default="ResNet50", 
                      help="Model type: ShuffleNet, ResNet18, ResNet34, ResNet50")
    parser.add_argument('--pretrained', type=bool, default=True, 
                      help="Whether to use pretrained model")
    # Hardware Parameters
    parser.add_argument('--device', type=str, default="cuda:7", 
                      help="Device to run the model on")
    parser.add_argument('--amp', type=bool, default=False, 
                      help="Whether to use automatic mixed precision")

    # Training Parameters
    parser.add_argument('--mode', type=str, default="all", 
                      help="Training communication networks")
    parser.add_argument('--shuffle', type=str, default="fixed", 
                      help="Shuffle mode for data loading")
    parser.add_argument('--size', type=int, default=200, 
                      help="Number of workers")
    parser.add_argument('--epochs', type=int, default=200, 
                      help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=42, 
                      help="Random seed for reproducibility")
    parser.add_argument('--patience', type=int, default=0,
                      help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument('--min_delta', type=float, default=1e-4,
                      help="Minimum change in the monitored quantity to qualify as an improvement")
    
    # Optimizer Parameters
    parser.add_argument('--lr', type=float, default=0.05, 
                      help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-4, 
                      help="Weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, 
                      help="Momentum for SGD")
    
    # Learning Rate Schedule Parameters
    parser.add_argument('--scheduler', type=str, default="multistep", 
                      help="Learning rate decay factor")
    parser.add_argument('--gamma', type=float, default=0.95, 
                      help="Learning rate decay factor")
    parser.add_argument('--warmup_step', type=int, default=0, 
                      help="Number of warmup steps")
    parser.add_argument('--milestones', type=int, nargs='+', default=None, 
                      help="Milestones for learning rate scheduling")
    
    # Data Distribution Parameters
    parser.add_argument('--nonIID', type=bool, default=False, 
                      help="Whether to use non-IID data distribution")
    parser.add_argument('--dirichlet', type=bool, default=False, 
                      help="Whether to use Dirichlet distribution for data splitting")
    parser.add_argument('--alpha', type=float, default=0.8, 
                      help="Alpha parameter for Dirichlet distribution")
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of training set to use (0~1)')

    args = parser.parse_args()

    # Automatically set milestones at 1/4, 1/2, and 3/4 of total epochs if not specified
    if args.milestones is None:
        N = args.epochs
        args.milestones = [N // 4, N // 2, (3 * N) // 4]

    return args

