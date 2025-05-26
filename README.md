## Overview
This project implements a decentralized learning framework for training neural networks using multiple workers. It supports various datasets (SVHN, CIFAR10, CIFAR100, TinyImageNet) and models (SqueezeNet, ResNet18, ResNet50). Each worker uses SGD as the optimizer。

Crucially, this project now includes implementations of various **data influence calculation methods (Influence Function, DICE, Delete)** to identify and potentially prune less influential data samples for improved efficiency and performance.

## Installation
   ```bash
   pip install torch torchvision numpy pydvl joblib zarr distributed matplotlib seaborn
   ```

## Usage
1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd decentralized_learning
   ```
2. Run the training process in two stages:
   
   **Stage 1: Train with the full dataset.**
   ```bash
   python main.py --stage 1 [other arguments like --dataset_name, --model, --num_workers, --epochs, --pruning_algorithm etc.]
   ```
   This will train the model on the full dataset and save the necessary models and indices for influence calculation.

   **Stage 2: Train with the pruned dataset.**
   ```bash
   python main.py --stage 2 --train_ratio [ratio] [same other arguments as Stage 1]
   ```
   Replace `[ratio]` with the desired training data ratio after pruning (e.g., 0.5 for 50%). This stage will load the influence scores (calculated in Stage 1) and prune the dataset accordingly before training.

   Ensure you use the same hyperparameters (dataset, model, num_workers, epochs, etc.) for both Stage 1 and Stage 2 to maintain consistency.

## Project Structure
```
decentralized_learning/
│
├── main.py                  # Main script for pruned training stages (incorporates influence calculation and pruning)        
├── workers/
│   └── worker_vision.py     # Worker node training logic
├── utils/
│   ├── args.py              # Command-line argument parsing
│   ├── scheduler.py         # Learning rate scheduler
│   ├── random.py            # Random seed setting
│   ├── dirichlet.py         # Dirichlet data partitioning
│   ├── early_stopping.py    # Early stopping mechanism (if applicable)
│   └── influence.py         # Implementations of Influence Function (IF), DICE, and Delete influence calculation. IF uses pyDVL's EkfacInfluence on the last linear layer.
├── networks/
│   └── __init__.py          # Model loading functions
├── training/
│   └── trainer.py           # Training process and core logic
├── data/
│   └── dataset_loader.py    # Dataset loading
└── visualization/
    └── plot_utils.py        # Visualization tools, currently focused on plotting training loss comparison between stages.
``` 
