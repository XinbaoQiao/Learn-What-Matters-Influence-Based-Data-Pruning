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
2. Run the main script:
   ```bash
   python main.py
   ```
   Use command-line arguments to configure dataset, model, decentralized learning parameters, and influence calculation methods (e.g., `--pruning_algorithm if`, `--stage 2`).

## Project Structure
```
decentralized_learning/
│
├── main.py                  # Main entry script: argument parsing, data loading, training, and result saving
├── main_pruned.py           # Main script for pruned training stages (incorporates influence calculation and pruning)
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
