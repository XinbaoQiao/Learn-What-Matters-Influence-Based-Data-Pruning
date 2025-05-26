import torch
import random
import numpy as np
import os

def seed_torch(seed=42):
    random.seed(seed)   # Python randomness
    os.environ['PYTHONHASHSEED'] = str(seed)    # Set Python hash seed to disable hash randomization for reproducibility
    np.random.seed(seed)   # Numpy randomness
    torch.manual_seed(seed)   # PyTorch CPU randomness
    torch.cuda.manual_seed(seed)   # PyTorch GPU randomness for current GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU randomness for all GPUs
    torch.backends.cudnn.benchmark = False   # If benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # Choose deterministic algorithms

# Keep set_seed function for backward compatibility
def set_seed(seed):
    seed_torch(seed) 