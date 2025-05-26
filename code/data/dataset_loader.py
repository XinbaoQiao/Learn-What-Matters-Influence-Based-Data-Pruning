import os
import torch
from torchvision import datasets, transforms
import urllib.request
import zipfile
from tqdm import tqdm
import shutil
from torch.utils.data import random_split

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, filepath):
    """Download a file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=filepath,
                                 reporthook=t.update_to)

class IndexDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the sample index."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

def load_dataset(args):
    if args.dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
        # Split test set into validation and test sets
        test_size = len(test_set)
        valid_size = test_size // 2
        test_size = test_size - valid_size
        valid_set, test_set = random_split(test_set, [valid_size, test_size])
        nb_class = 10
    elif args.dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transform)
        # Split test set into validation and test sets
        test_size = len(test_set)
        valid_size = test_size // 2
        test_size = test_size - valid_size
        valid_set, test_set = random_split(test_set, [valid_size, test_size])
        nb_class = 100
    elif args.dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  
        ])
        train_set = datasets.FashionMNIST(root=args.dataset_path, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=args.dataset_path, train=False, download=True, transform=transform)
        # Split test set into validation and test sets
        test_size = len(test_set)
        valid_size = test_size // 2
        test_size = test_size - valid_size
        valid_set, test_set = random_split(test_set, [valid_size, test_size])
        nb_class = 10
    elif args.dataset_name == "tinyimagenet":
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download and extract Tiny ImageNet if not exists
        dataset_dir = os.path.join(args.dataset_path, 'tiny-imagenet-200')
        if not os.path.exists(dataset_dir):
            zip_path = os.path.join(args.dataset_path, "tiny-imagenet-200.zip")
            if not os.path.exists(zip_path):
                print('Downloading Tiny ImageNet...')
                os.makedirs(args.dataset_path, exist_ok=True)
                download_url("http://cs231n.stanford.edu/tiny-imagenet-200.zip", zip_path)
            
            print('Extracting Tiny ImageNet...')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(args.dataset_path)
            os.remove(zip_path)
            print('Done!')
            
        train_set = datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=transform)
        test_set = datasets.ImageFolder(root=os.path.join(dataset_dir, 'val'), transform=transform)
        # Split test set into validation and test sets
        test_size = len(test_set)
        valid_size = test_size // 2
        test_size = test_size - valid_size
        valid_set, test_set = random_split(test_set, [valid_size, test_size])
        nb_class = 200
    elif args.dataset_name == "svhn":
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.SVHN(root=args.dataset_path, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root=args.dataset_path, split='test', download=True, transform=transform)
        # Split test set into validation and test sets
        test_size = len(test_set)
        valid_size = test_size // 2
        test_size = test_size - valid_size
        valid_set, test_set = random_split(test_set, [valid_size, test_size])
        nb_class = 10
    else:
        raise ValueError("Dataset not supported")


    # Wrap all sets with IndexDataset
    train_set = IndexDataset(train_set)
    valid_set = IndexDataset(valid_set)
    test_set = IndexDataset(test_set)

    return train_set, valid_set, test_set, nb_class
