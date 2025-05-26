import os
import torch
from torch.optim import SGD
from tqdm import tqdm
import copy
from utils.scheduler import Warmup_MultiStepLR
from utils.early_stopping import EarlyStopping
from workers.worker_vision import Worker_Vision
from utils.utils import generate_P, update_dsgd, merge_without_update, merge_model
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import concurrent.futures
from utils.random import seed_torch
from utils.args import parse_args

args = parse_args()
seed_torch(args.seed)

class Trainer:
    def __init__(self, args, model, train_dataloaders, valid_dataloaders, test_dataloaders, result_dir):
        seed_torch(args.seed)  # Use random seed specified in command line arguments
        self.args = args
        self.result_dir = result_dir
        self.train_dataloaders = train_dataloaders
        self.valid_dataloaders = valid_dataloaders
        self.test_dataloaders = test_dataloaders
        
        # Initialize workers
        self.worker_list = []
        # Move model to device first
        model = model.to(args.device)
        for rank in range(args.num_workers):
            # Use same model instance
            optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
            if args.scheduler == "multistep":
                scheduler = Warmup_MultiStepLR(
                    optimizer, 
                    warmup_step=args.warmup_step, 
                    milestones=args.milestones, 
                    gamma=args.gamma
                )
            elif args.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=args.gamma)
            else:
                raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
            worker = Worker_Vision(model, rank, optimizer, scheduler, 
                                 train_dataloaders[rank], args.device, 0, 10, args.num_workers, False)
            self.worker_list.append(worker)
        
        # Initialize early stopping
        # self.early_stopping = EarlyStopping(
        #     patience=args.patience,
        #     delta=args.min_delta,
        #     verbose=False
        # )
        # self.best_model_path = os.path.join(result_dir, 'best_model.pth')
        
        # Generate communication matrix
        self.P = generate_P(args.mode, args.num_workers)

    def _get_global_model(self):
        """Calculate the global model (average of all worker models)"""
        global_model = copy.deepcopy(self.worker_list[0].model)
        global_model = global_model.to(self.args.device)
        
        with torch.no_grad():
            for param in global_model.parameters():
                param.zero_()
            for worker in self.worker_list:
                for global_param, worker_param in zip(global_model.parameters(), worker.model.parameters()):
                    global_param.add_(worker_param.data / len(self.worker_list))
        
        return global_model

    def _calculate_global_loss(self, dataloader):
        """Calculate the loss of the global model (average of all models)"""
        global_model = self._get_global_model()
        global_model.eval()
        
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                data, target = batch[:2]
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = global_model(data)
                if not isinstance(output, torch.Tensor):
                    output = output.logits
                loss = torch.nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item() * len(data)
                total_samples += len(data)
        
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def _calculate_global_accuracy(self, dataloader):
        """Calculate accuracy using the global model (average of all worker models)"""
        global_model = self._get_global_model()
        global_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch[:2]
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                
                outputs = global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100.0 * correct / total

    def train(self):
        # Save args before training starts
        import json
        args_dict = vars(self.args)
        with open(os.path.join(self.result_dir, 'args.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)

        # Calculate total steps
        steps = self.args.epochs 
        print("\nTraining Progress:")
        pbar = tqdm(total=steps, desc="Training", position=0, leave=True, disable=False)
        # Initialize loss and accuracy history
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accs = []
        valid_accs = []
        test_accs = []
        lr_list = []
        
        for step in range(steps):
            # Model merging (gossip communication)
            merge_model(self.worker_list, self.P)
            # Local updates
            update_dsgd(self.worker_list, self.P, self.args, self.train_dataloaders)
            # Calculate global model metrics
            global_train_loss = self._calculate_global_loss(self.train_dataloaders[0])
            global_valid_loss = self._calculate_global_loss(self.valid_dataloaders[0])
            global_test_loss = self._calculate_global_loss(self.test_dataloaders[0])
            global_train_acc = self._calculate_global_accuracy(self.train_dataloaders[0])
            global_valid_acc = self._calculate_global_accuracy(self.valid_dataloaders[0])
            global_test_acc = self._calculate_global_accuracy(self.test_dataloaders[0])
            # Save metrics
            train_losses.append(global_train_loss)
            valid_losses.append(global_valid_loss)
            test_losses.append(global_test_loss)
            train_accs.append(global_train_acc)
            valid_accs.append(global_valid_acc)
            test_accs.append(global_test_acc)
            
            # Save global model and local model parameters for each epoch
            models_dir = os.path.join(self.result_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save global model parameters
            global_model = self._get_global_model()
            torch.save(global_model.state_dict(), os.path.join(models_dir, f'global_model_epoch_{step}.pth'))
            
            # Save local model parameters (parallelized)
            def save_worker_model(i, worker, models_dir, step):
                torch.save(worker.model.state_dict(), os.path.join(models_dir, f'local_model_{i}_epoch_{step}.pth'))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(save_worker_model, i, worker, models_dir, step)
                    for i, worker in enumerate(self.worker_list)
                ]
                concurrent.futures.wait(futures)
            
            # Record current learning rate
            current_lr = self.worker_list[0].optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)
            
            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{global_train_loss:.4f}',
                'valid_loss': f'{global_valid_loss:.4f}',
                'test_loss': f'{global_test_loss:.4f}',
                'train_acc': f'{global_train_acc:.1f}%',
                'valid_acc': f'{global_valid_acc:.1f}%',
                'test_acc': f'{global_test_acc:.1f}%'
            })
            
        pbar.close()
        print("\nTraining completed!")
        
        # Save learning rates to models/lr_list.npy
        np.save(os.path.join(models_dir, 'lr_list.npy'), np.array(lr_list))

        print(f"Saving metrics to {self.result_dir}")
        # Save metrics to files
        metrics_dir = os.path.join(self.result_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        np.save(os.path.join(metrics_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(metrics_dir, 'train_accs.npy'), np.array(train_accs))
        np.save(os.path.join(metrics_dir, 'valid_losses.npy'), np.array(valid_losses))
        np.save(os.path.join(metrics_dir, 'valid_accs.npy'), np.array(valid_accs))
        np.save(os.path.join(metrics_dir, 'test_losses.npy'), np.array(test_losses))
        np.save(os.path.join(metrics_dir, 'test_accs.npy'), np.array(test_accs))

