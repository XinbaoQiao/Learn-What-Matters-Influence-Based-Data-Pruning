import os
import torch
from torch.optim import SGD
from tqdm import tqdm
import copy
from utils.scheduler import Warmup_MultiStepLR
from utils.early_stopping import EarlyStopping
from workers.worker_vision import Worker_Vision
from visualization.plot_utils import plot_metrics
from utils.utils import generate_P, update_dsgd, merge_without_update, merge_model
from torch.optim.lr_scheduler import ExponentialLR

class Trainer:
    def __init__(self, args, model, train_dataloaders, valid_dataloaders, test_dataloaders, result_dir):
        self.args = args
        self.result_dir = result_dir
        self.train_dataloaders = train_dataloaders
        self.valid_dataloaders = valid_dataloaders
        self.test_dataloaders = test_dataloaders
        
        # Initialize workers
        self.worker_list = []
        for rank in range(args.size):
            model = model.to(args.device)
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
                                 train_dataloaders[rank], args.device, 0, 10, args.size, False)
            self.worker_list.append(worker)
        
        # Initialize early stopping
        # self.early_stopping = EarlyStopping(
        #     patience=args.patience,
        #     delta=args.min_delta,
        #     verbose=False
        # )
        # self.best_model_path = os.path.join(result_dir, 'best_model.pth')
        
        # Generate communication matrix
        self.P = generate_P(args.mode, args.size)

    def _calculate_global_loss(self, dataloader):
        """Calculate the loss of the global model (average of all models)."""
        # Create a temporary model to hold the global average
        global_model = copy.deepcopy(self.worker_list[0].model)
        global_model = global_model.to(self.args.device)
        
        # Average all model parameters
        with torch.no_grad():
            for param in global_model.parameters():
                param.zero_()
            for worker in self.worker_list:
                for global_param, worker_param in zip(global_model.parameters(), worker.model.parameters()):
                    global_param.add_(worker_param.data / len(self.worker_list))
        
        # Calculate loss on dataset
        global_model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                data, target = batch[0].to(self.args.device), batch[1].to(self.args.device)
                output = global_model(data)
                if not isinstance(output, torch.Tensor):
                    output = output.logits
                loss = torch.nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item() * len(data)
                total_samples += len(data)
        
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def _calculate_global_accuracy(self, dataloader):
        """Calculate accuracy using the global model (first worker's model)"""
        model = self.worker_list[0].model
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)
                
                outputs = model(inputs)
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
        
        for step in range(steps):
            # Model merging (gossip communication)
            # merge_without_update(self.worker_list, self.P, self.args, self.train_dataloaders)
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
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'train_loss': f'{global_train_loss:.4f}',
                'valid_loss': f'{global_valid_loss:.4f}',
                'test_loss': f'{global_test_loss:.4f}',
                'train_acc': f'{global_train_acc:.1f}%',
                'valid_acc': f'{global_valid_acc:.1f}%',
                'test_acc': f'{global_test_acc:.1f}%'
            })
            
            # Early stopping check using validation loss
            # if self.early_stopping(global_valid_loss, self.worker_list[0].model, self.best_model_path):
            #     print("\nEarly stopping triggered")
            #     break
        
        pbar.close()
        print("\nTraining completed!")
        
        # Save metrics to files
        metrics_dir = os.path.join(self.result_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save metrics as numpy arrays
        import numpy as np
        np.save(os.path.join(metrics_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(metrics_dir, 'train_accs.npy'), np.array(train_accs))
        np.save(os.path.join(metrics_dir, 'valid_losses.npy'), np.array(valid_losses))
        np.save(os.path.join(metrics_dir, 'valid_accs.npy'), np.array(valid_accs))
        np.save(os.path.join(metrics_dir, 'test_losses.npy'), np.array(test_losses))
        np.save(os.path.join(metrics_dir, 'test_accs.npy'), np.array(test_accs))
        
        # Plot all metrics
        plot_metrics(train_losses, train_accs, valid_losses, valid_accs, test_losses, test_accs, self.result_dir)
