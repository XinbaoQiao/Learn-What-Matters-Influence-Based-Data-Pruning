import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

class Worker_Vision:
    def __init__(self, model, rank, optimizer, scheduler, train_loader, device, choose_node, choose_batch, size, train_to_end):
        self.model = model
        self.rank = rank
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.size = size
        self.train_loader = train_loader
        self.device = device
        self.choose_node = choose_node
        self.choose_batch = choose_batch
        self.current_batch_index = -1
        self.train_to_end = train_to_end
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.current_iter = 0

    def update_iter(self):
        """Update the current iteration counter."""
        self.current_iter += 1

    def step(self, valid_loader):
        # Training step
        self.model.train()
        try:
            batch = next(iter(self.train_loader))
            self.current_batch_index += 1
        except StopIteration:
            print("Iteration ended")
            return

        self.data, self.target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(self.data)
        if not isinstance(output, torch.Tensor):
            output = output.logits
        loss = criterion(output, self.target)
        self.train_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()

        # Validation step
        self.model.eval()
        with torch.no_grad():
            try:
                valid_batch = next(iter(valid_loader))
                valid_data, valid_target = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                valid_output = self.model(valid_data)
                if not isinstance(valid_output, torch.Tensor):
                    valid_output = valid_output.logits
                self.valid_loss = criterion(valid_output, valid_target).item()
            except StopIteration:
                print("Validation iteration ended")
                return

    def update_grad(self):
        self.optimizer.step()
        self.scheduler.step()

    def get_train_loss(self):
        return self.train_loss

    def get_valid_loss(self):
        return self.valid_loss 