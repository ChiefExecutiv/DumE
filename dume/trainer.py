import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
# from mingpt.utils import CfgNode as CN

def CN():
    pass

class Trainer:

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device to be trained on
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.model.to(self.device)

        print(f"Running on device: {self.device}")

        # Variables that will be assigned to the trainer class??
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        if not hasattr(config, 'grad_norm_clip'):
            config.grad_norm_clip = 1.0  # or any default value you prefer

        # dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle = False,
            pin_memory = True,
            batch_size = config.batch_size,
            num_workers = config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch(x, y)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backpropagation and update parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break