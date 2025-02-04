from dume.model import DumE
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dataclasses import dataclass

"""
Script to load model weights and carry out inference with the trained and saved model.
"""

@dataclass
class ModelConfig:
    vocab_size: int 
    block_size: int 
    embed_dim: int
    num_heads: int
    num_layers : int
    dropout : float
    attn_drop : float
    resid_drop: float

    def summary(self):
        """
        Print a summary of the configuration.
        """
        config_dict = vars(self)
        print("Model Configuration:")
        for key, value in config_dict.items():
            print(f"{key}: {value}")



class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[ch] for ch in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    

with open("A_dataset_cleaned.txt", "r") as f: # The dataset isn't included in this repository
    text = f.read()


train_dataset = CharDataset(text, 128)


model_config = ModelConfig(
    vocab_size=train_dataset.vocab_size,
    block_size=128,
    embed_dim=256,
    num_heads=4,
    num_layers=4,
    dropout=0.1,
    attn_drop=0.1,
    resid_drop=0.1
)


def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DumE(model_config)
    model.to(device)
    model.load_state_dict(checkpoint)
    return model

model = load_model('model_v2.pt')
context = "[Creative Writing]"

with torch.no_grad():
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to('cpu')
    # print(x)
    y = model.generate(x, 500)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(f"Input: {context}")
    print("Generated text:")
    print(completion)

