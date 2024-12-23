import torch
from torch.utils.data import Dataset, DataLoader
from dume.model import DumE
from dume.trainer import Trainer


"""
This trains DumE as a character level language model on a Wikipedia article.
"""
class ModelConfig:
    """
    Configuration for the language model.
    """
    def __init__(self, vocab_size, block_size, embed_dim, num_heads, num_layers, dropout, attn_drop, resid_drop):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_drop = attn_drop
        self.resid_drop = resid_drop

    def summary(self):
        """
        Print a summary of the configuration.
        """
        config_dict = vars(self)
        print("Model Configuration:")
        for key, value in config_dict.items():
            print(f"{key}: {value}")


class TrainerConfig:
    """
    Configuration for the trainer.
    """
    def __init__(self, batch_size, num_workers, learning_rate, max_iters, eval_interval, eval_iters):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

    def summary(self):
        """
        Prints a summary of the training configuration.
        """
        config_dict = vars(self)
        print("Trainer Configuration:")
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

if __name__ == "__main__":
    # Load text data
    with open("big-bang.txt", "r") as f: # The dataset isn't included in this repository
        text = f.read()

    # Initialize dataset and configurations
    block_size = 128
    train_dataset = CharDataset(text, block_size)

    model_config = ModelConfig(
        vocab_size=train_dataset.vocab_size,
        block_size=block_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        attn_drop=0.1,
        resid_drop=0.1
    )

    trainer_config = TrainerConfig(
        batch_size=32,
        num_workers=2,
        learning_rate=3e-4,
        max_iters=1000,
        eval_interval=100,
        eval_iters=10
    )

    # Create model and trainer
    model = DumE(model_config)
    trainer = Trainer(trainer_config, model, train_dataset)

    # Print configurations
    model_config.summary()
    trainer_config.summary()

    # Train the model
    trainer.run()

    # Saving the model
    torch.save(model.state_dict(), "trained_model.pt")
    print("Model saved as trained_model.pt")

    # Generate samples
    with torch.no_grad():
        context = "The Big Bang"
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(trainer.device)
        y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
        completion = ''.join([train_dataset.itos[int(i)] for i in y])
        print("Generated text:")
        print(completion)
