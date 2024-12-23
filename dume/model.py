import torch
import torch.nn as nn
from torch.nn import functional as F
import math
"""
Definition of DumE, a small language model created by me on a weekend
"""



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads

        if config.embed_dim % config.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        
        # Query, Key and Value projections
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim) # These can also be projected individaully
        
        # Output Projection
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # dropout to avoid overfitting
        self.attn_dropout = nn.Dropout(config.attn_drop)
        self.resid_dropout = nn.Dropout(config.resid_drop)

        # Returns a causal mask to ensure attention isn't applied to future tokens. It's applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension

        # Query, key, values for all the heads - a head is essentially a single instance of the attention mechanism
        qkv = self.qkv_proj(x) # Shape: (B, T, 3*C)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # Split into query, key and value tensors

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.num_heads))

        # Apply causal mask
        causal_mask = self.bias[:, :, :T, :T]  # Extract relevant part of the mask
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Attention output
        attn_output = attn_probs @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # Reassemble heads

        # Output projection
        y = self.resid_dropout(self.output_proj(attn_output))
        return y
    

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)

        # Feed-forward neural network
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_drop)
        )

    def forward(self, x):
        # Apply LayerNorm, Attention, and Residual Connection
        attn_output = self.attn(self.ln_1(x))
        x = x + attn_output

        # Apply LayerNorm, MLP, and Residual Connection
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        return x
    

class DumE(nn.Module):
    def __init__(self, config):
        super(DumE, self).__init__()

        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # LayerNorm before output
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Output projection layer
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Store block size
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Ensure sequence length does not exceed block size
        assert T <= self.block_size, "Sequence length exceeds model block size"

        # Compute token and position embeddings
        token_embeddings = self.token_embedding(idx)  # Shape: (B, T, C)
        position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)  # Shape: (1, T, C)

        x = token_embeddings + position_embeddings

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final LayerNorm and output projection
        x = self.ln_f(x)
        logits = self.head(x)  # Shape: (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop the context window to block size
            idx_cond = idx[:, -self.block_size:]

            # Forward pass to get logits
            logits = self(idx_cond)

            # Focus only on the last token's logits
            logits = logits[:, -1, :]

            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx