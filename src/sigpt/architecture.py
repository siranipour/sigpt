from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from sigpt.config import ModelConfig

# This needs to be a global constant so that model.compile downstream
# knows to only consider the flash attention path.
USE_FLASH_ATTENTION: Final[bool] = True


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(config.vocab_size, config.n_embed),
                pos_embedding=nn.Embedding(config.block_size, config.n_embed),
                hidden=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight sharing semantics for embedding/unembedding layers
        self.transformer.token_embedding.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError("Input context length exceeds max block size {}")
        token_embedding = self.transformer.token_embedding(idx)
        time_grid = torch.arange(T, device=idx.device)
        pos_embedding = self.transformer.pos_embedding(time_grid)

        x = pos_embedding + token_embedding
        for block in self.transformer.hidden:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.h1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.activation = nn.GELU()
        self.h2 = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x: torch.Tensor):
        x = self.h1(x)
        x = self.activation(x)
        x = self.h2(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads

        if self.n_embed % self.n_heads != 0:
            raise ValueError(
                f"The number of attention heads, {self.n_heads}, does not evenly "
                f"divide the embedding dimension, {self.n_embed}"
            )

        self.hs = self.n_embed // self.n_heads

        self.kqv = nn.Linear(self.n_embed, 3 * self.n_embed)
        # For projecting the concatenated output of causal self-attention back
        # into the residual pathway
        self.proj = nn.Linear(self.n_embed, self.n_embed)

        if USE_FLASH_ATTENTION:
            # Register the causal self-attention mask. Broadcast to (1, 1, T, T)
            # where the dummy 1, 1 indices are for batch and head indices.
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))[None, None, ...],
            )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        kqv = self.kqv(x)  # (B, T, 3 * C)
        k, q, v = kqv.split(self.n_embed, dim=-1)  # Each array is then (B, T, C)

        # Function to reshape each attention tensor to include a head dimension.
        # Recall that the embedding dimension has been asserted to be divisible
        # by the number of heads at the __init__ level of this module.
        # Transpose the middle 2 dims to treat the number of heads as a batch
        # dimension.
        def attn_reshape(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        k, q, v = map(attn_reshape, (k, q, v))  # (B, nh, T, hs)

        if USE_FLASH_ATTENTION:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = (k @ q.transpose(-1, -2)) / (self.hs**0.5)  # (B, nh, T, T)
            # Only slice out the part of the causal mask we need for this batch
            attn = attn.where(self.causal_mask[..., :T, :T] != 0, -torch.inf)
            # Normalize reduce operation weights
            attn = F.softmax(attn, dim=-1)
            y = attn @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)
