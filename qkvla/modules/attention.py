from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention over a sequence shaped [batch, tokens, dim]."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(y)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention from query tokens to context tokens."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        ctx_len = context.shape[1]
        q = self.q(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        kv = self.kv(context).view(bsz, ctx_len, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(y)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout: nn.Dropout,
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))
    weights = scores.softmax(dim=-1)
    weights = dropout(weights)
    return weights @ v

