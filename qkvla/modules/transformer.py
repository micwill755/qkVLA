from __future__ import annotations

import torch
from torch import nn

from qkvla.modules.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from qkvla.modules.norms import AdaptiveRMSNorm, RMSNorm


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = (
            MultiHeadCrossAttention(dim, num_heads, dropout)
            if use_cross_attention
            else None
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.self_attn(self.self_norm(x))
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.cross_norm(x), context)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with timestep conditioning through adaptive LayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = (
            MultiHeadCrossAttention(dim, num_heads, dropout)
            if use_cross_attention
            else None
        )
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = FeedForward(dim, dropout=dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 6))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = self.modulation(
            cond
        ).chunk(6, dim=-1)
        x = x + gate_sa[:, None] * self.self_attn(
            modulate(self.self_norm(x), shift_sa, scale_sa)
        )
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.cross_norm(x), context)
        x = x + gate_ff[:, None] * self.ffn(
            modulate(self.ffn_norm(x), shift_ff, scale_ff)
        )
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class AdaRMSNormTransformerBlock(nn.Module):
    """Transformer block with adaRMSNorm timestep conditioning.

    This is useful for pi0/pi0.5-style flow action experts, where the continuous
    flow time is injected directly into the action-expert transformer layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_norm = AdaptiveRMSNorm(dim, cond_dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_norm = RMSNorm(dim)
        self.cross_attn = (
            MultiHeadCrossAttention(dim, num_heads, dropout)
            if use_cross_attention
            else None
        )
        self.ffn_norm = AdaptiveRMSNorm(dim, cond_dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed_x, gate_sa = self.self_norm(x, cond)
        x = x + gate_sa * self.self_attn(normed_x)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.cross_norm(x), context)
        normed_x, gate_ff = self.ffn_norm(x, cond)
        x = x + gate_ff * self.ffn(normed_x)
        return x
