from __future__ import annotations

import torch
from torch import nn

from qkvla.modules.embeddings import SinusoidalTimestepEmbedding
from qkvla.modules.transformer import AdaLNTransformerBlock


class ActionDenoiserTransformer(nn.Module):
    """Transformer denoiser over action chunks shaped [batch, horizon, action_dim]."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        model_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        prediction_type: str = "noise",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.prediction_type = prediction_type
        self.action_in = nn.Linear(action_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon, model_dim))
        self.time_emb = SinusoidalTimestepEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    model_dim,
                    num_heads,
                    model_dim,
                    dropout=dropout,
                    use_cross_attention=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.action_out = nn.Linear(model_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.shape[1] != self.horizon:
            raise ValueError("noisy_actions horizon does not match model horizon")
        x = self.action_in(noisy_actions) + self.pos
        cond = self.time_emb(t)
        for block in self.blocks:
            x = block(x, cond, context)
        return self.action_out(self.norm(x))

