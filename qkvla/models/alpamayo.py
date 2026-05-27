from __future__ import annotations

import torch
from torch import nn

from qkvla.models.action_experts import AlpamayoDiffusionTrajectoryDecoder
from qkvla.models.common import ToyVLMContext
from qkvla.modules.encoders import NumericSequenceEncoder
from qkvla.modules.transformer import TransformerBlock


class ReasoningBridge(nn.Module):
    """Toy reasoning-token module before the diffusion expert."""

    def __init__(
        self,
        model_dim: int,
        num_reasoning_tokens: int = 16,
        depth: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_reasoning_tokens, model_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    num_heads,
                    use_cross_attention=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.query.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            x = block(x, context)
        return self.norm(x)


class AlpamayoStyleVLA(nn.Module):
    """Alpamayo-inspired reason-then-denoise architecture."""

    def __init__(
        self,
        image_channels: int,
        patch_size: int,
        vocab_size: int,
        proprio_dim: int,
        action_dim: int,
        horizon: int,
        model_dim: int = 256,
        egomotion_dim: int = 12,
        expert_depth: int = 12,
    ) -> None:
        super().__init__()
        self.context_encoder = ToyVLMContext(
            image_channels, patch_size, vocab_size, proprio_dim, model_dim
        )
        self.egomotion_encoder = NumericSequenceEncoder(egomotion_dim, model_dim)
        self.reasoning = ReasoningBridge(model_dim)
        self.trajectory_expert = AlpamayoDiffusionTrajectoryDecoder(
            trajectory_dim=action_dim,
            horizon=horizon,
            model_dim=model_dim,
            depth=expert_depth,
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        egomotion_history: torch.Tensor,
        noisy_trajectory: torch.Tensor,
        diffusion_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(images, token_ids, proprio)
        context = torch.cat((context, self.egomotion_encoder(egomotion_history)), dim=1)
        reasoning_tokens = self.reasoning(context)
        expert_context = torch.cat((context, reasoning_tokens), dim=1)
        pred_noise = self.trajectory_expert(
            noisy_trajectory, diffusion_t, expert_context
        )
        return {
            "reasoning_tokens": reasoning_tokens,
            "pred_noise": pred_noise,
        }
