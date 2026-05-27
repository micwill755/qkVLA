from __future__ import annotations

import torch
from torch import nn

from qkvla.models.common import ToyVLMContext
from qkvla.modules.action_denoiser import ActionDenoiserTransformer


class GR00TStyleVLA(nn.Module):
    """GR00T-inspired dual-system VLA.

    System 2 is represented by `context_encoder`. System 1 is represented by a
    diffusion transformer action expert.
    """

    def __init__(
        self,
        image_channels: int,
        patch_size: int,
        vocab_size: int,
        proprio_dim: int,
        action_dim: int,
        horizon: int,
        model_dim: int = 256,
    ) -> None:
        super().__init__()
        self.context_encoder = ToyVLMContext(
            image_channels, patch_size, vocab_size, proprio_dim, model_dim
        )
        self.action_expert = ActionDenoiserTransformer(
            action_dim=action_dim,
            horizon=horizon,
            model_dim=model_dim,
            prediction_type="noise",
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        noisy_actions: torch.Tensor,
        diffusion_t: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_encoder(images, token_ids, proprio)
        return self.action_expert(noisy_actions, diffusion_t, context)

