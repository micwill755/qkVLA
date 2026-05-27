from __future__ import annotations

import torch
from torch import nn

from qkvla.models.common import ToyVLMContext
from qkvla.models.action_experts import GR00TActionFlowExpert


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
        num_embodiments: int = 1,
        expert_depth: int = 16,
    ) -> None:
        super().__init__()
        self.context_encoder = ToyVLMContext(
            image_channels,
            patch_size,
            vocab_size,
            proprio_dim,
            model_dim,
            include_proprio=False,
        )
        self.action_expert = GR00TActionFlowExpert(
            state_dim=proprio_dim,
            action_dim=action_dim,
            horizon=horizon,
            model_dim=model_dim,
            depth=expert_depth,
            num_embodiments=num_embodiments,
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_t: torch.Tensor,
        embodiment_id: int = 0,
    ) -> torch.Tensor:
        context = self.context_encoder(images, token_ids, proprio)
        return self.action_expert(
            proprio,
            noisy_actions,
            flow_t,
            context,
            embodiment_id,
        )
