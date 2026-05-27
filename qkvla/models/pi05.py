from __future__ import annotations

import torch
from torch import nn

from qkvla.models.common import ToyVLMContext
from qkvla.modules.action_denoiser import ActionDenoiserTransformer


class Pi05StyleVLA(nn.Module):
    """pi0.5-inspired VLA with a flow-matching action expert.

    This skeleton keeps the action expert separate from heterogeneous heads so
    we can later add semantic subtask, detection, or language losses.
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
        num_subtasks: int = 32,
    ) -> None:
        super().__init__()
        self.context_encoder = ToyVLMContext(
            image_channels, patch_size, vocab_size, proprio_dim, model_dim
        )
        self.action_expert = ActionDenoiserTransformer(
            action_dim=action_dim,
            horizon=horizon,
            model_dim=model_dim,
            prediction_type="velocity",
        )
        self.subtask_head = nn.Linear(model_dim, num_subtasks)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(images, token_ids, proprio)
        pooled_context = context.mean(dim=1)
        velocity = self.action_expert(noisy_actions, flow_t, context)
        return {
            "velocity": velocity,
            "subtask_logits": self.subtask_head(pooled_context),
        }

