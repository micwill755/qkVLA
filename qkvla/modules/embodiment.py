from __future__ import annotations

import torch
from torch import nn


class EmbodimentActionAdapter(nn.Module):
    """Embodiment-specific action projection.

    GR00T-style systems need robot-specific state/action adapters around a
    shared action expert. This small adapter is the from-scratch placeholder for
    that pattern.
    """

    def __init__(self, action_dim: int, model_action_dim: int, num_embodiments: int) -> None:
        super().__init__()
        self.in_proj = nn.ModuleList(
            [nn.Linear(action_dim, model_action_dim) for _ in range(num_embodiments)]
        )
        self.out_proj = nn.ModuleList(
            [nn.Linear(model_action_dim, action_dim) for _ in range(num_embodiments)]
        )

    def encode(self, actions: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        return self.in_proj[embodiment_id](actions)

    def decode(self, actions: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
        return self.out_proj[embodiment_id](actions)
