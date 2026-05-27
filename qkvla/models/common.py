from __future__ import annotations

import torch
from torch import nn

from qkvla.modules.encoders import PatchImageEncoder, ProprioEncoder, TextTokenEncoder


class ToyVLMContext(nn.Module):
    """Small stand-in for a VLM context encoder.

    This is intentionally simple. Later we can replace it with a real vision
    backbone, language model, or frozen VLM while keeping the action expert API.
    """

    def __init__(
        self,
        image_channels: int,
        patch_size: int,
        vocab_size: int,
        proprio_dim: int,
        model_dim: int,
        include_proprio: bool = True,
    ) -> None:
        super().__init__()
        self.include_proprio = include_proprio
        self.image = PatchImageEncoder(image_channels, patch_size, model_dim)
        self.text = TextTokenEncoder(vocab_size, model_dim)
        self.proprio = ProprioEncoder(proprio_dim, model_dim) if include_proprio else None
        self.type_embed = nn.Parameter(torch.zeros(1, 3, model_dim))

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self.image(images) + self.type_embed[:, 0:1]
        text_tokens = self.text(token_ids) + self.type_embed[:, 1:2]
        tokens = [image_tokens, text_tokens]
        if self.proprio is not None:
            tokens.append(self.proprio(proprio) + self.type_embed[:, 2:3])
        return torch.cat(tokens, dim=1)
