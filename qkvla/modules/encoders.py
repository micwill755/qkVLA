from __future__ import annotations

import torch
from torch import nn

from qkvla.modules.transformer import TransformerBlock


class PatchImageEncoder(nn.Module):
    """Tiny patch encoder for image observations."""

    def __init__(
        self,
        image_channels: int,
        patch_size: int,
        model_dim: int,
        max_patches: int = 256,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        patch_dim = image_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, model_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        bsz, channels, grid_h, grid_w, patch_h, patch_w = patches.shape
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            bsz, grid_h * grid_w, channels * patch_h * patch_w
        )
        return self.proj(patches) + self.pos[:, : patches.shape[1]]


class TextTokenEncoder(nn.Module):
    """Small text-token encoder used before real tokenizer/backbone integration."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        max_tokens: int = 256,
        depth: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, model_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token(token_ids) + self.pos[:, : token_ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ProprioEncoder(nn.Module):
    """Projects robot state/proprioception into one context token."""

    def __init__(self, proprio_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)[:, None]


class NumericSequenceEncoder(nn.Module):
    """Projects a numeric history sequence into context tokens."""

    def __init__(self, input_dim: int, model_dim: int, max_steps: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_steps, model_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.proj(sequence) + self.pos[:, : sequence.shape[1]]
