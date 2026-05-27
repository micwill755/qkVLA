from __future__ import annotations

import math

import torch
from torch import nn

from qkvla.modules.norms import RMSNorm


class FourierEncoder(nn.Module):
    """Log-spaced Fourier features for scalar action/timestep values."""

    def __init__(self, dim: int, max_freq: float = 100.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("FourierEncoder dim must be even")
        freqs = torch.logspace(0, math.log10(max_freq), steps=dim // 2)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = x[..., None] * self.freqs * (2 * math.pi)
        return torch.cat((torch.sin(args), torch.cos(args)), dim=-1) * math.sqrt(2)


class MLPEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for idx in range(num_layers):
            out_dim = output_dim if idx == num_layers - 1 else hidden_dim
            layers.extend([RMSNorm(hidden_dim, eps=1e-5), nn.Linear(hidden_dim, out_dim)])
            if idx != num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerWaypointActionProjection(nn.Module):
    """GR00T/Alpamayo-style per-waypoint action projection with timestep features."""

    def __init__(
        self,
        action_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_fourier_feats: int = 20,
        max_freq: float = 100.0,
    ) -> None:
        super().__init__()
        self.action_features = nn.ModuleList(
            [FourierEncoder(num_fourier_feats, max_freq) for _ in range(action_dim)]
        )
        self.time_features = FourierEncoder(num_fourier_feats, max_freq)
        input_dim = num_fourier_feats * (action_dim + 1)
        self.encoder = MLPEncoder(input_dim, hidden_dim, output_dim, num_layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        bsz, horizon, _ = actions.shape
        action_feats = torch.cat(
            [encoder(actions[:, :, idx]) for idx, encoder in enumerate(self.action_features)],
            dim=-1,
        )
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None].expand(bsz, horizon)
        elif timesteps.ndim == 3:
            timesteps = timesteps[..., -1].expand(bsz, horizon)
        time_feats = self.time_features(timesteps)
        x = torch.cat((action_feats, time_feats), dim=-1)
        x = self.encoder(x.flatten(0, 1)).reshape(bsz, horizon, -1)
        return self.norm(x)


class CategorySpecificMLP(nn.Module):
    """Small category-specific MLP bank for embodiment-specific projections."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    RMSNorm(hidden_dim, eps=1e-5),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_categories)
            ]
        )

    def forward(self, x: torch.Tensor, category_ids: torch.Tensor | int = 0) -> torch.Tensor:
        if isinstance(category_ids, int):
            return self.mlps[category_ids](x)
        outs = torch.empty(*x.shape[:-1], self.output_dim, device=x.device, dtype=x.dtype)
        for category in category_ids.unique().tolist():
            mask = category_ids == category
            outs[mask] = self.mlps[int(category)](x[mask])
        return outs
