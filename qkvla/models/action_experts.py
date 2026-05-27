from __future__ import annotations

import torch
from torch import nn

from qkvla.modules.action_denoiser import ActionDenoiserTransformer
from qkvla.modules.action_projection import CategorySpecificMLP, PerWaypointActionProjection
from qkvla.modules.embeddings import SinusoidalTimestepEmbedding
from qkvla.modules.transformer import AdaLNTransformerBlock


class GR00TActionFlowExpert(nn.Module):
    """GR00T-style action flow-matching expert.

    Public descriptions identify GR00T N1's System 1 as a diffusion transformer
    trained with action flow matching. This module predicts the flow velocity
    for a noisy action chunk conditioned on VLM/proprio context tokens.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        model_dim: int = 256,
        input_embedding_dim: int | None = None,
        depth: int = 16,
        num_heads: int = 8,
        num_embodiments: int = 1,
        num_timestep_buckets: int = 1000,
    ) -> None:
        super().__init__()
        input_embedding_dim = input_embedding_dim or model_dim
        self.horizon = horizon
        self.num_timestep_buckets = num_timestep_buckets
        self.state_encoder = CategorySpecificMLP(
            num_embodiments, state_dim, model_dim, input_embedding_dim
        )
        self.action_encoder = PerWaypointActionProjection(
            action_dim, input_embedding_dim, hidden_dim=max(model_dim, input_embedding_dim)
        )
        self.position_embedding = nn.Embedding(horizon, input_embedding_dim)
        self.in_proj = nn.Linear(input_embedding_dim, model_dim)
        self.time_emb = SinusoidalTimestepEmbedding(model_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    model_dim,
                    num_heads,
                    model_dim,
                    use_cross_attention=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.action_decoder = CategorySpecificMLP(
            num_embodiments, model_dim, model_dim, action_dim
        )

    def forward(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        flow_t: torch.Tensor,
        context: torch.Tensor,
        embodiment_id: torch.Tensor | int = 0,
    ) -> torch.Tensor:
        if noisy_actions.shape[1] != self.horizon:
            raise ValueError("noisy_actions horizon does not match model horizon")
        if state.ndim == 2:
            state = state[:, None]
        t_bucket = (flow_t * self.num_timestep_buckets).long()
        state_tokens = self.state_encoder(state, embodiment_id)
        action_tokens = self.action_encoder(noisy_actions, t_bucket)
        positions = torch.arange(self.horizon, device=noisy_actions.device)
        action_tokens = action_tokens + self.position_embedding(positions)[None]
        x = self.in_proj(torch.cat((state_tokens, action_tokens), dim=1))
        cond = self.time_emb(t_bucket)
        for block in self.blocks:
            x = block(x, cond, context)
        pred_tokens = self.norm(x[:, -self.horizon :])
        return self.action_decoder(pred_tokens, embodiment_id)


class Pi05FlowActionExpert(nn.Module):
    """pi0.5-style flow-matching action expert with adaRMSNorm conditioning."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        model_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.net = ActionDenoiserTransformer(
            action_dim=action_dim,
            horizon=horizon,
            model_dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            prediction_type="velocity",
            conditioning_norm="adarms",
        )

    def forward(
        self, noisy_actions: torch.Tensor, flow_t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.net(noisy_actions, flow_t, context)


class AlpamayoDiffusionTrajectoryDecoder(nn.Module):
    """Alpamayo-style diffusion trajectory decoder.

    The real Alpamayo-R1 decoder is a large diffusion action decoder conditioned
    on Cosmos-Reason outputs, visual context, and egomotion history. This module
    keeps that interface but uses our local transformer denoiser.
    """

    def __init__(
        self,
        trajectory_dim: int,
        horizon: int,
        model_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.net = ActionDenoiserTransformer(
            action_dim=trajectory_dim,
            horizon=horizon,
            model_dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            prediction_type="noise",
            conditioning_norm="adaln",
        )

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        diffusion_t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(noisy_trajectory, diffusion_t, context)
