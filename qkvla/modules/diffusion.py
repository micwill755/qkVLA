from __future__ import annotations

import torch
from torch import nn


class DDPMScheduler(nn.Module):
    """Small DDPM scheduler for continuous data."""

    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = gather_by_timestep(self.alpha_bars, t, x0.ndim)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        context: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        device = device or self.betas.device
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            pred_noise = model(x, t, context) if context is not None else model(x, t)
            mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            x = mean if step == 0 else mean + beta.sqrt() * torch.randn_like(x)
        return x


class FlowMatchingPath:
    """Linear flow matching path between noise and clean data.

    This follows the GR00T/OpenPI implementation convention:
    x_t = (1 - t) * noise + t * data, with target velocity data - noise.
    """

    def sample_path(
        self, data: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(data) if noise is None else noise
        t_view = t.view(t.shape[0], *([1] * (data.ndim - 1)))
        xt = (1.0 - t_view) * noise + t_view * data
        velocity = data - noise
        return xt, velocity

    @torch.no_grad()
    def euler_sample(
        self,
        step_fn,
        shape: tuple[int, ...],
        num_steps: int = 10,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        device = device or "cpu"
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        for idx in range(num_steps):
            t = torch.full((shape[0],), idx / num_steps, device=device, dtype=x.dtype)
            x = x + dt * step_fn(x, t)
        return x


class OpenPIFlowMatchingPath:
    """OpenPI pi0/pi0.5 flow convention.

    OpenPI trains with x_t = t * noise + (1 - t) * actions and target
    u_t = noise - actions, then samples from t=1 down to t=0.
    """

    def sample_path(
        self, data: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(data) if noise is None else noise
        t_view = t.view(t.shape[0], *([1] * (data.ndim - 1)))
        xt = t_view * noise + (1.0 - t_view) * data
        velocity = noise - data
        return xt, velocity

    @torch.no_grad()
    def euler_sample(
        self,
        step_fn,
        shape: tuple[int, ...],
        num_steps: int = 10,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        device = device or "cpu"
        x = torch.randn(shape, device=device)
        dt = -1.0 / num_steps
        for idx in range(num_steps):
            t = torch.full((shape[0],), 1.0 + idx * dt, device=device, dtype=x.dtype)
            x = x + dt * step_fn(x, t)
        return x


def gather_by_timestep(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    out = values[t]
    return out.view(t.shape[0], *([1] * (ndim - 1)))
