"""Train a tiny DDPM on a 2D mixture of Gaussians.

This is deliberately small: the point is to make the diffusion equations in
docs/00_diffusion_math.md executable without image or robotics complexity.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(batch_size: int) -> torch.Tensor:
    """Eight Gaussian modes around a circle."""
    angles = torch.randint(0, 8, (batch_size,), device=DEVICE) * (2 * math.pi / 8)
    centers = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * 2.0
    return centers + 0.08 * torch.randn(batch_size, 2, device=DEVICE)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Denoiser(nn.Module):
    def __init__(self, time_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.time_dim = time_dim

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        return self.net(torch.cat((x_t, t_emb), dim=-1))


class Diffusion:
    def __init__(self, timesteps: int = 200) -> None:
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=DEVICE)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t][:, None]
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int) -> torch.Tensor:
        model.eval()
        x = torch.randn(n, 2, device=DEVICE)
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=DEVICE, dtype=torch.long)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            pred_noise = model(x, t)
            mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        model.train()
        return x


def train() -> None:
    torch.manual_seed(7)
    diffusion = Diffusion()
    model = Denoiser().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for step in range(3_000):
        x0 = make_batch(512)
        t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=DEVICE)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(f"step={step:04d} loss={loss.item():.4f}")

    save_plot(diffusion.sample(model, 2_000).cpu())


def save_plot(samples: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5)
    plt.axis("equal")
    plt.title("Toy 2D DDPM Samples")
    plt.tight_layout()
    path = out_dir / "toy_2d_samples.png"
    plt.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    train()

