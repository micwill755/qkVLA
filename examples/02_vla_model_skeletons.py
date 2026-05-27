"""Smoke-test the GR00T/pi0.5/Alpamayo-style model skeletons."""

from __future__ import annotations

import torch

from qkvla.models import AlpamayoStyleVLA, GR00TStyleVLA, Pi05StyleVLA


def fake_batch() -> dict[str, torch.Tensor]:
    batch = 2
    horizon = 8
    action_dim = 7
    return {
        "images": torch.randn(batch, 3, 32, 32),
        "token_ids": torch.randint(0, 128, (batch, 12)),
        "proprio": torch.randn(batch, 10),
        "egomotion_history": torch.randn(batch, 4, 12),
        "noisy_actions": torch.randn(batch, horizon, action_dim),
        "t": torch.randint(0, 100, (batch,)),
    }


def main() -> None:
    torch.manual_seed(7)
    batch = fake_batch()
    kwargs = {
        "image_channels": 3,
        "patch_size": 8,
        "vocab_size": 128,
        "proprio_dim": 10,
        "action_dim": 7,
        "horizon": 8,
        "model_dim": 128,
    }

    groot = GR00TStyleVLA(**kwargs)
    pi05 = Pi05StyleVLA(**kwargs)
    alpamayo = AlpamayoStyleVLA(**kwargs)

    groot_out = groot(
        batch["images"],
        batch["token_ids"],
        batch["proprio"],
        batch["noisy_actions"],
        batch["t"].float() / 100,
    )
    pi05_out = pi05(
        batch["images"],
        batch["token_ids"],
        batch["proprio"],
        batch["noisy_actions"],
        batch["t"].float() / 100,
    )
    alpamayo_out = alpamayo(
        batch["images"],
        batch["token_ids"],
        batch["proprio"],
        batch["egomotion_history"],
        batch["noisy_actions"],
        batch["t"],
    )

    print("GR00T-style noise:", tuple(groot_out.shape))
    print("pi0.5-style velocity:", tuple(pi05_out["velocity"].shape))
    print("pi0.5 subtask logits:", tuple(pi05_out["subtask_logits"].shape))
    print("Alpamayo-style noise:", tuple(alpamayo_out["pred_noise"].shape))
    print("Alpamayo reasoning:", tuple(alpamayo_out["reasoning_tokens"].shape))


if __name__ == "__main__":
    main()
