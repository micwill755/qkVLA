# From Papers To Code

This repo will build simplified versions of the model families we are studying.
The goal is not to clone production checkpoints. The goal is to reconstruct the
architectural ideas from scratch in small PyTorch modules.

## Paper Anchors

These skeletons are inspired by public architecture descriptions:

```text
GR00T N1:
  dual-system VLA with VLM context and a diffusion transformer action module

pi0.5:
  VLA with flow-matching action expert and heterogeneous co-training signals

Alpamayo / Alpamayo-R1:
  reasoning-first VLA with a diffusion trajectory/action expert
```

## Shared Modules

The model files should stay thin. Reusable objects live under `qkvla/modules/`:

```text
attention.py        self-attention and cross-attention
transformer.py      transformer blocks and adaptive LayerNorm blocks
embeddings.py       timestep embeddings
diffusion.py        DDPM and flow-matching utilities
action_denoiser.py  transformer action expert
encoders.py         toy image, text, and proprio encoders
```

## GR00T-Style Skeleton

File:

```text
qkvla/models/groot.py
```

Code idea:

```text
System 2 context encoder:
  image tokens + language tokens + proprio token

System 1 action expert:
  noisy action chunk + diffusion timestep
  cross-attend to context tokens
  predict action noise
```

This mirrors the dual-system idea: semantic context first, fast diffusion action
generation second.

## pi0.5-Style Skeleton

File:

```text
qkvla/models/pi05.py
```

Code idea:

```text
context encoder:
  image/language/proprio tokens

flow action expert:
  noisy action chunk + continuous flow time
  predict velocity field

extra supervision heads:
  semantic subtask logits as a placeholder for heterogeneous co-training
```

This gives us a place to add pi0.5-style mixed objectives later without mixing
them into the action expert.

## Alpamayo-Style Skeleton

File:

```text
qkvla/models/alpamayo.py
```

Code idea:

```text
context encoder:
  image/language/history tokens

reasoning bridge:
  small set of reasoning tokens cross-attends to context

trajectory expert:
  noisy trajectory + diffusion timestep
  cross-attend to context + reasoning tokens
  predict trajectory noise
```

This models the reason-then-denoise pattern: reasoning tokens become additional
conditioning for the trajectory diffusion expert.

## Next Build Steps

1. Move the toy 2D DDPM example onto `qkvla/modules/diffusion.py`.
2. Add a synthetic reaching dataset for action diffusion.
3. Train `ActionDenoiserTransformer` on synthetic action chunks.
4. Add a tiny image diffusion model.
5. Add a tiny SEDD-inspired text diffusion model.
6. Replace toy encoders with real backbones once the small models work.
