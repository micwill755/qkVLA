# From Papers To Code

This repo will build simplified versions of the model families we are studying.
The goal is not to clone production checkpoints. The goal is to reconstruct the
architectural ideas from scratch in small PyTorch modules.

## Paper Anchors

These skeletons are inspired by public architecture descriptions:

```text
GR00T N1:
  dual-system VLA with VLM context and an action flow-matching DiT module

pi0.5:
  VLA with flow-matching action expert, adaRMS-style time conditioning, and
  heterogeneous co-training signals

Alpamayo / Alpamayo-R1:
  reasoning-first VLA with egomotion/history context and a diffusion trajectory
  decoder
```

## Exactness Boundary

We can make the local code exact only where the public sources expose enough
detail. The current files now follow the published architecture patterns, but
they still use local placeholder backbones instead of production-scale external
models:

```text
GR00T N1 production backbone:
  NVIDIA-Eagle / SmolLM-style VLM components are represented by ToyVLMContext.

pi0.5 production backbone:
  PaliGemma/Gemma/OpenPI-style VLM and projections are represented by
  ToyVLMContext plus Pi05FlowActionExpert.

Alpamayo production backbone:
  Cosmos-Reason and multi-camera video preprocessing are represented by
  ToyVLMContext plus egomotion-history tokens.
```

The action-generation side is where this repo is closest to the papers:

```text
GR00T-style: action flow-matching DiT expert
pi0.5-style: flow-matching action expert with adaRMS-style time conditioning
Alpamayo-style: diffusion trajectory decoder conditioned on reasoning/history
```

## Reference Code Inspected

The local implementation was updated after inspecting these public files:

```text
Physical-Intelligence/openpi
  src/openpi/models/pi0_config.py
  src/openpi/models/pi0.py
  src/openpi/models/gemma.py

NVIDIA/Isaac-GR00T
  gr00t/configs/model/gr00t_n1d7.py
  gr00t/model/gr00t_n1d7/gr00t_n1d7.py
  gr00t/model/modules/dit.py
  gr00t/model/modules/flowmatching_modules.py
  gr00t/model/modules/embodiment_conditioned_mlp.py

autowarefoundation/alpamayo-autoware
  src/alpamayo1_5/models/alpamayo1_5.py
  src/alpamayo1_5/models/action_in_proj.py
  src/alpamayo1_5/diffusion/flow_matching.py
```

Important mechanics copied into our local design:

```text
OpenPI pi0/pi0.5:
  pi0.5 places state in discrete language tokens.
  pi0.5 uses adaRMSNorm to inject the flow timestep.
  flow path: x_t = t * noise + (1 - t) * actions in OpenPI code.
  target: u_t = noise - actions.
  sampling integrates from noise at t=1 toward actions at t=0.

GR00T N1.7:
  Cosmos/Qwen-style VLM backbone feeds a DiT/AlternateVLDiT action head.
  state encoder is embodiment-specific.
  action encoder is multi-embodiment and includes timestep features.
  flow path: noisy = (1 - t) * noise + t * actions.
  target: velocity = actions - noise.
  inference uses Euler integration from random noise toward actions.

Alpamayo 1.5:
  reasoning VLA builds an expert from the VLM text config.
  action_in_proj consumes noisy action x and diffusion/flow time t.
  expert denoiser runs over future trajectory/action tokens.
  action_out_proj maps expert hidden states back to trajectory action space.
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
  noisy action chunk + flow timestep
  cross-attend to context tokens
  predict action velocity
```

This mirrors the dual-system idea: semantic context first, fast action
flow-matching generation second.

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
  adaRMS-style timestep conditioning
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
  image/language/egomotion-history tokens

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
