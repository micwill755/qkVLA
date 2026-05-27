# qkVLA

Learning diffusion for Vision-Language-Action systems, from first principles to
robot action generation.

## Path

1. `docs/00_diffusion_math.md` - the "paper math" for continuous diffusion,
   score matching, flow matching, discrete text diffusion, and action diffusion.
2. `docs/01_vla_diffusion_map.md` - how the math maps onto GR00T, pi0.5, and
   Alpamayo-style VLA architectures.
3. `examples/01_toy_2d_diffusion.py` - a tiny DDPM trained on a 2D toy dataset.
4. `qkvla/modules/` - reusable attention, transformer, diffusion, and action
   denoising blocks.
5. `qkvla/models/` - GR00T/pi0.5/Alpamayo-style model skeletons built from the
   shared modules.

## First Experiment

Install the basics in your preferred Python environment:

```bash
pip install torch matplotlib
```

Run the toy model:

```bash
python3 examples/01_toy_2d_diffusion.py
```

It trains a small MLP denoiser on a 2D mixture of Gaussians and writes generated
samples to `artifacts/toy_2d_samples.png`.

Smoke-test the VLA model skeletons:

```bash
python3 examples/02_vla_model_skeletons.py
```

## Why This Order

Diffusion becomes much less mysterious if we climb it in this order:

1. Continuous vectors: learn how noise is added and removed.
2. Images: same math, larger tensors, usually a U-Net or DiT.
3. Text: discrete states, mask/categorical corruption, SEDD-style ratio/score
   learning instead of Gaussian noise.
4. Actions: continuous trajectories conditioned on vision, language, and robot
   state.
5. VLA: a VLM provides semantic context; a diffusion or flow action expert
   generates smooth action chunks.
