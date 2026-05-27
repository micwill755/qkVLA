# Diffusion In VLA Architectures

This note maps our learning path onto three architectures we care about:
GR00T, pi0.5, and Alpamayo.

## Common Pattern

The shared design is:

```text
observations + instruction
        |
        v
vision-language backbone / reasoner
        |
        v
action expert conditioned on VLM context
        |
        v
diffusion or flow denoising over action chunks / trajectories
```

This is why we are learning diffusion before VLA implementation. The action head
is not merely a regression layer; it is a conditional generative model over
future behavior.

## GR00T N1

GR00T N1 is described as a VLA model with a dual-system architecture:

```text
System 2: vision-language module
System 1: diffusion transformer for motor actions
```

The important lesson for this repo:

```text
VLM tokens provide semantic context.
Diffusion transformer produces temporally smooth motor actions.
The system is trained on heterogeneous robot, video, and synthetic data.
```

Our simplified version should become:

```text
image encoder + text encoder + proprio encoder
        -> context tokens
DiT-style action denoiser
        -> H-step action chunk
```

## pi0.5

pi0.5 builds on pi0 and emphasizes open-world generalization through co-training
on heterogeneous examples:

```text
image observations
language commands
object detections
semantic subtask prediction
low-level actions
multiple robots and web-scale knowledge sources
```

The important lesson:

```text
The action generator is only one part of the recipe.
Generalization comes from mixing low-level action learning with higher-level
semantic supervision and broad visual-language knowledge.
```

For this repo, pi0.5 suggests a future training interface where examples can be
one of several types:

```text
VQA / caption / object detection
semantic subtask labels
robot action trajectories
```

At first, we can fake the VLM context with small embeddings and focus on the
action diffusion math.

## Alpamayo

Alpamayo-style reasoning VLA separates:

```text
Phase 1: VLM generates reasoning text.
Phase 2: expert model plus diffusion sampler generates trajectory predictions.
```

The important lesson:

```text
Reasoning text can be an interface between perception/language and action.
The trajectory expert consumes continuous noisy actions through projection
layers, not token embeddings.
```

This gives us a clean toy architecture:

```text
prompt/image/history -> reasoning tokens
reasoning tokens + noisy trajectory + timestep -> expert transformer
expert output -> predicted noise / velocity
diffusion sampler -> trajectory
```

## Project Milestones

1. Continuous diffusion math and a 2D DDPM.
2. Tiny image diffusion on MNIST or CIFAR-sized images.
3. Tiny discrete text diffusion inspired by SEDD.
4. Conditional action diffusion on synthetic reaching trajectories.
5. VLA skeleton: frozen toy vision/text encoders plus action diffusion head.
6. Alpamayo-style reasoning bridge: text reasoner context into trajectory
   expert.
7. GR00T/pi0.5-style action transformer: context tokens plus flow/diffusion
   action chunk generation.

