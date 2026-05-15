# 4X Mamba

Clean public release repo for the 4X strategy-game Mamba world model.

This repo is intentionally small. It contains the model architecture, lightweight loading helpers, and examples. The actual weights live on Hugging Face:

- Current world model: https://huggingface.co/aimar00/4x-mamba-world-model
- Legacy checkpoint: https://huggingface.co/aimar00/4x-mamba

## What Is Included

- `MambaWorldModel` architecture code
- Canonical state/action dataclasses
- Hash/token state encoder
- Action token encoder
- Mamba-style SSM dynamics block
- Latent transition path
- Policy, reward, and value heads
- Hugging Face config/weights download helpers
- Small load/inspect examples

## What Is Not Included

- No training code
- No simulator code
- No corpus
- No infrastructure scripts
- No benchmark harness
- No unrelated project files

## Current Model

Hugging Face repo: `aimar00/4x-mamba-world-model`

- Architecture: `MambaWorldModel`
- Parameters: `28,709,251`
- Hidden size: `512`
- Layers: `8`
- Max state tokens: `512`
- Format: `model.safetensors`

The model predicts in latent space. It encodes a state and action, applies Mamba-style SSM dynamics, predicts the next latent, and scores actions with policy/reward/value heads.

## Legacy Checkpoint

Hugging Face repo: `aimar00/4x-mamba`

- Parameters: `27,648,512`
- Format: `model.safetensors`
- Public artifact contains model weights only, not optimizer or scheduler state.

The legacy checkpoint is kept for reference. The architecture code in this repo targets the current `MambaWorldModel`.

## PyTorch And CUDA

The public implementation is ordinary PyTorch so it is easy to inspect and run. PyTorch can run it on CPU or CUDA depending on your local install:

```python
model = load_world_model(config, weights)
model = model.cuda()
```

The release repo does not vendor the fused Mamba/Triton backend. That keeps the public code clean and portable; it is not meant to be the fastest possible inference package.

## Quick Start

```bash
pip install -e .
python examples/inspect_model.py aimar00/4x-mamba-world-model
```

This downloads the config and prints the model metadata. It does not download the large weights unless requested.

```bash
python examples/download_weights.py aimar00/4x-mamba-world-model
```

Load the current world model:

```bash
python examples/load_world_model.py
```

That example reads `examples/example_inputs.json`, builds a `CanonicalState` and `CanonicalAction`, runs them through the state encoder/action encoder, and prints latent/head outputs.

The model implementation is in `src/fourx_mamba/model.py`. It includes only the released architecture: canonical state/action dataclasses, state encoder, action encoder, Mamba-style SSM dynamics block, and scalar policy/reward/value heads.

## State Encoder

The released state encoder is wired into `MambaWorldModel`.

It accepts two input channels:

- `tokens`: integer tokens, clipped into the configured `token_vocab_size`
- `fields`: structured entity fields with `entity_type`, `entity_id`, `field_name`, and `value`

Fields are hash-bucketed into learned embeddings, combined with token embeddings, projected, and normalized into one latent vector per state.

## Status

These are research checkpoints trained on synthetic 4X strategy-game traces. They are packaged for inspection and lightweight loading, not production serving.
