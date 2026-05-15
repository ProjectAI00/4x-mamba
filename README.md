# 4x Mamba

Clean public release repo for the 4X strategy-game Mamba checkpoints.

This repository intentionally contains only the small public-facing code and docs. The model weights live on Hugging Face:

- Legacy model: https://huggingface.co/aimar00/4x-mamba
- Current world model: https://huggingface.co/aimar00/4x-mamba-world-model

## What Is Included

- Minimal `MambaWorldModel` architecture code
- Minimal Python helper for downloading configs and weights from Hugging Face
- Small examples for inspecting the released artifacts
- No training code
- No simulator code
- No corpus
- No AWS scripts
- No internal project files

## Models

### `aimar00/4x-mamba`

Legacy Mamba-style checkpoint.

- Parameters: `27,648,512`
- Format: `model.safetensors`
- Public artifact contains model weights only, not optimizer or scheduler state.

### `aimar00/4x-mamba-world-model`

Current 4X world model checkpoint.

- Architecture: `MambaWorldModel`
- Parameters: `28,709,251`
- Hidden size: `512`
- Layers: `8`
- Max state tokens: `512`
- Format: `model.safetensors`

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

The model implementation is in `src/fourx_mamba/model.py`. It includes only the released architecture: canonical state/action dataclasses, state encoder, action encoder, Mamba-style SSM dynamics block, and scalar policy/reward/value heads.

## Status

These are research checkpoints trained on synthetic 4X strategy-game traces. They are not official game integrations and are not packaged as production inference servers.
