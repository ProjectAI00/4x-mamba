from __future__ import annotations

import json
from pathlib import Path

import torch

from fourx_mamba import CanonicalAction, CanonicalField, CanonicalState, DEFAULT_REPO, download_weights, load_config, load_world_model


def load_example() -> tuple[CanonicalState, CanonicalAction]:
    payload = json.loads((Path(__file__).with_name("example_inputs.json")).read_text())
    state = CanonicalState(
        tokens=tuple(payload["state"]["tokens"]),
        fields=tuple(CanonicalField(**field) for field in payload["state"]["fields"]),
    )
    action = CanonicalAction(tokens=tuple(payload["action"]["tokens"]))
    return state, action


def main() -> None:
    config = load_config(DEFAULT_REPO)
    weights = download_weights(DEFAULT_REPO)
    model = load_world_model(config, weights)

    state, action = load_example()
    with torch.no_grad():
        out = model([state], [action])

    print(f"loaded: {DEFAULT_REPO}")
    print(f"state latent shape: {tuple(out['latent'].shape)}")
    print(f"next latent shape: {tuple(out['next_latent'].shape)}")
    print(f"policy: {float(out['policy'][0]):.4f}")
    print(f"reward: {float(out['reward'][0]):.4f}")
    print(f"value: {float(out['value'][0]):.4f}")


if __name__ == "__main__":
    main()
