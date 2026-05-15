from __future__ import annotations

from fourx_mamba import CanonicalAction, CanonicalState, DEFAULT_REPO, download_weights, load_config, load_world_model


def main() -> None:
    config = load_config(DEFAULT_REPO)
    weights = download_weights(DEFAULT_REPO)
    model = load_world_model(config, weights)

    state = CanonicalState(tokens=(1, 2, 3, 4))
    action = CanonicalAction(tokens=(10, 20, 30))
    out = model([state], [action])

    print(f"loaded: {DEFAULT_REPO}")
    print(f"policy: {float(out['policy'][0]):.4f}")
    print(f"reward: {float(out['reward'][0]):.4f}")
    print(f"value: {float(out['value'][0]):.4f}")


if __name__ == "__main__":
    main()
