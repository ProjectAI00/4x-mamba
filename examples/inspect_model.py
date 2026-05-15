from __future__ import annotations

import sys

from fourx_mamba import DEFAULT_REPO, load_config


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    config = load_config(repo_id)

    print(f"repo: {repo_id}")
    print(f"architecture: {config.get('architecture', 'unknown')}")
    print(f"parameters: {config.get('parameter_count', 'unknown')}")
    for key in ("d_model", "n_layers", "max_state_tokens", "action_vocab_size", "token_vocab_size"):
        if key in config:
            print(f"{key}: {config[key]}")


if __name__ == "__main__":
    main()
