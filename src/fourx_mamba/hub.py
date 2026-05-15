"""Hugging Face helpers for the public 4X Mamba artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

LEGACY_REPO = "aimar00/4x-mamba"
DEFAULT_REPO = "aimar00/4x-mamba-world-model"


def download_config(repo_id: str = DEFAULT_REPO, *, cache_dir: str | Path | None = None) -> Path:
    """Download `config.json` for a released checkpoint."""

    return Path(hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir))


def load_config(repo_id: str = DEFAULT_REPO, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Load model metadata without downloading the large weights file."""

    return json.loads(download_config(repo_id, cache_dir=cache_dir).read_text())


def download_weights(repo_id: str = DEFAULT_REPO, *, cache_dir: str | Path | None = None) -> Path:
    """Download `model.safetensors` for a released checkpoint."""

    return Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=cache_dir))
