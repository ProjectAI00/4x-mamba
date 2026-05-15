"""Minimal model code for the released 4X Mamba world-model checkpoint."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CanonicalField:
    entity_type: str
    entity_id: str
    field_name: str
    value: str


@dataclass(frozen=True)
class CanonicalState:
    tokens: tuple[int, ...] = ()
    fields: tuple[CanonicalField, ...] = ()


@dataclass(frozen=True)
class CanonicalAction:
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class MambaWorldModelConfig:
    d_model: int = 512
    n_layers: int = 8
    action_vocab_size: int = 4096
    token_vocab_size: int = 4096
    max_state_tokens: int = 512
    reward_scale: float = 100.0
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 256

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MambaWorldModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass(frozen=True)
class EntityStateEncoderConfig:
    d_model: int = 512
    type_buckets: int = 1024
    id_buckets: int = 4096
    field_buckets: int = 4096
    value_buckets: int = 8192
    token_vocab_size: int = 4096
    max_tokens: int = 512
    token_mix: float = 0.25


class EntityStateEncoder(nn.Module):
    """Encode canonical fields and fallback tokens into one latent state vector."""

    def __init__(self, config: EntityStateEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.type_embedding = nn.Embedding(config.type_buckets, config.d_model)
        self.id_embedding = nn.Embedding(config.id_buckets, config.d_model)
        self.field_embedding = nn.Embedding(config.field_buckets, config.d_model)
        self.value_embedding = nn.Embedding(config.value_buckets, config.d_model)
        self.token_embedding = nn.Embedding(config.token_vocab_size, config.d_model)
        self.field_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def _bucket(value: str, buckets: int) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % max(1, buckets)

    def _field_summary(self, fields: Sequence[CanonicalField]) -> Tensor:
        device = self.type_embedding.weight.device
        if not fields:
            return torch.zeros(self.config.d_model, device=device)
        type_ids = torch.tensor([self._bucket(f.entity_type, self.config.type_buckets) for f in fields], device=device)
        id_ids = torch.tensor([self._bucket(f.entity_id, self.config.id_buckets) for f in fields], device=device)
        field_ids = torch.tensor([self._bucket(f.field_name, self.config.field_buckets) for f in fields], device=device)
        value_ids = torch.tensor([self._bucket(f.value, self.config.value_buckets) for f in fields], device=device)
        rows = (
            self.type_embedding(type_ids)
            + self.id_embedding(id_ids)
            + self.field_embedding(field_ids)
            + self.value_embedding(value_ids)
        )
        return self.field_projection(rows).mean(dim=0)

    def _token_summary(self, state: CanonicalState) -> Tensor:
        device = self.type_embedding.weight.device
        if not state.tokens:
            return torch.zeros(self.config.d_model, device=device)
        ids = torch.tensor(
            [max(0, min(self.config.token_vocab_size - 1, int(token))) for token in state.tokens[: self.config.max_tokens]],
            device=device,
        )
        return self.token_embedding(ids).mean(dim=0)

    def forward(self, states: Sequence[CanonicalState]) -> Tensor:
        rows = [self._field_summary(state.fields) + self.config.token_mix * self._token_summary(state) for state in states]
        if not rows:
            return torch.empty((0, self.config.d_model), device=self.type_embedding.weight.device)
        return self.output_norm(torch.stack(rows, dim=0))


class _RMSNormGated(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-5, group_size: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.group_size = group_size

    def forward(self, x: Tensor, z: Tensor | None = None) -> Tensor:
        dtype = x.dtype
        x = x.float()
        if z is not None:
            x = x * F.silu(z.float())
        if self.group_size is None:
            out = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        else:
            groups = x.shape[-1] // self.group_size
            grouped = x.reshape(*x.shape[:-1], groups, self.group_size)
            out = (grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)).reshape_as(x)
        return (out * self.weight.float()).to(dtype)


class PurePyTorchSSM(nn.Module):
    """Small pure-PyTorch Mamba2-style block matching the released weights."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, headdim: int, ngroups: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, bias=True, kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = _RMSNormGated(self.d_inner, group_size=self.d_inner // self.ngroups)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _scan(self, x: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        batch, seqlen, _, _ = x.shape
        group_idx = torch.arange(self.nheads, device=x.device) // (self.nheads // self.ngroups)
        b_heads = b.index_select(2, group_idx)
        c_heads = c.index_select(2, group_idx)
        state = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=x.device, dtype=torch.float32)
        outputs: list[Tensor] = []
        for idx in range(seqlen):
            decay = torch.exp(dt[:, idx].float() * a).unsqueeze(-1).unsqueeze(-1)
            drive = torch.einsum("bh,bhp,bhn->bhpn", dt[:, idx].float(), x[:, idx].float(), b_heads[:, idx].float())
            state = state * decay + drive
            y_t = torch.einsum("bhpn,bhn->bhp", state, c_heads[:, idx].float())
            y_t = y_t + self.D.float().view(1, self.nheads, 1) * x[:, idx].float()
            outputs.append(y_t.to(x.dtype))
        return torch.stack(outputs, dim=1)

    def forward(self, u: Tensor) -> Tensor:
        batch, seqlen, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, x_b_c, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        x_b_c = self.act(self.conv1d(x_b_c.transpose(1, 2)).transpose(1, 2)[:, :seqlen])
        x, b, c = torch.split(x_b_c, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
        y = self._scan(
            x.reshape(batch, seqlen, self.nheads, self.headdim),
            dt,
            -torch.exp(self.A_log.float()),
            b.reshape(batch, seqlen, self.ngroups, self.d_state),
            c.reshape(batch, seqlen, self.ngroups, self.d_state),
        ).reshape(batch, seqlen, self.d_inner)
        return self.out_proj(self.norm(y, z))


class MambaWorldModel(nn.Module):
    """Released 4X world model: state/action encoders, SSM dynamics, scalar heads."""

    def __init__(self, config: MambaWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.state_encoder = EntityStateEncoder(
            EntityStateEncoderConfig(
                d_model=config.d_model,
                token_vocab_size=config.token_vocab_size,
                max_tokens=config.max_state_tokens,
            )
        )
        self.action_embedding = nn.Embedding(config.action_vocab_size, config.d_model)
        self.action_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.dynamics_backbone = nn.ModuleList(
            [
                PurePyTorchSSM(
                    d_model=config.d_model,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    headdim=config.mamba_headdim,
                    ngroups=config.mamba_ngroups,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.dynamics_norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.n_layers)])
        self.latent_norm = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Sequential(nn.Linear(config.d_model * 2, config.d_model), nn.SiLU(), nn.Linear(config.d_model, 1))
        self.reward_head = nn.Sequential(nn.Linear(config.d_model * 2, config.d_model), nn.SiLU(), nn.Linear(config.d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.SiLU(), nn.Linear(config.d_model, 1))

    def encode(self, states: CanonicalState | Sequence[CanonicalState]) -> Tensor:
        rows = [states] if isinstance(states, CanonicalState) else list(states)
        return self.state_encoder(rows)

    def encode_action(self, actions: CanonicalAction | Sequence[CanonicalAction]) -> Tensor:
        rows = [actions] if isinstance(actions, CanonicalAction) else list(actions)
        if not rows:
            return torch.empty((0, self.action_embedding.embedding_dim), device=self.action_embedding.weight.device)
        max_len = max(1, max(len(action.tokens) for action in rows))
        ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.action_embedding.weight.device)
        mask = torch.zeros((len(rows), max_len), dtype=torch.float, device=self.action_embedding.weight.device)
        vocab_max = self.action_embedding.num_embeddings - 1
        for row_idx, action in enumerate(rows):
            clipped = [max(0, min(vocab_max, int(token))) for token in action.tokens[:max_len]] or [0]
            ids[row_idx, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=ids.device)
            mask[row_idx, : len(clipped)] = 1.0
        embedded = self.action_embedding(ids)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.action_projection(pooled)

    def predict_next_latent(self, latent: Tensor, action_latent: Tensor) -> Tensor:
        sequence = torch.stack((latent, action_latent), dim=1)
        hidden = sequence
        for block, norm in zip(self.dynamics_backbone, self.dynamics_norms, strict=True):
            hidden = norm(hidden + block(hidden))
        return self.latent_norm(latent + hidden[:, -1])

    def forward(self, states: Sequence[CanonicalState], actions: Sequence[CanonicalAction]) -> dict[str, Tensor]:
        latent = self.encode(states)
        action_latent = self.encode_action(actions)
        next_latent = self.predict_next_latent(latent, action_latent)
        pair = torch.cat((next_latent, action_latent), dim=-1)
        return {
            "latent": latent,
            "next_latent": next_latent,
            "policy": self.policy_head(pair).squeeze(-1),
            "reward": self.reward_head(pair).squeeze(-1),
            "value": self.value_head(next_latent).squeeze(-1),
        }


def load_world_model(config: dict[str, Any], weights_path: str | Path, *, map_location: str = "cpu") -> MambaWorldModel:
    """Instantiate `MambaWorldModel` and load released `model.safetensors` weights."""

    model = MambaWorldModel(MambaWorldModelConfig.from_dict(config))
    state = load_file(str(weights_path), device=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
