"""Gin torso bake-off (Phase 4).

Five torsos, one head contract: every GinNet maps a concatenated gin
observation to ``(policy_logits[56], value, aux_logits[deck])`` through a
torso that outputs a 128-wide hidden vector, then fixed heads. Only the
torso varies; the policy/value/opponent-hand heads are fixed.

Gin observation layout (built by the collector)::

    [belief features F][event tokens K as floats][raw tensor 644]

Card blocks are the first ``11 * deck`` belief dims (11 blocks, then 14
scalars); each torso selects its view of this shared observation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ginrl.env.features import EVENT_PAD, EVENT_VOCAB
from ginrl.nets.actor_critic import LOGIT_FLOOR, ResidualBlock

GIN_TORSOS = ("mlp", "set", "seq", "raw", "hybrid")
N_CARD_BLOCKS = 11
N_SCALARS = 14
RAW_DIM = 644


def gin_obs_slices(
    feat_dim: int, event_window: int, raw_dim: int = RAW_DIM
) -> tuple[slice, slice, slice]:
    """Belief/event/raw slices of the concatenated gin observation."""
    return (
        slice(0, feat_dim),
        slice(feat_dim, feat_dim + event_window),
        slice(feat_dim + event_window, feat_dim + event_window + raw_dim),
    )


def _mlp(in_dim: int, hidden: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(max(0, layers - 1)):
        mods += [ResidualBlock(hidden), nn.ReLU()]
    return nn.Sequential(*mods)


class MlpTorso(nn.Module):
    """A: residual MLP over the flat belief features."""

    def __init__(self, feat_dim: int, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(feat_dim, hidden, layers)

    def forward(
        self, feat: torch.Tensor, events: torch.Tensor, raw: torch.Tensor, deck: int
    ) -> torch.Tensor:
        return self.net(feat)


class SetTorso(nn.Module):
    """B: set encoder over card embeddings.

    Each present (block, card) pair contributes ``card_emb + type_emb``;
    the torso mean-pools over the SET of present pairs (permutation
    invariant), then mixes with the scalar features.
    """

    def __init__(self, feat_dim: int, deck: int, hidden: int = 128) -> None:
        super().__init__()
        self.deck = deck
        self.card_emb = nn.Embedding(deck, 64)
        self.type_emb = nn.Embedding(N_CARD_BLOCKS, 64)
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(128, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(
        self, feat: torch.Tensor, events: torch.Tensor, raw: torch.Tensor, deck: int
    ) -> torch.Tensor:
        blocks = feat[:, : N_CARD_BLOCKS * self.deck].view(-1, N_CARD_BLOCKS, self.deck)
        scalars = feat[:, N_CARD_BLOCKS * self.deck :]
        present = blocks.clamp(min=0.0)
        count = present.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(-1)
        key = self.card_emb.weight.unsqueeze(0) + self.type_emb.weight.unsqueeze(1)
        pooled = (present.unsqueeze(-1) * key).sum(dim=(1, 2)) / count
        return self.mix(torch.cat([pooled, self.scalars(scalars)], dim=-1))


class SeqTorso(nn.Module):
    """C: sequence encoder over the public action history.

    A GRU reads the recent-event window; its final state mixes with an MLP
    of the current belief features (history alone cannot see the hand).
    """

    def __init__(self, feat_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(EVENT_VOCAB, 64, padding_idx=EVENT_PAD)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.state = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(128, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(
        self, feat: torch.Tensor, events: torch.Tensor, raw: torch.Tensor, deck: int
    ) -> torch.Tensor:
        _, last = self.gru(self.tok_emb(events.long()))
        return self.mix(torch.cat([last.squeeze(0), self.state(feat)], dim=-1))


class RawTorso(nn.Module):
    """D: no-features control — residual MLP on the raw 644-dim tensor."""

    def __init__(self, raw_dim: int = RAW_DIM, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.net = _mlp(raw_dim, hidden, layers)

    def forward(
        self, feat: torch.Tensor, events: torch.Tensor, raw: torch.Tensor, deck: int
    ) -> torch.Tensor:
        return self.net(raw)


class HybridTorso(nn.Module):
    """E: GRU event prefix + attention over the set-encoded hand.

    The GRU state queries the card-set keys (same encoding as SetTorso);
    attention output mixes with the GRU state and scalar features.
    """

    def __init__(self, feat_dim: int, deck: int, hidden: int = 128) -> None:
        super().__init__()
        self.deck = deck
        self.tok_emb = nn.Embedding(EVENT_VOCAB, 64, padding_idx=EVENT_PAD)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.card_emb = nn.Embedding(deck, 64)
        self.type_emb = nn.Embedding(N_CARD_BLOCKS, 64)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(192, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def _keys(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = feat[:, : N_CARD_BLOCKS * self.deck].view(-1, N_CARD_BLOCKS, self.deck)
        present = blocks.clamp(min=0.0) > 0.5
        keys = self.card_emb.weight.unsqueeze(0) + self.type_emb.weight.unsqueeze(1)
        keys = keys.expand(feat.shape[0], -1, -1, -1).flatten(1, 2)
        return keys, ~present.flatten(1)

    def forward(
        self, feat: torch.Tensor, events: torch.Tensor, raw: torch.Tensor, deck: int
    ) -> torch.Tensor:
        _, last = self.gru(self.tok_emb(events.long()))
        query = last.squeeze(0).unsqueeze(1)
        keys, pad_mask = self._keys(feat)
        attended, _ = self.attn(query, keys, keys, key_padding_mask=pad_mask)
        scalars = feat[:, N_CARD_BLOCKS * self.deck :]
        out = torch.cat([attended.squeeze(1), last.squeeze(0), self.scalars(scalars)], dim=-1)
        return self.mix(out)


TORSOS: dict[str, type] = {
    "mlp": MlpTorso,
    "set": SetTorso,
    "seq": SeqTorso,
    "raw": RawTorso,
    "hybrid": HybridTorso,
}


class GinNet(nn.Module):
    """One torso + fixed policy/value/opponent-hand heads for gin.

    ``forward(obs, mask)`` matches the trainer's call shape: the
    concatenated observation is split into torso views internally.
    """

    def __init__(
        self,
        torso: str = "mlp",
        *,
        feat_dim: int,
        deck: int,
        n_actions: int = 56,
        hidden: int = 128,
        layers: int = 2,
        raw_dim: int = RAW_DIM,
        event_window: int = 32,
    ) -> None:
        super().__init__()
        if torso not in TORSOS:
            raise ValueError(f"unknown torso {torso!r}; expected one of {GIN_TORSOS}")
        self.torso_name = torso
        self.deck = deck
        self.slices = gin_obs_slices(feat_dim, event_window, raw_dim)
        self.torso: nn.Module
        if torso in ("set", "hybrid"):
            self.torso = TORSOS[torso](feat_dim, deck, hidden)
        elif torso == "seq":
            self.torso = TORSOS[torso](feat_dim, hidden)
        elif torso == "raw":
            self.torso = TORSOS[torso](raw_dim, hidden, layers)
        else:
            self.torso = TORSOS[torso](feat_dim, hidden, layers)
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
        self.aux_head = nn.Linear(hidden, deck)

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_slice, e_slice, r_slice = self.slices
        h = self.torso(obs[:, f_slice], obs[:, e_slice], obs[:, r_slice], self.deck)
        logits = self.policy_head(h).masked_fill(~mask, LOGIT_FLOOR)
        return logits, self.value_head(h).squeeze(-1), self.aux_head(h)

    @property
    def obs_dim(self) -> int:
        return self.slices[2].stop


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
