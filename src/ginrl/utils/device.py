"""Learner device selection (AGENTS.md contract).

The learner runs on the benchmarked device from `spiel_facts`
(`DEVICE_RECOMMENDATION`, written by `make probe` at the model size actually
being trained). Actors always run on CPU. Call once, in the learner process,
before any CUDA/MPS context exists — never fork after initialising an MPS
context (landmine 9).
"""

from __future__ import annotations

import torch

from ginrl import spiel_facts

CPU = "cpu"


def pick_device(prefer: str = "auto") -> torch.device:
    """Return the learner device.

    `prefer="auto"` follows the probe benchmark; an explicit `"cpu"` /
    `"mps"` / `"cuda"` is honoured when available and raises when it is not
    (fail loud beats silently training on the wrong device).
    """
    if prefer != "auto":
        if prefer == CPU:
            return torch.device(CPU)
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        raise ValueError(f"requested device {prefer!r} is not available")
    rec = getattr(spiel_facts, "DEVICE_RECOMMENDATION", CPU)
    if rec == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if rec == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(CPU)
