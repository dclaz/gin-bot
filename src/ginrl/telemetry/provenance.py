"""Provenance helpers: every artifact carries git SHA, config and facts hashes."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git_sha() -> str:
    """Short HEAD SHA, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spiel_facts_hash() -> str:
    return file_sha256(Path(__file__).resolve().parents[1] / "spiel_facts.py")
