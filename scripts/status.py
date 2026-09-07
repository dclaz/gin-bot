"""`make status`: git sha, gates hash, env versions, device recommendation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    gates = ROOT / "configs" / "gates.yaml"
    print(f"git: {git_sha()}")
    print("phase: 0 (environment lock-in)")
    if gates.exists():
        print(f"gates.yaml sha256: {hashlib.sha256(gates.read_bytes()).hexdigest()}")
    else:
        print("gates.yaml: MISSING")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from ginrl import spiel_facts as facts  # type: ignore[import-not-found]

        print(f"spiel: {facts.OPEN_SPIEL_VERSION} torch: {facts.TORCH_VERSION}")
        print(f"device recommendation: {facts.DEVICE_RECOMMENDATION}")
    except ImportError:
        print("generated facts: MISSING (run `make probe`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
