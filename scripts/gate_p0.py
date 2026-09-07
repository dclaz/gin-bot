"""Acceptance gate for Phase 0 — environment lock-in.

Asserts every bullet of IMPLEMENTATION_PLAN.md gate-p0 against the live
machine. Never edit this script to make it pass; if a threshold is wrong,
the commit changing configs/gates.yaml says old value, new value, why.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def load_spiel_facts() -> object:
    spec = importlib.util.spec_from_file_location(
        "spiel_facts", ROOT / "src" / "ginrl" / "spiel_facts.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    gates = yaml.safe_load((ROOT / "configs" / "gates.yaml").read_text())
    expected = gates["gate-p0"]

    try:
        import pyspiel  # noqa: F401
        import torch  # noqa: F401

        check("import pyspiel, torch", True)
    except ImportError as exc:
        check("import pyspiel, torch", False, str(exc))
        print(f"gate-p0 FAIL ({len(FAILURES)} checks)")
        return 1

    md = ROOT / "docs" / "ENV_FACTS.md"
    py = ROOT / "src" / "ginrl" / "spiel_facts.py"
    check("ENV_FACTS.md exists", md.exists())
    check("spiel_facts.py exists", py.exists())
    if py.exists():
        check(
            "generated facts newer than pyproject.toml",
            py.stat().st_mtime > (ROOT / "pyproject.toml").stat().st_mtime,
            "touch pyproject.toml only via probe-confirmed edits",
        )
        check(
            "ENV_FACTS.md newer than pyproject.toml",
            md.stat().st_mtime > (ROOT / "pyproject.toml").stat().st_mtime,
        )

    facts = load_spiel_facts()
    check(
        "241 actions",
        facts.NUM_DISTINCT_ACTIONS == expected["num_distinct_actions"] == 241,
        str(facts.NUM_DISTINCT_ACTIONS),
    )
    check(
        "644 observation dims",
        facts.OBSERVATION_TENSOR_SIZE == expected["observation_tensor_size"] == 644,
        str(facts.OBSERVATION_TENSOR_SIZE),
    )
    check(
        "information_state_tensor raises",
        facts.INFORMATION_STATE_TENSOR_RAISES is True,
    )
    check(
        "GinRummyUtils importable with meld API",
        "min_deadwood" in facts.GIN_RUMMY_UTILS_METHODS,
        ",".join(facts.GIN_RUMMY_UTILS_METHODS),
    )
    check("simple bot constructs", facts.SIMPLE_BOT_CONSTRUCTS is True)
    check(
        "device recommendation is cpu/mps",
        facts.DEVICE_RECOMMENDATION in expected["learner_device_allowed"],
        str(facts.DEVICE_RECOMMENDATION),
    )

    if FAILURES:
        print(f"gate-p0 FAIL ({len(FAILURES)} checks)")
        return 1
    print("gate-p0 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
