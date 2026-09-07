"""Cross-file drift guard.

Every defect in the 2026-09-07 review was cross-file drift: a gate renamed in
one file but not another, a doc citing a target that does not exist, a ruff
rule that no longer exists. This module catches that class.

Rule: parse structure (Makefile non-comment lines, TOML, YAML, Markdown code
spans/blocks) — never grep prose, or explanations get read as rules.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
GATES_YAML = ROOT / "configs" / "gates.yaml"
PYPROJECT = ROOT / "pyproject.toml"

DOCS = [
    ROOT / "README.md",
    ROOT / "METHODOLOGY.md",
    ROOT / "IMPLEMENTATION_PLAN.md",
    ROOT / "AGENTS.md",
    *sorted((ROOT / "docs").glob("*.md")),
]

GATE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:(?!.*=)")
MAKE_CITE_RE = re.compile(r"make\s+([a-z0-9][a-zA-Z0-9-]*)")

# METHODOLOGY.md section 5.1, Tier 1: gated, always on.
TIER1_TOOLS = [
    "points_per_hand",
    "fit_ratings",
    "cyclic_fraction",
    "nash_averaging",
    "rlbr",
]


def make_targets() -> set[str]:
    """Target names defined on Makefile non-comment lines."""
    targets: set[str] = set()
    for line in MAKEFILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = GATE_TARGET_RE.match(stripped)
        if match and match.group(1) != ".PHONY":
            targets.add(match.group(1))
    return targets


def doc_cited_targets() -> set[str]:
    """`make <target>` tokens inside Markdown code spans and fenced blocks only.

    Prose ("make it work", "make the call") is not code and is ignored.
    """
    cited: set[str] = set()
    for doc in DOCS:
        if not doc.exists():
            continue
        lines = doc.read_text().splitlines()
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            spans = re.findall(r"`([^`]*)`", line) if not in_fence else [line]
            for span in spans:
                for token in MAKE_CITE_RE.findall(span):
                    cited.add(token)
    return cited


def test_gates_yaml_keys_match_makefile_gate_targets() -> None:
    gates = yaml.safe_load(GATES_YAML.read_text())
    yaml_keys = {k for k in gates if re.fullmatch(r"gate-p\d+", str(k))}
    make_gates = {t for t in make_targets() if re.fullmatch(r"gate-p\d+", t)}
    assert yaml_keys - make_gates == set(), (
        f"gates.yaml keys missing in Makefile: {yaml_keys - make_gates}"
    )
    assert make_gates - yaml_keys == set(), (
        f"Makefile gates missing in gates.yaml: {make_gates - yaml_keys}"
    )


def test_doc_cited_make_targets_exist() -> None:
    targets = make_targets()
    missing: dict[str, str] = {}
    for token in doc_cited_targets():
        if token == "gate-pN":  # pattern shorthand for the whole gate family
            if not any(re.fullmatch(r"gate-p\d+", t) for t in targets):
                missing[token] = "no gate-pN targets defined"
        elif token not in targets:
            missing[token] = "cited in docs but not a Makefile target"
    assert missing == {}, f"doc citations without targets: {missing}"


def test_ruff_ignore_list_has_no_removed_rules() -> None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    config = tomllib.loads(PYPROJECT.read_text())
    lint = config.get("tool", {}).get("ruff", {}).get("lint", {})
    codes = list(lint.get("select", [])) + list(lint.get("ignore", []))
    assert codes, "expected a non-empty ruff select/ignore list to audit"
    # A select/ignore entry may be a bare prefix ("E") or a full code
    # ("E501"); `ruff rule` only resolves full codes, so probe each entry
    # through `ruff check --select` on an empty file instead. Exit 0 means
    # the selector resolved; anything else means ruff rejected it.
    probe = ROOT / ".ruff_selector_probe.py"
    probe.write_text("")
    try:
        unknown = []
        for code in codes:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "ruff",
                    "check",
                    "--isolated",
                    "--no-cache",
                    "--select",
                    str(code),
                    str(probe),
                ],
                capture_output=True,
                cwd=ROOT,
            )
            if result.returncode != 0:
                unknown.append(code)
    finally:
        probe.unlink(missing_ok=True)
    assert unknown == [], f"ruff selectors unknown to installed ruff: {unknown}"


def _walk(node: object, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str, object]]:
    """Yield (path, key-or-index, value) for every mapping entry recursively."""
    out: list[tuple[tuple[str, ...], str, object]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.append((path, str(key), value))
            out.extend(_walk(value, path + (str(key),)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_walk(value, path + (str(i),)))
    return out


def test_tier1_tools_are_gated() -> None:
    text = GATES_YAML.read_text().lower()
    for tool in TIER1_TOOLS:
        assert tool in text, f"Tier 1 tool '{tool}' (METHODOLOGY section 5.1) has no gate threshold"


def test_alpha_rank_is_tier3_and_ungated() -> None:
    gates = yaml.safe_load(GATES_YAML.read_text())
    alpha_entries = [
        (path, key, value)
        for path, key, value in _walk(gates)
        if "alpha" in key.lower() or (isinstance(value, str) and "alpha" in value.lower())
    ]
    assert alpha_entries, "expected alpha-rank to be recorded in gates.yaml as Tier 3"
    # Scalar anchors (e.g. alpha_rank_clone_drift_anchor) are measurements,
    # not evaluation methods; only the alpha_rank mapping carries tiering.
    alpha_entries = [
        (path, key, value) for path, key, value in alpha_entries if isinstance(value, dict)
    ]
    assert alpha_entries, "expected an alpha_rank mapping in gates.yaml"
    for path, key, value in alpha_entries:
        full = path + (key,)
        assert full[0] == "gate-p6", f"alpha-rank entry outside gate-p6: {'.'.join(full)}"
        # The tiering record is the alpha_rank mapping itself.
        node = value if key == "alpha_rank" else gates["gate-p6"]["alpha_rank"]
        assert node.get("required") is False, (
            f"alpha-rank entry must carry 'required: false' (Tier 3, off by default): {'.'.join(full)}"
        )
        assert "tier 3" in str(node.get("tier", "")).lower(), (
            f"alpha-rank entry must be labelled Tier 3: {'.'.join(full)}"
        )


@pytest.mark.skipif(not MAKEFILE.exists(), reason="Makefile not written yet")
def test_makefile_exists_for_other_tests() -> None:
    assert GATES_YAML.exists(), "configs/gates.yaml must exist alongside the Makefile"
