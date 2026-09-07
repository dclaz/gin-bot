"""Environment probe (Phase 0).

Interrogates the live engine and hardware, writes the generated facts:
  docs/ENV_FACTS.md (human) and src/ginrl/spiel_facts.py (machine).

`--check` re-probes and diffs the stable subset (engine facts, versions)
against the committed files; benchmark timings, the device recommendation
and timestamps are volatile and excluded from the comparison.

Usage:
  uv run python scripts/probe_env.py
  uv run python scripts/probe_env.py --check
"""

from __future__ import annotations

import argparse
import datetime
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FACTS_MD = ROOT / "docs" / "ENV_FACTS.md"
SPIEL_FACTS_PY = ROOT / "src" / "ginrl" / "spiel_facts.py"

# Module-level names in spiel_facts.py excluded from drift comparison.
# Engine facts are stable across machines; torch build flags, MPS presence,
# benchmark timings and the device recommendation are hardware-dependent.
VOLATILE_PY_PREFIXES = (
    "PROBE_TIMESTAMP",
    "DEVICE_RECOMMENDATION",
    "BENCHMARK_",
    "TORCH_VERSION",
    "MPS_BUILT",
    "MPS_AVAILABLE",
)


def _stable_lines(path: Path, volatile_prefixes: tuple[str, ...]) -> list[str]:
    return [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(volatile_prefixes) and not line.startswith("# Generated")
    ]


def _stable_md_lines(path: Path) -> list[str]:
    """Only the '## Gin rummy game facts' section.

    The platform section (OS, CPU topology, torch build) varies by machine
    and the benchmark section varies run to run; neither is drift.
    """
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("## Gin rummy"))
        end = next(i for i, line in enumerate(lines) if line.startswith("## Microbenchmark"))
    except StopIteration:
        return []
    return lines[start:end]


def cpu_topology() -> dict[str, int | None]:
    def sysctl(key: str) -> int | None:
        try:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip()) if out.returncode == 0 else None
        except Exception:
            return None

    import os

    return {
        "perf_cores": sysctl("hw.perflevel0.logicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.logicalcpu"),
        "logical_cpus": os.cpu_count(),
    }


def torch_facts() -> dict[str, object]:
    import torch

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available() if mps_built else False
    bf16_ok = False
    if mps_available:
        try:
            a = torch.randn(8, 8, dtype=torch.bfloat16, device="mps")
            (a @ a).cpu()
            bf16_ok = True
        except Exception:
            bf16_ok = False
    return {
        "version": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_bf16_matmul_ok": bf16_ok,
    }


def engine_facts() -> dict[str, object]:
    import pyspiel
    from pyspiel import gin_rummy as gr

    game = pyspiel.load_game("gin_rummy")
    facts: dict[str, object] = {
        "open_spiel_version": "2.0.2",
        "num_players": game.num_players(),
        "num_distinct_actions": game.num_distinct_actions(),
        "observation_tensor_size": game.observation_tensor_size(),
        "max_game_length": game.max_game_length(),
        "min_utility": game.min_utility(),
        "max_utility": game.max_utility(),
        "params": dict(game.get_parameters()),
        "draw_upcard_action": gr.DRAW_UPCARD_ACTION,
        "draw_stock_action": gr.DRAW_STOCK_ACTION,
        "pass_action": gr.PASS_ACTION,
        "knock_action": gr.KNOCK_ACTION,
        "meld_action_base": gr.MELD_ACTION_BASE,
        "num_meld_actions": gr.NUM_MELD_ACTIONS,
        "wall_stock_size": gr.WALL_STOCK_SIZE,
        "default_gin_bonus": gr.DEFAULT_GIN_BONUS,
        "default_undercut_bonus": gr.DEFAULT_UNDERCUT_BONUS,
        "default_knock_card": gr.DEFAULT_KNOCK_CARD,
        "default_hand_size": gr.DEFAULT_HAND_SIZE,
        "default_num_ranks": gr.DEFAULT_NUM_RANKS,
        "default_num_suits": gr.DEFAULT_NUM_SUITS,
    }

    rng = random.Random(0)
    state = game.new_initial_state()
    while state.is_chance_node():
        outs, probs = zip(*state.chance_outcomes(), strict=True)
        state.apply_action(rng.choices(outs, list(probs))[0])

    try:
        state.information_state_tensor(0)
        facts["information_state_tensor_raises"] = False
    except Exception:
        facts["information_state_tensor_raises"] = True

    utils = gr.GinRummyUtils(13, 4, 10)
    facts["gin_rummy_utils_methods"] = sorted(
        m
        for m in (
            "min_deadwood",
            "best_meld_group",
            "all_melds",
            "legal_melds",
            "legal_discards",
            "all_layoffs",
            "meld_to_int",
            "int_to_meld",
            "card_value",
        )
        if hasattr(utils, m)
    )
    facts["to_dict_keys"] = sorted(state.to_dict().keys())
    obs_struct = state.to_observation_struct(0).to_dict()
    facts["observation_struct_keys"] = sorted(obs_struct.keys())

    try:
        resampled = state.resample_from_infostate(0, lambda: rng.random())
        facts["resample_from_infostate_works"] = resampled is not None
    except Exception:
        facts["resample_from_infostate_works"] = False

    try:
        pyspiel.make_simple_gin_rummy_bot(game.get_parameters(), 0)
        facts["simple_bot_constructs"] = True
    except Exception:
        facts["simple_bot_constructs"] = False

    action_strings = {}
    for action in range(game.num_distinct_actions()):
        action_strings[action] = state.action_to_string(action)
    facts["num_action_strings"] = len(action_strings)
    facts["action_strings"] = action_strings
    return facts


def raw_stepping_benchmark(steps_budget: int = 20000) -> dict[str, float]:
    """Raw env stepping: step + observation + legal mask, no features."""
    import pyspiel

    game = pyspiel.load_game("gin_rummy")
    rng = random.Random(7)
    steps = 0
    start = time.perf_counter()
    while steps < steps_budget:
        state = game.new_initial_state()
        while not state.is_terminal() and steps < steps_budget:
            if state.is_chance_node():
                outs, probs = zip(*state.chance_outcomes(), strict=True)
                state.apply_action(rng.choices(outs, list(probs))[0])
            else:
                state.observation_tensor(state.current_player())
                legal = state.legal_actions()
                state.apply_action(rng.choice(legal))
            steps += 1
    elapsed = time.perf_counter() - start
    return {"steps": float(steps), "seconds": elapsed, "steps_per_sec": steps / elapsed}


def device_benchmark() -> dict[str, object]:
    """Representative-MLP stand-in at configured batch sizes (Phase 0).

    The real torso does not exist until Phase 3; re-run the device decision
    per architecture in Phase 4 and at full size in Phase 5 (landmine 9).
    """
    import torch

    torch.manual_seed(0)
    torso = torch.nn.Sequential(
        torch.nn.Linear(644, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
    )
    batch_sizes = [32, 256]
    per_device: dict[str, dict[str, float]] = {}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        net = torso.to(device).float().train()
        times: dict[str, float] = {}
        for batch in batch_sizes:
            x = torch.randn(batch, 644, device=device)
            for _ in range(5):  # warmup
                net.zero_grad()
                net(x).sum().backward()
            if device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(15):
                net.zero_grad()
                net(x).sum().backward()
            if device == "mps":
                torch.mps.synchronize()
            times[f"batch_{batch}_ms_per_iter"] = (time.perf_counter() - start) / 15 * 1000
        per_device[device] = times
    totals = {d: sum(t.values()) for d, t in per_device.items()}
    recommendation = min(totals, key=lambda d: totals[d])
    return {"per_device_ms_per_iter": per_device, "recommendation": recommendation}


def collect_all() -> dict[str, object]:
    return {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": cpu_topology(),
        "torch": torch_facts(),
        "engine": engine_facts(),
        "raw_stepping": raw_stepping_benchmark(),
        "device_benchmark": device_benchmark(),
    }


def _lit(value: object) -> str:
    """Python literal rendered the way `ruff format` would (double quotes)."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list):
        # Multi-line with a magic trailing comma: ruff format leaves it alone.
        return "[\n" + "".join(f"    {_lit(v)},\n" for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in value.items()) + "}"
    return repr(value)


def render_spiel_facts_py(facts: dict[str, object]) -> str:
    eng = facts["engine"]
    assert isinstance(eng, dict)
    torch_info = facts["torch"]
    assert isinstance(torch_info, dict)
    bench = facts["device_benchmark"]
    assert isinstance(bench, dict)
    raw = facts["raw_stepping"]
    lines = [
        "# Generated by scripts/probe_env.py — never hand-edit. Run `make probe`.",
        f"PROBE_TIMESTAMP = {_lit(facts['timestamp'])}",
        "",
        f"OPEN_SPIEL_VERSION = {_lit(eng['open_spiel_version'])}",
        f"NUM_PLAYERS = {_lit(eng['num_players'])}",
        f"NUM_DISTINCT_ACTIONS = {_lit(eng['num_distinct_actions'])}",
        f"OBSERVATION_TENSOR_SIZE = {_lit(eng['observation_tensor_size'])}",
        f"MAX_GAME_LENGTH = {_lit(eng['max_game_length'])}",
        f"MIN_UTILITY = {_lit(eng['min_utility'])}",
        f"MAX_UTILITY = {_lit(eng['max_utility'])}",
        "",
        "GAME_PARAMS = {",
    ]
    params = eng["params"]
    assert isinstance(params, dict)
    for key in sorted(params):
        lines.append(f"    {_lit(key)}: {_lit(params[key])},")
    lines.append("}")
    for name in (
        "draw_upcard_action",
        "draw_stock_action",
        "pass_action",
        "knock_action",
        "meld_action_base",
        "num_meld_actions",
        "wall_stock_size",
    ):
        lines.append(f"{name.upper()} = {_lit(eng[name])}")
    lines += [
        "",
        f"INFORMATION_STATE_TENSOR_RAISES = {_lit(eng['information_state_tensor_raises'])}",
        f"GIN_RUMMY_UTILS_METHODS = {_lit(sorted(eng['gin_rummy_utils_methods']))}",
        f"RESAMPLE_FROM_INFOSTATE_WORKS = {_lit(eng['resample_from_infostate_works'])}",
        f"SIMPLE_BOT_CONSTRUCTS = {_lit(eng['simple_bot_constructs'])}",
        "",
        f"TORCH_VERSION = {_lit(torch_info['version'])}",
        f"MPS_BUILT = {_lit(torch_info['mps_built'])}",
        f"MPS_AVAILABLE = {_lit(torch_info['mps_available'])}",
        "",
        f"DEVICE_RECOMMENDATION = {_lit(bench['recommendation'])}",
        'BENCHMARK_STANDIN = "mlp-644-512-512-256-fp32-fwd-bwd"',
    ]
    bench_times = bench["per_device_ms_per_iter"]
    assert isinstance(bench_times, dict)
    for device in sorted(bench_times):
        for key in sorted(bench_times[device]):
            lines.append(
                f"BENCHMARK_{device.upper()}_{key.upper()} = {_lit(round(bench_times[device][key], 4))}"
            )
    assert isinstance(raw, dict)
    lines.append(f"BENCHMARK_RAW_STEPS_PER_SEC = {_lit(round(raw['steps_per_sec'], 1))}")
    lines.append("")
    return "\n".join(lines)


def render_env_facts_md(facts: dict[str, object]) -> str:
    eng = facts["engine"]
    assert isinstance(eng, dict)
    torch_info = facts["torch"]
    assert isinstance(torch_info, dict)
    bench = facts["device_benchmark"]
    assert isinstance(bench, dict)
    raw = facts["raw_stepping"]
    assert isinstance(raw, dict)
    params = eng["params"]
    assert isinstance(params, dict)
    out = [
        "# Environment facts",
        "",
        "Generated by `scripts/probe_env.py` — do not hand-edit. Machine twin: "
        "`src/ginrl/spiel_facts.py`.",
        "",
        f"Probed at: {facts['timestamp']}",
        "",
        "## Platform",
        "",
        f"- OS: {facts['platform']} ({facts['machine']})",
        f"- Python: {facts['python']}",
        f"- CPU perf cores: {facts['cpu']['perf_cores']}, "
        f"efficiency cores: {facts['cpu']['efficiency_cores']}, "
        f"logical: {facts['cpu']['logical_cpus']}",
        f"- torch {torch_info['version']}: MPS built={torch_info['mps_built']}, "
        f"available={torch_info['mps_available']}, "
        f"bf16 matmul={torch_info['mps_bf16_matmul_ok']}",
        f"- open_spiel {eng['open_spiel_version']}",
        "",
        "## Gin rummy game facts",
        "",
        f"- players={eng['num_players']}, actions={eng['num_distinct_actions']}, "
        f"observation dims={eng['observation_tensor_size']}, "
        f"max length={eng['max_game_length']}, utility [{eng['min_utility']}, {eng['max_utility']}]",
        "- params: " + ", ".join(f"{k}={params[k]}" for k in sorted(params)),
        f"- draw_upcard={eng['draw_upcard_action']}, draw_stock={eng['draw_stock_action']}, "
        f"pass={eng['pass_action']}, knock={eng['knock_action']}, "
        f"meld_base={eng['meld_action_base']}, melds={eng['num_meld_actions']}, "
        f"wall_stock={eng['wall_stock_size']}",
        f"- information_state_tensor raises: {eng['information_state_tensor_raises']} "
        "(only observation_tensor exists)",
        f"- GinRummyUtils methods: {', '.join(sorted(eng['gin_rummy_utils_methods']))}",
        f"- resample_from_infostate works: {eng['resample_from_infostate_works']}",
        f"- make_simple_gin_rummy_bot constructs: {eng['simple_bot_constructs']}",
        f"- action strings recorded: {eng['num_action_strings']}",
        "",
        "## Microbenchmark",
        "",
        "Stand-in MLP (644-512-512-256, fp32 fwd+bwd); the real torso lands in "
        "Phase 3, so re-run the device decision per architecture (landmine 9).",
        "",
    ]
    bench_times = bench["per_device_ms_per_iter"]
    assert isinstance(bench_times, dict)
    for device in sorted(bench_times):
        for key in sorted(bench_times[device]):
            out.append(f"- {device} {key}: {bench_times[device][key]:.2f} ms/iter")
    out.append(f"- raw stepping: {raw['steps_per_sec']:.0f} steps/s (no features)")
    out.append(f"- device recommendation: **{bench['recommendation']}**")
    out.append("")
    return "\n".join(out)


def write_outputs(facts: dict[str, object]) -> None:
    SPIEL_FACTS_PY.parent.mkdir(parents=True, exist_ok=True)
    (SPIEL_FACTS_PY.parent / "__init__.py").touch(exist_ok=True)
    SPIEL_FACTS_PY.write_text(render_spiel_facts_py(facts))
    ENV_FACTS_MD.write_text(render_env_facts_md(facts))


def check() -> int:
    """Regenerate to a temp dir and diff the stable subset against the tree."""
    import tempfile

    if not SPIEL_FACTS_PY.exists() or not ENV_FACTS_MD.exists():
        print("facts-check FAIL: generated files missing; run `make probe` first")
        return 1
    committed_py = _stable_lines(SPIEL_FACTS_PY, VOLATILE_PY_PREFIXES)
    committed_md = _stable_md_lines(ENV_FACTS_MD)

    facts = collect_all()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_py = Path(tmp) / "spiel_facts.py"
        tmp_md = Path(tmp) / "ENV_FACTS.md"
        tmp_py.write_text(render_spiel_facts_py(facts))
        tmp_md.write_text(render_env_facts_md(facts))
        fresh_py = _stable_lines(tmp_py, VOLATILE_PY_PREFIXES)
        fresh_md = _stable_md_lines(tmp_md)

    failures = 0
    if committed_py != fresh_py:
        failures += 1
        print("facts-check FAIL: src/ginrl/spiel_facts.py drifted from the live engine:")
        for line in fresh_py:
            if line not in committed_py:
                print(f"  live-only: {line}")
        for line in committed_py:
            if line not in fresh_py:
                print(f"  file-only: {line}")
    if committed_md != fresh_md:
        failures += 1
        print("facts-check FAIL: docs/ENV_FACTS.md drifted from the live engine.")
    if failures == 0:
        print("facts-check OK: generated facts match the live engine")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe env/engine, write generated facts.")
    parser.add_argument("--check", action="store_true", help="diff live facts vs committed files")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    facts = collect_all()
    write_outputs(facts)
    print(f"wrote {ENV_FACTS_MD} and {SPIEL_FACTS_PY}")
    print(f"device recommendation: {facts['device_benchmark']['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
