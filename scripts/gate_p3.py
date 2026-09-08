"""Phase 3 gate: trainer core on Kuhn and Leduc.

Reads thresholds from configs/gates.yaml (gate-p3). The ladder:
last-iterate exact NashConv below max on both games (pinned recipe+seed,
CPU 1-thread, deterministic). The guards: ablation (same trainer with
regularisation off must do worse on Leduc), estimator (GAE vs
Monte-Carlo, same Leduc budget — GAE must be no worse than noise, and the
winner is recorded as the gin estimator). Determinism (same seed gives
bit-identical checkpoints). RL-BR calibration (final-iterate BR returns vs
a fixed uniform policy recover the exact uniform NashConv within
tolerance on both games — the check that failed before the terminal-credit
fix and passes after). Provenance (run dirs carry config + JSONL record).

`--quick` divides step budgets by 20 and asserts machinery + finiteness
only: convergence needs the full budget, so threshold asserts are
info-only in quick mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from open_spiel.python import policy as policy_lib  # noqa: E402
from open_spiel.python.algorithms import best_response  # noqa: E402

from ginrl.config import Seeds, TrainerConfig  # noqa: E402
from ginrl.eval.exploitability import load_small_game  # noqa: E402
from ginrl.eval.rlbr import (  # noqa: E402
    simulate_return,
    train_br,
    uniform_fixed,
)
from ginrl.nets.actor_critic import MaskedActorCritic  # noqa: E402
from ginrl.train.driver import net_config_for_game  # noqa: E402
from ginrl.train.loop import train_selfplay  # noqa: E402

GATES = yaml.safe_load((ROOT / "configs" / "gates.yaml").read_text())["gate-p3"]
FAILURES: list[str] = []
DEVICE = torch.device("cpu")
SIM_DEALS = 6000


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def exact_br_sum(game_name: str) -> float:
    """Exact uniform-policy NashConv via exhaustive best responses."""
    game = load_small_game(game_name)
    base = policy_lib.UniformRandomPolicy(game)
    total = 0.0
    for seat in (0, 1):
        br = best_response.BestResponsePolicy(game, seat, base)
        total += br.value(game.new_initial_state())
    return total


def ladder_cfg(steps: int, seed: int, **kw: object) -> TrainerConfig:
    base: dict[str, object] = dict(
        n_envs=16,
        rollout_len=128,
        epochs=2,
        minibatches=4,
        lr=3e-4,
        reg_coef=0.1,
        magnet_mode="uniform",
        seeds=Seeds(master=seed),
        eval_every=25,
    )
    base.update(kw)
    return TrainerConfig(total_steps=steps, **base)  # type: ignore[arg-type]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    div = 20 if args.quick else 1
    enforce = not args.quick

    def budget(key: str) -> int:
        return max(4096, int(GATES[key] / div))

    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="gate-p3-"))
    if args.quick:
        print("gate-p3 --quick: budgets divided by 20, thresholds info-only", flush=True)

    # 1. Anchors: exact uniform NashConv matches the filed constants.
    for game, key in (
        ("kuhn_poker", "kuhn_uniform_nash_conv"),
        ("leduc_poker", "leduc_uniform_nash_conv"),
    ):
        exact = exact_br_sum(game)
        check(
            f"anchor_{game}",
            round(exact, 4) == GATES[key],
            f"exact uniform NashConv {exact:.4f} (filed {GATES[key]})",
        )

    # 2/3. Ladders: last-iterate exact NashConv below max within budget.
    kuhn_seed, leduc_seed = GATES["kuhn_seed"], GATES["leduc_seed"]
    kuhn = train_selfplay(
        "kuhn_poker",
        ladder_cfg(budget("kuhn_budget_steps"), kuhn_seed, anneal="linear"),
        kuhn_seed,
        DEVICE,
        tmp / "kuhn",
    )
    check(
        "kuhn_ladder",
        (not enforce) or kuhn.final_nash_conv <= GATES["kuhn_nash_conv_max"],
        f"final NC {kuhn.final_nash_conv:.4f} "
        f"(max {GATES['kuhn_nash_conv_max']}, {budget('kuhn_budget_steps')} steps)",
    )
    leduc_cfg = ladder_cfg(budget("leduc_budget_steps"), leduc_seed)
    leduc = train_selfplay("leduc_poker", leduc_cfg, leduc_seed, DEVICE, tmp / "leduc")
    check(
        "leduc_ladder",
        (not enforce) or leduc.final_nash_conv <= GATES["leduc_nash_conv_max"],
        f"final NC {leduc.final_nash_conv:.4f} "
        f"(max {GATES['leduc_nash_conv_max']}, {budget('leduc_budget_steps')} steps)",
    )

    # 4. Ablation guard: the same trainer with regularisation off must do
    # worse on Leduc, else the regularisation is not wired in.
    control = train_selfplay(
        "leduc_poker",
        ladder_cfg(budget("leduc_budget_steps"), leduc_seed, reg_coef=0.0),
        leduc_seed,
        DEVICE,
        tmp / "leduc-noreg",
    )
    check(
        "ablation",
        (not enforce) or leduc.final_nash_conv < control.final_nash_conv,
        f"reg {leduc.final_nash_conv:.3f} vs no-reg {control.final_nash_conv:.3f}",
    )

    # 5. Estimator guard: GAE vs Monte-Carlo on Leduc, same budget. GAE
    # must be no worse than noise (Kuhn already favours GAE decisively);
    # the winner is recorded as the gin estimator.
    mc = train_selfplay(
        "leduc_poker",
        ladder_cfg(budget("leduc_budget_steps"), leduc_seed, advantage="mc"),
        leduc_seed,
        DEVICE,
        tmp / "leduc-mc",
    )
    gae_min = min(c for _, c in leduc.evals) if leduc.evals else leduc.final_nash_conv
    mc_min = min(c for _, c in mc.evals) if mc.evals else mc.final_nash_conv
    margin = GATES["estimator_noise_margin"]
    winner = "gae" if gae_min <= mc_min else "mc"
    check(
        "estimator",
        (not enforce) or gae_min <= mc_min + margin,
        f"GAE min {gae_min:.3f} vs MC min {mc_min:.3f} (margin {margin}; gin estimator: {winner})",
    )

    # 6. Determinism: same seed, bit-identical checkpoints on CPU.
    tiny = TrainerConfig(
        total_steps=max(4096, 20000 // div),
        n_envs=4,
        rollout_len=32,
        epochs=1,
        minibatches=2,
        seeds=Seeds(master=99),
    )
    da = train_selfplay("kuhn_poker", tiny, 99, DEVICE, tmp / "det-a")
    db = train_selfplay("kuhn_poker", tiny, 99, DEVICE, tmp / "det-b")
    ha = hashlib.sha256((tmp / "det-a" / "checkpoint.pt").read_bytes()).hexdigest()
    hb = hashlib.sha256((tmp / "det-b" / "checkpoint.pt").read_bytes()).hexdigest()
    _ = (da, db)
    check("determinism", ha == hb, f"checkpoint sha {ha[:12]}… equal={ha == hb}")

    # 7. RL-BR calibration: final-iterate BR sums recover exact NC.
    br_specs = (
        (
            "kuhn_poker",
            budget("br_kuhn_steps"),
            64,
            2,
            "rlbr_calibration_tolerance_kuhn",
            "kuhn_uniform_nash_conv",
        ),
        (
            "leduc_poker",
            budget("br_leduc_steps"),
            128,
            2,
            "rlbr_calibration_tolerance_leduc",
            "leduc_uniform_nash_conv",
        ),
    )
    for game, steps, hidden, layers, tol_key, exact_key in br_specs:
        seed = GATES["br_seed"]
        cfg = TrainerConfig(
            total_steps=steps,
            n_envs=64,
            rollout_len=128,
            epochs=2,
            minibatches=4,
            lr=3e-4,
            reg_coef=0.1,
            magnet_mode="uniform",
            seeds=Seeds(master=seed),
        )
        rets = []
        for seat in (0, 1):
            net, _ = train_br(
                game,
                uniform_fixed,
                seat,
                cfg,
                seed,
                DEVICE,
                make_net=lambda game=game, hidden=hidden, layers=layers: MaskedActorCritic(
                    net_config_for_game(game, hidden, layers)
                ),
            )
            rets.append(
                simulate_return(
                    load_small_game(game),
                    net,
                    seat,
                    uniform_fixed,
                    max(100, SIM_DEALS // div),
                    seed + seat,
                )
            )
        gap = abs(sum(rets) - GATES[exact_key])
        check(
            f"rlbr_{game}",
            (not enforce) or gap <= GATES[tol_key],
            f"BR sum {sum(rets):.4f} vs exact {GATES[exact_key]} "
            f"(gap {gap:.4f}, tol {GATES[tol_key]})",
        )

    # 8. Provenance: run dirs carry config + the JSONL record with a header.
    ok_prov = True
    for run in ("kuhn", "leduc"):
        cfg_path, rec_path = tmp / run / "config.json", tmp / run / "metrics.jsonl"
        kinds: list[str] = []
        if cfg_path.is_file() and rec_path.is_file():
            kinds = [
                json.loads(line).get("type", "?") for line in rec_path.read_text().splitlines()
            ]
        ok_prov = ok_prov and kinds[:1] == ["header"] and "scalar" in kinds
    check("provenance", ok_prov, "config.json + header-led metrics.jsonl in run dirs")

    print(f"gate-p3 finished in {time.time() - t0:.0f}s", flush=True)
    if FAILURES:
        print(f"FAILURES: {FAILURES}", flush=True)
        return 1
    print("gate-p3: all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
